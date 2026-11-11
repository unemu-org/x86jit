# Dense JIT code memory — decision-17 phases 2 and 3

Implementation and validation on 2026-11-10. Phase 1 baseline: `c161085`.
**MEASURED** means executed locally; **READ** means checked in source or API
contracts; **ASSUMED** identifies platform behavior still requiring native coverage.

**Current contract:** [Final phase 3 ownership and maintenance](#final-phase-3-ownership-and-maintenance)
supersedes the incremental boundaries and remaining-work lists in the historical
handoffs below. Phase 4 stress/platform validation belongs to `xj-2jb.5`.

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
sealed/committed boundaries remain available to reclamation. In strict anonymous
mode, a sealed page must never become RW again within the same reservation,
including after both grace periods. New writes use only the RW frontier. The
reclamation policy is to release a wholly free anonymous region and obtain a
fresh reservation; sealed holes in a partially live region cannot become reusable
extents. Decommit alone does not authorize changing their protection to RW.
Neither code-page reuse nor this empty-region release is implemented yet.

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

## Phase 3 lifetime protocol

**Design; implementation status is recorded below.** Retirement must make the
cache entry unavailable and clear chain/IBTC slots under the cache's publication
locks, then advance a retirement epoch. Holding the exclusive publication locks
across the retirement hook and map mutation establishes that ordering: a new
resolver cannot observe the old map after validating the new epoch. Each vCPU
must announce execution participation
**before resolving any compiled pointer**, recheck the epoch, and retain that
participation through compiled calls, chain transfers and cold fault handling.
At a dispatcher boundary it discards local fast entries and return predictions
when the epoch changes, releases participation, and resolves again on reentry.
An idle or blocked vCPU has no participation and cannot pin retirement; a vCPU
blocked inside a compiled helper must retain participation until it has discarded
every pointer into the unit. Epoch publication, participation and validation
initially use SeqCst operations so the writer cannot miss a reader that can still
enter the retired generation. Invalidation, tier replacement, rejected completion
and teardown must all transfer allocation ownership to this retirement path.
Direct compiled-ABI entry users need the same lifetime protection as VM dispatch;
the current copyable `CompiledPtr` is not an execution-participation handle.

Execution grace alone does not authorize address reuse. After it completes,
remove the unit's code-map range. Signal readers select one of two counters:
load the active counter index, increment it, validate that the index is
unchanged, then load an immutable root. On the first validation failure, retain
that counter, increment the other counter, then load the root without further
validation. A guard retains its counter (both on fallback) and root until it has
copied its result. Every index, counter and root operation is SeqCst. The writer
publishes the replacement root before switching the active index, moves obsolete
roots into a draining batch, and waits only for the previous counter. Further
publications accumulate separately; the writer cannot switch again until the
draining batch is released. It never waits under the writer mutex: later
publications or cold maintenance poll progress.

In the SeqCst order, a validated reader that can load an obsolete root
incremented the old counter before the switch. A fallback reader loads no root
until both increments complete; after that point no batch can drain either
counter until the reader releases it. A writer that observed zero before an
increment can only free roots already superseded before the fallback reader's
root load. The writer always publishes a replacement root before inspecting
counters for reclamation. A reader delayed before increment/validation can join
a later reuse of the same counter index, but loads its root only after
successful validation or both increments, so it cannot resurrect an earlier
root. Roots retired after a switch remain pending until the preceding batch
drains and the next switch covers their readers. Thus continuously arriving
validated new readers do not extend the old grace period. A fallback reader can
briefly pin both counters, trading a bounded number of reader-entry operations
for additional writer delay. A nested signal can read while a writer is
interrupted, including during destruction of old roots: readers never lock,
allocate, free, use TLS or change reference counts. Entry performs one index
load, one validation, at most two increments and one root load; publication
cannot force a validation retry. Atomic RMWs can retry as LL/SC loops, so no
bounded wall-clock reader-entry latency is claimed. Nonlocal jumps abandoning a
guard remain unsupported.

After **both** grace periods and range removal, release the retired generation's
trap/source/instruction tables, unwind recipes and module records before
returning its allocation extent to a coalescing free list.
`unregister_with_completion` returns a map-borrowing `RangeRemoval` token naming
the removal's publication version; `unregister` retains its boolean
compatibility API. Each draining batch records the highest publication version
it covers. Completion advances only after all roots in that batch have been
destroyed, and polling prunes shared metadata pools before reporting completion.
Pending roots cannot complete until the preceding batch and their own batch have
drained. Later publications/readers cannot revoke completion or make a token
describe another removal at the same address. Tokens require no per-removal
registry or allocation and reject an inherited process identity before locking.
A token establishes only signal-reader grace; execution quiescence and compiler
ownership remain caller obligations. Every allocation/reuse needs a fresh
generation carried by observer publish/retire/release events. Old handles cannot
release a new generation. Only wholly free pages may be decommitted, including
backing shared by both alias views; empty Linux/FreeBSD regions may be unmapped.
macOS retains its single reservation and its existing non-inherited fork
contract. Live code never moves. The existing instruction-publication
generation/ISB protocol remains necessary when reclaimed storage is written
again; it supplies neither lifetime grace.

**Step 2 boundary:** the two-counter signal-reader implementation and its witnesses
are the first executable prerequisite. Per-unit handles, execution participation,
retirement notifications, completion tokens, module-table removal, allocator
reuse/decommit and embedder counters/events remain subsequent steps. Until those
are implemented together, invalidation must continue retaining executable storage
and must not start calling `unregister` routinely.

### Step 2 validation and handoff (2026-11-10)

**MEASURED, macOS arm64:** tests used Rust 1.98.1, offline Cargo, an empty
`RUSTC_WRAPPER`, the isolated target directory, at most two build jobs and
`--test-threads 2`. Each test command was wrapped in `timeout -k 5 180`;
none timed out. No full suite or benchmark was run. The single code-memory bound
invocation passed its 100,000-unit test in 45.946 seconds.

| Check | Result |
|---|---|
| `cargo nextest run --offline -p x86jit-core --lib` | 247 passed |
| `cargo nextest run --offline -p x86jit-cranelift --lib` | 127 passed, four skipped |
| Backend `code_memory_bound` and `memory_stats` binaries | Five passed |
| `region_fault_state`, with Unicorn | 26 passed |
| `fault_path`, with Unicorn | 16 passed |
| `fault_metadata`, with Unicorn | 59 passed |
| `bg_tier`, with Unicorn | 24 passed |
| `smc`, with Unicorn | 57 passed |
| Rust 1.95 offline clippy, all targets/features, `-D warnings` | Passed |
| Rust 1.95 offline fmt check; diff whitespace check | Passed |
| Offline workspace/tests check, x86-64 Linux, Nix Rust 1.96.1 | Passed |
| Offline workspace/tests check, AArch64 Linux, Nix Rust 1.96.1 | Passed |

The existing direct-rustc landing-symbol fixture initially failed with the host
linker; selecting the Nix linker in PATH made the complete core library run pass.
The backend arena child/grandchild fork witness and the JIT shutdown/fork witness
passed. The five new code-map witnesses cover continuous reader overlap, every
reader-entry boundary, delayed selection across multiple switches, pending roots
seen by an older counter, and nested lookups at writer switch boundaries.
Restoring the global-zero prerequisite made the overlap witness fail (192 node
bytes retained versus 48 expected). The intended source was restored and all 25
code-map tests passed again.

**READ:** only the signal-reader prerequisite is implemented. The current
copyable entry pointers and append-only compiler records still require the
execution/ownership work above before code storage can be reclaimed. Steps 3–6
remain: handles/generations and all retirement triggers/predictor flushes;
execution grace and range-removal completion tokens; bounded module records and
metadata lifetimes; coalescing/reuse/decommit/empty-region release; compatible
embedder counters/events; and end-to-end retirement/reuse witnesses. In
particular the passing bound test measures retained dense storage, not bounded
retire/recompile storage. **ASSUMED:** native Linux/FreeBSD behavior remains
unmeasured here; Linux cross-checks establish compilation only.

### Step 3a: retirement handoff and execution participation (2026-11-10)

**READ — implemented boundary, not completed phase 3.** `code_lifetime.rs` adds a
backend-owned domain shared with its VM cache through `Backend::code_lifetime`.
Wrapping backends must forward that hook. Finalized, successfully registered
units receive process-wide monotonically increasing `CodeUnit` identities;
`RetiredCodeUnit` carries an immutable identity and retirement epoch. These are
published-unit identities, **not yet releasable provider extent handles**. Failed
or unpublished definitions that never registered still belong to their module.
The raw `CompiledPtr` and `CachedBlock` representation is unchanged.

The cache counts publication ownership, including one unit published under
multiple keys. Replacement, invalidation and cache destruction transfer its last
publication to the retired queue. Rejected foreground/background publications
discard only units without another publication. Backend shutdown joins the worker
and retires discarded ready completions; backend destruction retires every
remaining registered unit after joining. Calling `Backend::shutdown` alone does
not retire cached live units: the backend can still exist afterward. Teardown
retirement requires excluding future pointer resolution, as documented on the
unsafe `retire_all` entry point.

Shared chain/IBTC slots now have stable `Arc<AtomicU64>` storage, with weak
registrations in the domain. Retirement clears them without acquiring compiler
locks while the cache publication locks are held, then advances the SeqCst epoch.
Range invalidation and destruction batch that work: they clear the slot registry
once per batch, not once per victim. The lock order is cache publication locks
then domain registry; the registry never acquires cache or compiler locks.
Compiler slot registration can acquire the registry while holding a module lock.
Slot fills validate the epoch and store under this same registry lock, including
when multiple caches share a domain. This prevents a fill from racing another
cache's retirement/clear. The existing `publish_if_current` closure runs under
these publication locks and must not reenter the engine. No observer or fetch
policy is invoked under the registry lock.

Each vCPU registers an execution participant at creation; dispatch reuses that
record without allocating on the successful run path. Switching to a different
VM/backend registers a new domain-local record. The dispatcher announces its
epoch before probing or resolving executable pointers. Entry stores
and validates the SeqCst epoch, retrying when retirement intervenes. An accepted
participant that predates retirement pins that generation through compiled calls,
chain targets, helpers, signal recovery and cold fault decoding. Each outer-loop
boundary drops its guard; an interpreted block releases participation before
execution. Returns and unwinding also release through RAII. Idle vCPUs have no
active announcement. A blocked compiled helper retains participation because its
return PC is still inside the unit. Later participants cannot prolong an older
ticket's execution grace.

Chain transfers check the domain epoch before following the next pointer and
return to dispatch on change. Dispatch flushes local fast entries, IBTC refill
bookkeeping and return-stack predictions before resolving in the new epoch.
Epoch values are domain-local: dispatch also flushes predictions when a vCPU is
used with another VM/backend, even if their epoch numbers happen to be equal.
Direct compiled-ABI users have a public participant/guard API and must announce
before resolution, discard old predictions and retain the guard through cold
decoding. A copied raw pointer does not retain its generation.

**Retention boundary:** execution-complete tickets authorize no release. This
change calls neither `codemap::unregister` nor provider `free_memory` on ordinary
retirement. Code-map source/trap/instruction tables and unwind recipes, module
declarations/blobs/symbols, slots, IBTC descriptors, executable extents and the
retired-ticket registry remain retained. Existing `MemoryStats` counters and the
compiled-code observer retain their phase-2 meanings. Signal-reader code and the
macOS arena/fork contract are unchanged. Retire/recompile storage is still
unbounded; the dense allocation bound does not establish reclamation.

**Remaining work, in dependency order:**

1. Finish step 3 ownership/admission: bind each registered identity to a provider
   allocation handle and compiler/module records, including pending/failed
   definitions. Carry generations through entry publication/admission before
   reuse; raw-pointer copies must not resurrect a retired/reused generation.
   Audit direct ABI users and wrapping backends when enabling actual release.
2. Add range-removal completion tickets to the existing two-counter code map;
   poll execution grace, remove ranges, then wait for the specific signal-reader
   batch. Retire module declarations, blobs, relocation/symbol references, slots,
   descriptors and all associated metadata before returning an extent. Bounded
   maintenance must consume completed tickets rather than retain them forever.
3. Step 4: coalescing free lists, aligned reuse, shared-backing page decommit and
   empty Linux/FreeBSD region release; macOS keeps its reservation and decommits
   wholly free pages. Include allocation/publication failure cleanup. Never move
   live code.
4. Step 5: compatible live/retired byte and unit gauges, reusable free bytes, and
   publish/retire/release observer events with allocation generation IDs.
5. Step 6: end-to-end witnesses that each trigger both retires **and reclaims**;
   bounded retire/recompile cycles that fail on the retained-storage baseline;
   held vCPU/signal-reader witnesses preventing extent reuse, reused-generation
   rejection, live fault classification during reclamation and platform/fork
   coverage. Current witnesses establish retirement/execution grace only.

**MEASURED, macOS arm64:** offline Cargo, empty `RUSTC_WRAPPER`, isolated target
directory, two build jobs and at most two nextest threads. Every test command
used `timeout -k 5 180`; no command timed out. No full suite or benchmark ran.
The dense bound binary ran once (two tests, including the 100,000-unit case,
33.896 seconds total). It still measures retained storage, not reclamation.

| Selected check | Result |
|---|---|
| Core library | 253 passed |
| Backend library | 127 passed, four skipped |
| Backend `memory_stats` binary | Seven passed |
| Backend `code_memory_bound` binary, single invocation | Two passed |
| `region_fault_state`, with Unicorn | 26 passed |
| `fault_path`, with Unicorn | 16 passed |
| `fault_metadata`, with Unicorn | 59 passed |
| `bg_tier`, with Unicorn | 24 passed |
| `smc`, with Unicorn | 57 passed |
| Rust 1.95 offline all-target/all-feature clippy, `-D warnings` | Passed |
| Rust 1.95 offline fmt check; diff whitespace check | Passed |
| Offline workspace/tests check, x86-64 Linux, Nix Rust 1.96.1 | Passed |
| Offline workspace/tests check, AArch64 Linux, Nix Rust 1.96.1 | Passed |

Six new core witnesses cover participant entry/retirement interleavings, old
versus new/idle participants, unwinding, last-publication ownership, replacement
and rejection, and teardown after a publication callback panics. Four added
backend witnesses exercise a compiled-to-compiled tier replacement, backend
destruction, a vCPU switching domains with equal epoch numbers, and a vCPU
blocked inside a compiled helper. The existing invalidation and rejected
background-completion fixtures now assert their retirement tickets too.

The helper witness failed as intended when execution announcements were
temporarily disabled: execution grace incorrectly appeared complete while its
return PC remained inside the retired unit. The intended source was restored
and all seven backend memory-statistics tests passed. The first fault-metadata
run caught one successful-run allocation from registering a participant per run;
registration moved to vCPU construction, and all 59 tests then passed, including
the allocation-free success-path witnesses. Registry snapshots/maintenance are
cold operations; dispatch announcement/validation does not allocate or lock.

Linux cross-checks used the Nix Rust 1.96.1 compiler and matching sysroot.
Native tests used Rust 1.98.1. That toolchain emits existing libc-declaration
signature warnings; the controlling Rust 1.95 clippy gate passes without warning
allowances or unrelated declaration changes. **ASSUMED:** Linux/FreeBSD native
execution and native x86-64 execution remain unmeasured; compile checks alone
cannot establish their runtime grace/fault/fork behavior.


### Step 3b: bounded signal entry and range-removal completion (2026-11-10)

**READ — consistent prerequisite boundary; phase 3 remains incomplete.**
`CodeMap::read_probed` performs one validation. On failure it retains the selected
counter, increments the other counter, then loads a root. The guard releases both
counters after copying its result. Writer publications cannot force an entry
retry; individual atomic RMWs may still retry on LL/SC hosts. The classifier's
signal-safety documentation and the protocol proof above describe that distinction.

`unregister_with_completion` preserves the existing boolean `unregister` API and
returns a `RangeRemoval` borrowing the map. Monotonic publication versions label
removals and draining batches; a completed-through watermark advances only after
batch destruction. Polling first reclaims eligible roots and prunes metadata
pools, then compares the token's version to the watermark. Pending publications
behind an older batch must pass their own counter drain. There is no per-token
registry, allocation or retained root, and dropping a token does not retain
metadata. A later removal at the same address cannot reuse the token's identity.
The token checks its process identity before acquiring the writer mutex; an
inherited token cannot touch a parent lock. Completion establishes signal-reader
grace only. Callers must establish execution quiescence before removal and still
own compiler/module and provider records through release.

Four witnesses cover forced validation failure and continued writer publication
at each fallback boundary, pending roots across two reader batches and later
address generations, metadata/pool release across 256 cycles with constant writer
bookkeeping, and inherited-token rejection while the writer mutex is held. The
bounded-entry witness also holds both counters through removal and checks that
completion remains false until the guard drops. The repeated-removal witness
covers source/trap tables, recipes and instruction descriptors. These are code-map
metadata witnesses, not executable-extent reuse or retire/recompile bounds.

**Remaining work, in dependency order:**

1. Bind published identities to generation-bearing provider extent handles and
   compiler records, including failed/pending definitions. Carry generations
   through cache publication/admission, direct ABI entry and wrapping backends;
   copied address-only entries must not resurrect a reused generation.
2. Wire execution-complete retirement tickets to range removal and the new
   completion tokens. Retire module declarations, blobs, relocation/symbol
   references, slots, IBTC descriptors and metadata before returning extents.
   Consume completed retirement tickets with bounded maintenance; the current
   retirement registry is still append-only. Ordinary retirement still calls
   neither `unregister` nor provider `free_memory`.
3. Add coalescing free lists, aligned reuse, failed-publication cleanup, shared
   backing decommit for wholly free pages and empty Linux/FreeBSD region release.
   macOS must retain its single non-inherited reservation and decommit free pages.
   Preserve live addresses and the cache/pipeline publication protocol on reuse.
4. Add compatible live/retired byte and unit gauges, reusable free bytes, and
   observer publish/retire/release events carrying allocation generation IDs.
5. Add end-to-end actual-reclamation witnesses for invalidation, tier replacement,
   rejected completion and teardown; held-vCPU/reader extent-reuse exclusion;
   stale-generation admission rejection; bounded retire/recompile code/module
   storage that fails on the retained-storage baseline; live fault classification
   during reclamation; and native platform/fork validation.

**READ:** provider storage, Cranelift records and predictor representations are
unchanged at this boundary. Releasing code with the current address-only
`CompiledPtr`/cache admission and append-only compiler/provider records would
permit stale-generation resurrection. Completion tokens alone must not enable
reuse. The existing macOS arena/fork contract remains in force.

**MEASURED, macOS arm64:** all Cargo builds used offline mode, an empty compiler
wrapper, the isolated target directory and two build jobs. Every test command
used `timeout -k 5 180` and `--test-threads 2`; no command timed out. No full suite
or benchmark ran. The dense bound binary ran once; its 100,000-unit case finished
in 38.456 seconds. It continues to measure retained dense storage, not reclamation.

| Selected check | Result |
|---|---|
| Core library | 257 passed |
| Backend library | 127 passed, four skipped |
| Backend `memory_stats` | Seven passed |
| Backend `code_memory_bound`, single invocation | Two passed |
| `region_fault_state`, with Unicorn | 26 passed |
| `fault_path`, with Unicorn | 16 passed |
| `fault_metadata`, with Unicorn | 59 passed |
| `bg_tier`, with Unicorn | 24 passed |
| `smc`, with Unicorn | 57 passed |
| Rust 1.95 offline all-target/all-feature clippy, `-D warnings` | Passed |
| Formatting and diff whitespace checks | Passed |
| Offline workspace/tests check, x86-64 Linux, Nix Rust 1.96.1 | Passed |
| Offline workspace/tests check, AArch64 Linux, Nix Rust 1.96.1 | Passed |

Restoring the original validation retry loop made the new entry-bound witness
fail on the second selection attempt (two attempts versus one). The intended
source was restored, and the final core library run passed all 257 tests. The
four new witnesses also passed in the focused 29-test code-map run; the final
library run additionally covers the strengthened fallback-completion and
instruction-descriptor assertions. The arena child/grandchild rebuild and JIT
shutdown/fork witnesses passed. The first focused build hit the previously
documented host SDK/linker mismatch; selecting the Nix linker in PATH resolved
it without changing repository build configuration.

Native tests used Rust 1.98.1; the controlling clippy gate used Rust 1.95. The
Linux checks used Nix Rust 1.96.1 and its matching sysroot. **ASSUMED:** native
Linux, FreeBSD and x86-64 runtime grace/fault/reuse behavior remains unmeasured.
Cross-compilation does not establish runtime correctness, and no actual code
extent reclamation or reuse is enabled at this boundary.

### Step 3c: provider extent identities and generation-aware admission (2026-11-10)

**READ — retained-storage boundary; phase 3 remains incomplete.** Each dense
provider allocation now owns a process-wide monotonically increasing generation,
region index, offset and complete reserved length (including veneer capacity).
The generation is reserved by `allocate`, before copying or finalization. Pending,
copy-failed and finalization-failed extents therefore retain their own identities,
independent of whether a function ever reaches the code map. An address index
resolves an exact RX allocation start to its record; interior addresses do not
resolve. Index capacity is included in the metadata payload counter; hash control
bytes and allocator overhead are excluded. The records and extents remain retained.

The narrow vendored integration adds a default `allocation_generation` provider
hook and a finalized-function query. Both lazy MXCSR modules use it. Foreground
and background publication carry that provider generation into `CodeUnit` and
`CompiledPtr`, without minting a second identity at publication. The system
provider (including unsupported-MAP_JIT fallback) has no per-allocation hook and
still receives a fresh identity at publication; its storage remains retained.
Finalized-function generation queries obey the same finalization-failure and
unfinalized-definition checks as entry queries.

`CompiledPtr` carries the address in `.0` and a private generation exposed by
`generation()`. Code retaining a pointer must copy the complete value. Constructing
an indefinitely retained/static ABI entry now uses `CompiledPtr::new(address)`;
the old tuple constructor is no longer available. This is a source migration for
raw-ABI consumers. Registered entries come from `CodeUnit::entry()`. The existing
`MemoryStats` fields and compiled-code observer signatures are unchanged, including
their retained-storage meanings; live/retired/reusable gauges and lifecycle events
have not been implemented.

JIT domains require generations. All four cache publication paths check the exact
live `(address, generation)` under the same registry mutex that updates publication
ownership. Source validation/marking still runs outside this mutex; admission
rechecks afterward, so retirement during that callback cannot publish an old
entry. Rejection preserves the previous cache entry and its memory accounting.
Discard and publication removal also match the generation, so a late completion
cannot retire a later unit at the same address. Address-only entries and entries
from another domain are refused. Legacy domains accept address-only entries for
static/retained backends. Every in-tree JIT wrapper now forwards `code_lifetime`;
a wrapper which omits it cannot publish its tracked entries into the VM cache.

Direct ABI callers can use `ExecutionGuard::admits(entry)` before calling a saved
entry, retaining participation through execution and cold decoding. It checks both
the guard's current epoch and the entry's live generation under the registry lock.
Admission followed by retirement is safe only because the participant still pins
that extent. An address reconstructed from a slot is used only within the existing
dispatch execution interval: generated slot clears and epoch checks still govern
those transfers. The relaxed-pointer publication bound fixture intentionally has
fixed retained ownership and no retirement; it does not demonstrate admission of
saved entries across reuse. Actual reclamation must finish the direct-ABI audit
before that ownership assumption can change.

**Remaining work, in dependency order:**

1. Build bounded compiler/module ownership and maintenance records tying each
   retired generation to its `FuncId`, slots, relocation/symbol references and
   provider extent. Dispose of failed/pending allocations as well as published
   units. The current Cranelift declarations, blobs, slots, descriptors, provider
   records and lifetime retirement queue are still append-only. Provider extent
   identity alone does not supply a safe module-record release operation.
2. For execution-complete tickets, remove the exact code-map range using the
   existing `RangeRemoval` completion token, then wait for that token before
   releasing metadata, compiler references or executable storage. Consume all
   completed tickets; do not poll or retain them forever. Handle backend teardown
   without blocking on an executing helper or leaking its deferred ownership.
3. Add coalescing free lists, aligned reuse and failure cleanup. Reuse needs dirty
   allocation tracking: the current `Region::flushed` bump watermark does not
   flush writes below that watermark. Preserve the publication/pipeline protocol.
4. Decommit wholly free shared backing pages and release empty Linux/FreeBSD
   regions. Their backing descriptors currently close immediately after mapping;
   shared-backing decommit must account for both aliases rather than advising just
   one view. macOS must return free logical regions to its process arena, retaining
   the single non-inherited reservation while decommitting free pages.
5. Add compatible live/retired byte and unit gauges, reusable-free-byte accounting
   and publish/retire/release observer events with allocation generation IDs.
6. Add actual-reclamation witnesses for all four triggers, held-vCPU and held-signal
   reader exclusion of reuse, live fault classification during reclamation, and
   bounded retire/recompile code **and module/metadata** storage that fails on the
   retained-storage baseline. The same-address registry witness in this step is
   synthetic and does not establish safe executable address reuse. Validate the
   macOS fork contract and obtain native Linux/FreeBSD/x86-64 runtime evidence.

No ordinary retirement calls `unregister` or provider `free_memory`, no retired
extent becomes reusable, and no decommit is enabled. The existing execution and
two-counter signal-reader protocols are unchanged. **ASSUMED:** native Linux,
FreeBSD and x86-64 reclamation behavior remains unmeasured.

**MEASURED, macOS arm64:** native tests used Rust 1.98.1, offline Cargo, an empty
compiler wrapper, the isolated target directory, two build jobs and at most two
nextest threads. Every test invocation used `timeout -k 5 180` and
`--test-threads 2`. No full suite or benchmark was run. The bound binary ran once;
its 100,000-unit case passed in 34.792 seconds and still measures retained storage.

| Selected check | Result |
|---|---|
| Core library | 261 passed |
| Backend library | 128 passed, four skipped |
| Backend `memory_stats` | Eight passed |
| Backend `code_memory_bound`, single invocation | Two passed |
| `region_fault_state`, with Unicorn | 26 passed |
| `fault_path`, with Unicorn | 16 passed |
| `fault_metadata`, with Unicorn | 59 passed |
| `bg_tier`, with Unicorn | 24 passed |
| `smc`, with Unicorn | 57 passed |
| Wrapper audit: `guest_trap_sites` | 14 passed |
| Wrapper audit: `fault_landing_pad` | 31 passed |
| Wrapper audit: `rust_memory_faults` | 18 passed |
| Wrapper audit: selected MMX, x87, fetch-policy and child-invalidation cases | Six passed |
| Rust 1.95 offline all-target/all-feature clippy, `-D warnings` | Passed |
| Rust 1.95 formatting; diff whitespace check | Passed |
| Offline workspace/tests check, x86-64 Linux, Nix Rust 1.96.1 | Passed |
| Offline workspace/tests check, AArch64 Linux, Nix Rust 1.96.1 | Passed |

Four new core witnesses cover stale same-address generations across all admission
paths, unmanaged/foreign-domain rejection, direct guard admission and retirement
during source validation. The provider witness covers identity before publication,
failed finalization, exact-start lookup and bookkeeping removal under explicit
whole-provider test cleanup. The existing module byte witness now checks that the
finalized-function query returns the provider-recorded identity. The new backend
integration witness checks identity preservation across materialization, cache and
direct ABI entry, participation pinning on invalidation, and stale-clone rejection.

Disabling the cache's generation admission check made the same-address witness
fail at its first stale `upgrade` assertion. The intended source was restored and
the final complete core library run passed. This is an admission falsification,
not an executable reuse or retained-storage-baseline falsification.

An initial backend run exposed wrappers missing `code_lifetime` forwarding. One
fixture timed out waiting for compiled entry and another looped during warmup;
the command watchdog terminated that run with exit 124. Forwarding was added to
all in-tree wrappers of a JIT/backend, including wrappers outside the five requested
integration binaries, and the corrected backend run plus the targeted wrapper
fixtures passed. The first core link attempt encountered the recorded SDK/linker
mismatch; selecting the Nix linker in PATH fixed it. Native Rust 1.98.1 still emits
the existing libc-declaration signature warnings. The Rust 1.95 clippy gate passes
without warning allowances. The backend arena child/grandchild and shutdown/fork
witnesses passed; no native Linux, FreeBSD or x86-64 runtime was available.

### Step 3d: explicit metadata collection and declaration recycling (2026-11-11)

**READ — consistent metadata boundary; phase 3 remains incomplete.** The rebase
preserves provider failure diagnostics, automatic alias/anonymous/system fallback,
the descriptor-free protection policies, allocation generations, execution grace
and the two-counter signal-reader protocol. The new anonymous execution witness
copies the complete `CompiledPtr`, preserving its generation. Conflict contents
are resolved; staging and continuing the rebase remain the maintainer's operation.

`JitBackend::collect_retired_metadata()` is an explicit cold maintenance call. It
serializes with compilation and returns without waiting for executing vCPUs or
signal readers. It acquires compiler locks and can wait for those locks. For each
execution-complete retirement it removes the exact code-map range, retains the
specific `RangeRemoval` token, and polls that token on subsequent calls. Only
after reader completion does it remove the unit's compiler blob, relocation
records, cached symbol and module PC index, drop its stable chain/IBTC/return
continuation slots, and consume the lifetime retirement ticket and removal token.
Inactive vCPU predictions cannot dereference freed slots: dispatch clears them
on the retirement epoch change before pointer resolution. Live units retain
their addresses and metadata. The classifier keeps its existing bounded
validation/fallback protocol and its qualification about LL/SC RMW retries.

Each MXCSR module owns pending definition slots by `FuncId` and finalized function
records by allocation generation. Finalization binds these before fault-map
publication, including background definitions finalized by a foreground compile.
The system provider receives its fresh identity at that same boundary. The narrow
vendored `retire_function_definition` operation keeps a vacant declaration; a
later definition reuses its ID and unchanged ABI signature. This bounds declaration
and blob-table growth by peak simultaneous definitions when maintenance is called.
x86jit uses baked helper addresses and indirect chain slots, so another unit has
no relocation to the retired definition's ID/name. The unsafe vendored operation
requires that property of its caller. Inherited backends reject collection by
process identity before touching copied compiler locks.

**Retention boundary:** this API does not release provider extents. Provider
allocation records/address indices, executable backing and cache-owned immutable
IBTC descriptors remain retained. Existing memory counters keep their phase-2
meanings; observer signatures are unchanged. Failed/unpublished definitions and
abandoned scratch slots remain module-owned; terminal allocation/finalization/
registration failures stop further compilation.
Collection is not wired into the VM dispatcher, memory snapshots or backend
teardown, and no ownership is transferred to a deferred teardown collector.
Dropping a backend still follows the previous retained-code contract. Metadata
collection alone does not establish bounded total metadata or code storage.

**READ — provider policies for the next storage step:** Linux/FreeBSD strict
anonymous W^X reuse is limited to the RW frontier. Sealed holes stay immutable;
release a wholly free region and obtain a fresh anonymous reservation instead of
reopening RX pages, even after grace. Decommit by itself does not make a sealed
extent reusable. Explicit RWX and alias modes can eventually reuse fully quiescent
holes with fresh cache/pipeline publication. macOS public no-fd configuration
continues using MAP_JIT; its process reservation/fork policy is unchanged. Both
explicit and automatic system-provider selection retain executable allocations;
per-unit executable reclamation there is unsupported. Metadata collection and
declaration recycling work with that provider as well.

**Remaining work, in dependency order:**

1. Attach releasable provider handles and storage ownership to the collected
   function records. Release storage before `finish_retirement` consumes the
   ticket; the current collection path deliberately retains that storage. Include
   failed/pending definitions and cache-owned immutable IBTC descriptors.
2. Implement aligned coalescing free lists, generation-checked extent release and
   reuse, and dirty-range cache maintenance below the existing bump watermark.
   Decommit wholly free backing pages behind both alias views; retain the backing
   handles needed for that operation. Release empty Linux/FreeBSD regions with
   stable live allocation indices. Retain/decommit the single macOS reservation.
   Respect the strict anonymous policy above.
3. Integrate maintenance with normal invalidation/replacement/rejected-completion
   flows and preserve deferred ownership through backend teardown when grace is
   incomplete. Waiting for a blocked compiled helper is not a teardown solution.
4. Add compatible live/retired byte and unit gauges, reusable-free-byte accounting,
   and generation-bearing publish/retire/release observer events.
5. Witness physical reclamation for all four triggers, bounded retire/recompile
   code and total metadata storage against the retained-storage baseline, held
   vCPU/reader exclusion of extent reuse, stale-generation rejection after actual
   address reuse, live fault classification during reclamation and native platform
   behavior. Current witnesses reclaim metadata and recycle declarations only.

**MEASURED — native macOS arm64, offline Cargo, two build/test threads, every test
command under `timeout -k 5 180`:** core library 262 passed; backend library 146
passed/four skipped; backend memory statistics eight passed; bound binary four
passed in one invocation, including the sole 100,000-unit case (62.843 seconds).
Unicorn-enabled `region_fault_state` 26 passed, `fault_path` 16 passed,
`fault_metadata` 59 passed, `bg_tier` 24 passed and `smc` 57 passed. No full suite
or benchmark was run. The bound case still measures fresh retained dense storage.

The new core witness holds execution and a code-map reader independently through
retirement and consumes the ticket only after both release. Three new backend
witnesses cover held execution, live source lookup during metadata collection,
slot destruction, and 32 retire/recompile cycles in both MXCSR modules with dense,
public no-fd and system providers. They assert one reused declaration per module
and empty completed function/ticket/removal registries. The macOS public no-fd
case uses MAP_JIT; native Linux/FreeBSD anonymous execution remains **ASSUMED**.

**MEASURED:** disabling declaration recycling made the repeated-cycle witness
fail at its first reused-ID assertion (`funcid1` instead of `funcid0`). The
intended source was restored and that single witness passed again. This is a
compiler-record bound falsification, not an executable-storage bound witness.
Rust 1.95 all-target/all-feature offline clippy with `-D warnings`, workspace
and vendor formatting, and diff whitespace checks passed. Native tests used
Rust 1.98.1, which reports the existing libc-declaration signature warnings;
the controlling Rust 1.95 gate needs no warning allowances.
Offline workspace/test checks passed for `x86_64-unknown-linux-gnu` and
`aarch64-unknown-linux-gnu` with Nix Rust 1.96.1 and its matching sysroot.
These are compilation checks; native Linux/FreeBSD/x86-64 reclamation behavior
remains **ASSUMED**. No test exceeded its timeout.


### Step 3e: explicit provider release, reuse and page discard (2026-11-11)

**READ — tested storage boundary; phase 3 remains incomplete.** This section
supersedes step 3d's retained-provider-storage boundary. Collection remains the
explicit `JitBackend::collect_retired_metadata()` cold operation. It takes the
compiler serialization lock but never waits for executing vCPUs or signal readers.
No dispatcher, statistics snapshot or teardown automatically invokes it yet.

The order is execution grace, exact code-map removal, that removal token's
signal-reader completion, compiler blob/relocation/symbol/PC-index retirement,
stable-slot destruction, provider extent release and retirement-ticket consumption.
The narrow provider patch adds an unsafe, generation-checked per-allocation
release operation, defaulting to unsupported; a JITModule forwarding operation
requires the definition to have already been retired. The system provider keeps
its executable storage, including when selected by automatic fallback.

Dense allocation records include their bump-alignment padding. Released holes
coalesce; aligned reuse splits a hole and can join a trailing hole to the unused
tail. Every allocation obtains a fresh process-wide generation, including at an
identical executable address. Stale or duplicate releases leave the replacement
untouched. Dense allocation records use swap removal and repair the moved record's
address index; empty region records are reused without shifting live neighbors.
Free, dirty and discarded-page interval storage scales with fragmentation and
peak simultaneous ownership, rather than the number of retirements. Live code
never moves. Dirty ranges track reused writes below the old bump watermark;
publication repeats cache and pipeline synchronization before exposing them.

Linux and FreeBSD retain an owned backing descriptor while an alias region
exists. Decommit operates on that shared object, so both views lose the backing:
Linux uses `fallocate(PUNCH_HOLE | KEEP_SIZE)`; FreeBSD uses `fspacectl(DEALLOC)`
and handles a partial successful result. Only complete pages inside coalesced
free extents qualify. Successful discard ranges are accounted once; new writes
remove their pages from that accounting. A discard failure records operation
`DecommitPages` (21) and its original errno, retains commitment in the counters,
and leaves the quiescent extent reusable. A later neighboring release retries
undiscarded pages. Empty Linux/FreeBSD regions unmap both views and close the
backing descriptor. Descriptor-free anonymous reservations never reuse holes,
including explicit RWX reservations at this boundary: a wholly free region is
unmapped and the next allocation obtains a fresh reservation. Sealed RX/R pages
in a partially live reservation remain immutable.

macOS retains the process-wide MAP_JIT reservation and its logical code regions,
reuses their quiescent holes, and submits wholly free pages with `MADV_FREE`.
**MEASURED:** mincore still reported resident pages immediately after that call;
its dirty/modified flags were cleared, while a neighboring live page retained its
bytes. This is surrender of dirty backing to kernel reclamation, not a claim of
immediate RSS reduction. **READ:** XNU's `VM_BEHAVIOR_FREE` path clears modified
state and makes the pages reclaimable; physical page removal is asynchronous.
The fork exclusion, per-thread write protection and process reservation remain
unchanged. Immediate physical page removal from MAP_JIT has not been established.

The legacy `set_compiled_code_observer` promises backend-lifetime code ranges.
Installing it therefore permanently retains this backend's executable allocations,
even if a callback panic later disables notifications. Explicit collection still
retires compiler/fault metadata and declarations. This preserves that API contract
until a compatible lifecycle observer lets an embedder opt into release events.
A new witness checks the retention promise across both collection and callback
panic. The old observer signature and `CompiledCode` layout are unchanged.

Existing storage counters now subtract reusable holes from `code_bytes_used`,
include them in `code_bytes_free`/largest extent, subtract successfully discarded
pages from `code_bytes_committed`, and remove released regions/views/reservations.
macOS commitment counts surrendered pages, independently of resident pages.
Allocation metadata includes the interval vectors. Finalized/emitted counters
keep their prior cumulative meanings; they are not the new live/retired gauges.

**MEASURED witnesses:** raw provider tests cover aligned splitting/coalescing,
backing discard, failed-discard diagnostics and accounting, stale generation
release rejection, publication cache maintenance below the old watermark,
empty-region release and stable live region indices. The anonymous witness
observes RX protection after a partial retirement and an absent old reservation
after complete retirement. A backend witness holds an execution participant over
a real compiled unit, proves that collection and recompilation cannot reuse its
extent, releases participation, then observes the exact address reused with a
new generation. Admission rejects the old entry and accepts/executes the new one;
a live neighbor retains source lookup and execution. Two-module repeated-cycle
witnesses bound provider storage, allocation/index/interval records, and compiler
declarations across 128 cycles with dense and public no-fd providers, with no
legacy observer installed. They do not bound the still-retained IBTC descriptors.

**MEASURED falsification:** temporarily disabling only the provider release call
made the independent repeated-retirement bound witness fail on cycle zero:
88 used bytes remained instead of zero. This reproduces the retained-storage
boundary of the starting revision; it was not a separate checkout/test of main.
The call was restored and the final backend library run passed.

**READ — remaining, in dependency order:**

1. Failed/unpublished definitions and abandoned scratch-slot cleanup, plus
   cache-owned immutable IBTC descriptor ownership and reclamation.
2. Automatic maintenance on normal retirement paths and deferred teardown
   ownership when execution or reader grace is incomplete. Teardown still retains
   storage; do not wait for blocked compiled helpers. macOS logical regions also
   need ownership/reuse across backend destruction.
3. Compatible live/retired bytes and unit gauges, reusable-free-byte accounting,
   and generation-bearing lifecycle publish/retire/release events. An embedder
   needs an explicit lifecycle-observer contract before observed code is recycled.
4. End-to-end physical reclamation witnesses for invalidation, tier replacement,
   rejected completion and teardown; a total-metadata bound including IBTC and
   failure paths; signal classification during actual concurrent reclamation;
   and native Linux/FreeBSD/x86-64 runtime validation. Cross checks are compilation
   evidence only. Immediate macOS resident-page removal remains unproven.

Primary OS references checked for this step:
[Linux fallocate(2)](https://man7.org/linux/man-pages/man2/fallocate.2.html),
[FreeBSD shared-object deallocation](https://cgit.freebsd.org/src/tree/sys/kern/uipc_shm.c?h=releng/15.0),
[XNU VM behavior and msync](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/vm/vm_map.c)
and [XNU page deactivation](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/vm/vm_object.c).


**MEASURED validation — native macOS arm64, offline Cargo, two build/test
threads, every test command under `timeout -k 5 180`:** core library 299 passed;
backend library 160 passed/four skipped; memory statistics eight passed;
`region_fault_state` 26, `fault_path` 16, `fault_metadata` 59, `bg_tier` 24 and
`smc` 57 passed with Unicorn. The sole `code_memory_bound` invocation passed all
four cases; its 100,000-unit case took 35.755 seconds. No full suite or benchmark
was run, and no invocation exceeded its timeout. Rust 1.95 all-target/all-feature
clippy with `-D warnings`, workspace/vendor formatting and diff whitespace checks
passed. Workspace/test checks passed for both Linux ISAs, plus backend/test
checking for FreeBSD arm64, using the matching Nix Rust 1.98.1 toolchain/sysroot.
These checks do not establish native Linux/FreeBSD reclamation behavior.

The starting rebase had four compile leftovers exposed by these gates: the
victim-free invalidation branch referenced the removed cache epoch field; a
lifetime test fixture lacked `legacy_context`; a vCPU reset fixture and the
cache-index memory example still used the removed tuple constructor. The branch
now advances the shared lifetime epoch, and the fixtures/example use the current
IR/entry constructors. The initial macOS residency assertion was corrected to
measure clean backing separately from residency, and the clippy interval-vector
initialization warning was fixed. Final checks use the restored release call.


## Final phase 3 ownership and maintenance

**READ — final implementation contract, 2026-11-11.** Invalidation, compiled tier
replacement, rejected background completion and backend destruction all unlink
units into the same generation-bearing retirement domain. Cache publication is
serialized with removal; shared chain and IBTC slots clear before epoch advance.
Each dispatcher participant enters before resolving pointers and retains its
announcement through calls, chains, helpers and cold fault decoding. Fast entries,
return predictions and refill bookkeeping are flushed on changed epochs. Idle
participants announce zero. A blocked helper retains its announcement and delays
only generations it could still enter. Live code never moves.

The existing two-counter code-map protocol is the signal-reader grace period.
New validated readers select the new counter; reclamation polls only the prior
counter. One validation failure pins both counters without a validation loop.
LL/SC implementations of atomic increments/decrements can retry, so neither
reader entry nor maintenance has a hard wall-clock bound. Exact range-removal
completion covers destruction of obsolete roots and trap/source/instruction
payloads, shared unwind recipes, and instruction intern pools before address
reuse. A generation is consumed only after those obligations and compiler-record
retirement have completed.

`reclamation.rs` owns cold collection. The first public compilation/submission
starts one maintenance thread per backend; an unused backend starts none. The
thread parks on a condition variable when no units or descriptors await
reclamation. Unit retirement, prediction retirement and shutdown wake it. Only
while reclamation remains pending does it retry at 10 ms intervals, independently
of dispatch, new compilations and statistics requests. Descriptor collection
skips the slot/participant registry scan when no retired descriptors exist.
After execution grace the collector takes compiler serialization only for
range-removal initiation/polling and final compiler/provider release. It never
waits for a vCPU or a signal reader. Each poll visits at most 64 units with a
rotating cursor; the generation index makes ticket validation and removal
constant-time rather than repeatedly scanning a large retirement vector.

The 10 ms interval is an intentional delay between pending-work polls, not a
per-unit latency bound. Candidate backlog and cursor rotation add delay: 100,000
simultaneously retired units require at least `ceil(100000 / 64) = 1563` polls,
so the intervals alone contribute approximately 15.6 seconds before the last
releases, even if both grace periods are complete. A generation that just missed
its visit waits for cursor rotation. OS descheduling, a current compiler
operation and observer callback time add further delays. This is bounded work
per candidate batch, not a real-time service guarantee. A stalled generation
does not prevent collection of another generation whose execution and reader
grace have completed. Explicit `collect_retired_metadata()` remains available;
a concurrent or reentrant collector returns zero. No collection is performed on
cache-hit dispatch or by `MemoryStats` (the existing code-map statistics sampler
can still collect obsolete code-map versions).

`set_worker_spawner` covers compiler, maintenance and deferred-reclamation
threads, including signal-mask setup and thread registration before callbacks.
Install it before the first compilation or submission. Shutdown joins the
compiler and maintenance threads, rejects future work, closes compiled cache
resolution/publication, retires remaining units and drains completed batches
without the polling delay. It reports pending execution or signal-reader grace
as a live resource. Successful `Vm::shutdown()` completes teardown reclamation
before returning and cannot spawn a deferred worker during later destruction.
The cache lifetime domain retains deferred-worker join handles across backend
replacement; VM shutdown joins them or reports their pending grace as a live
resource before certifying the fork boundary.

Ordinary backend destruction also closes the lifetime domain before retirement.
Caches reject compiled lookup afterward with one predictable domain-state check;
the epoch change clears vCPU-private predictions. Destruction tries collection;
if grace or backlog remains, a deferred owner retains the entire compiler state,
helper counters, stable slots and provider until collection completes. Destruction
does not wait for blocked execution or readers. The deferred owner drops modules
with `free_memory` only after all unit retirements finish, also releasing retained
system-provider allocations. If the spawner cannot create the maintenance
thread, compilation enters code-space-exhaustion fallback. If it cannot create a
deferred owner, ownership is intentionally retained rather than freeing active
code. A participant or signal handler abandoned forever necessarily retains its
storage; nonlocal escape from signal guards remains unsupported.

Unpublished definitions use a scoped guard: every error or unwind destroys
abandoned scratch slots, pending slots, blobs, relocation/symbol/PC-index records,
and returns the declaration for reuse. A narrow vendor discard hook accepts a
never-published pending or failed-finalization definition. Initialization failure
before blob construction releases its provider allocation directly. Background
batch failures and finalized entries rejected before code-map registration use
the same cleanup. System-provider allocation release remains unsupported per
unit; its backing is released at safe final module teardown. Successfully
registered entries use both grace periods, including rejected cache completions.

Immutable IBTC descriptors belong to the execution domain, rather than a cache
vector retained until VM destruction. Clearing shared predictions moves active
descriptors into epoch-tagged retirement storage. Collection drops each only
after every participant which could have loaded its pointer leaves that epoch.
The domain outlives cache destruction when participants remain. Conservative
prediction flushes also retire descriptors when an invalidation advances the
cache epoch without retiring the underlying shared compiled unit. Dead weak-slot
registrations are pruned by cold collection. Storage is bounded by peak live
ownership, fragmentation and currently incomplete grace periods, rather than the
number of completed retirement cycles.

Provider release follows execution grace, exact range removal and reader grace,
observer Release completion, compiler-record and stable-slot destruction. Dense
holes coalesce; aligned reuse gets a fresh process-wide generation and repeats
instruction publication/cache synchronization. Complete free backing pages are
discarded; empty Linux/FreeBSD regions release both aliases and their descriptor.
Anonymous W^X reservations remain immutable until wholly free, then release as
a whole. The macOS process MAP_JIT reservation remains non-inherited and never
unmaps; safe final provider teardown decommits its logical regions and returns
them to a coalescing process pool for other backend owners. Within a live provider,
free holes remain local for reuse. `MADV_FREE` surrenders dirty backing but does
not promise immediate RSS reduction. The fork contract is unchanged.

`MemoryStats.backend` adds `code_bytes_live`, `code_bytes_retired`,
`code_units_live`, `code_units_retired`, `code_bytes_reusable` (holes excluding
unused tails), and `code_ibtc_descriptors`. Byte gauges count emitted unit lengths,
excluding alignment and veneer reservation; provider used/committed counters
retain those distinctions. Legacy `compiled_units_live` and
`compiled_units_total` remain cumulative finalized-unit counters for compatibility.
Retired gauges include units waiting for execution, reader or collector progress.
Legacy observer retention and unsupported system-provider release can leave
provider used bytes after these unit gauges reach zero. Snapshots are independent
and take the lifetime registry briefly, never compiler locks.

The additive `set_code_lifecycle_observer` API must be installed before compiling.
Each event includes `CodeUnit` (host range plus generation), the original
`CompiledCode` metadata (including guest entry and block/region kind), and Publish,
Retire or Release. Publish runs on the compiler thread before exposing the entry. Retire
runs on a collector after unlinking, possibly one polling interval later. Release
runs after both grace periods and before provider reuse; consumers must stop
using the range before returning. Notifications for one generation are ordered.
Callbacks execute outside compiler, registry and observer locks and may reenter
compilation or explicit collection; collection itself is guarded against reentry.
They must not wait for maintenance or call shutdown. Callback panics disable
notifications with the existing containment rules; consumers must discard their
references on panic. System providers may retain storage after Release. Installing
the legacy `set_compiled_code_observer` still permanently pins executable storage
through backend lifetime, including after callback panic; its signature and
`CompiledCode` layout are unchanged, even if a lifecycle observer is also present.

### Witness scope and deferred validation

**READ:** retirement witnesses cover actual cache invalidation, compiled tier
replacement, dispatcher rejection of a background completion, and destruction
with an execution participant held. They inspect allocation retirement/release,
metadata and module record removal, generation-bearing ordered events, and
post-grace ownership. A reentrant Release callback compiles another unit and
proves its old extent is still allocated until that callback returns. Failure
witnesses exercise all four injected foreground/background failure stages and
abandoned translation scratch storage. Repeated automatic cycles bound executable
storage, declarations, compiler and slot registries, IBTC descriptors, code-map
roots, trap/source/instruction tables and unwind recipes. Existing held-execution
and held-signal-reader witnesses establish independent grace-period protection.
A real-signal witness faults repeatedly in a live compiled unit while another
backend retires/recompiles and reuses storage. A macOS witness transfers the same
logical region between provider owners. See measured results below.

`xj-2jb.5` retains stress and platform coverage: native Linux/FreeBSD and x86-64
execution, alias backing discard and PSS/unique residency measurements, concurrent
multi-vCPU compilation/publication/invalidation/tiering and repeated address reuse,
continuous/nested signals, cross-thread instruction visibility, unwind/fork under
load, resource-failure and maintenance-thread creation failure campaigns, and
performance/codegen review plus the real-program ladder. Cross-target checks here
are compilation evidence only. Immediate macOS resident-page removal remains
unproven. No live-code compaction is planned.

### Final phase 3 validation (2026-11-11)

**MEASURED — native macOS arm64, Rust 1.98.1, offline Cargo, two build jobs and
`--test-threads 2`; each test command used `timeout -k 5 180`:**

| Check | Result |
|---|---|
| Core library | 300 passed |
| Backend library | 168 passed, four skipped |
| `memory_stats` | Nine passed |
| `code_memory_bound` (one invocation) | Four passed; 100,000-unit case 49.425 s |
| `region_fault_state`, Unicorn | 26 passed |
| `fault_path`, Unicorn | 17 passed, including concurrent real-signal reclamation |
| `fault_metadata`, Unicorn | 59 passed |
| `bg_tier`, Unicorn | 24 passed, including shutdown/fork/rebuild |
| `smc`, Unicorn | 57 passed |
| Final lifecycle payload/reentrant callback rechecks | Two passed |
| Final descriptor ownership recheck | One passed |
| Final total-metadata cycle recheck | One passed |
| Rust 1.95 offline clippy, all targets/features, `-D warnings` | Passed |
| Workspace and changed vendor-source formatting; diff whitespace | Passed |
| Workspace/tests, x86-64 Linux, matching Nix Rust 1.98.1/sysroot | Passed |
| Workspace/tests, AArch64 Linux, matching Nix Rust 1.98.1/sysroot | Passed |

The single packed bound run preceded the final additive lifecycle publication
payload enrichment; it was not repeated under the load limit. Final lifecycle
and repeated-reclamation witnesses were rerun after that enrichment. No full
suite or benchmark was run, and no invocation exceeded its timeout. Native
runtime evidence is macOS arm64; the Linux checks establish compilation only.

**MEASURED falsification against main:** the unchanged embedder-API witness
`repeated_invalidation_recompilation_bounds_storage_and_metadata` was run against
an archived source copy of main `73ed5d6`, without creating a worktree or changing
this branch. It failed on cycle zero after its five-second deadline with 44
retired executable bytes still allocated. The same witness passed all 16 cycles
on this branch. Disabling descriptor collection separately made the total-
metadata witness fail with one retained IBTC descriptor instead of zero; the
intended source was restored and the final witness passed. The bounded cycle
witnesses exercise reclamation, rather than only fresh packed allocation.

The older retention-only memory-statistics fixtures now hold execution while
examining retired state and wait for automatic removal afterward. The fork
fixture accounts for the additional maintenance thread and still observes the
original baseline thread count after shutdown, before forking and after rebuild.
Deferred teardown witnesses wait for provider counters as well as weak-owner
expiry: a weak upgrade can fail while the final destructor is still releasing
its logical regions. A 129-unit witness covers more than one 64-candidate batch.
