---
id: decision-27
title: 'Compiled Machine execution: J0 evidence, identity and stage order'
date: '2026-11-24'
status: accepted
---

Decider: maintainer (2026-11-24), accepting every recommended answer in the
"Maintainer questions" table below. It extends decision-23 §8; J1-J5 keep their
own decisions and witness gates.

# Compiled Machine execution: J0

This record is xj-to9's J0, extending decision-23 §8. Acceptance does not silently
authorize every later tier: J1–J5 retain their own decisions and witness gates. No engine
implementation is included here.

## Context and evidence

The embedder reports approximately 8 M retired instructions/s for a protected32 register loop and
1.83–1.89 M/s with paging, against 96–100 M/s in real mode. Its target is **at least 35 M/s for
protected32 with paging** on an ARM64 macOS host. Those are supplied observations, not measurements
reproduced here; its devices, scheduling slices and exact ROM are not in this repository.

Repository baseline: `0bc16f55e557f3a4dac12a35335b175d41a42a54`. Paging and V86 are public. Checked
protected and paged execution remain interpreted, one instruction per block. Direct real-mode
execution uses multi-instruction IR. Paging has the common translation boundary but no functional
TLB yet; the accepted paging design explicitly allowed this uncached first implementation.

### Method

Host: Apple M5 Pro, ARM64, 18 cores (6 performance + 12 efficiency), 64 GB; macOS 27.0.1 (26A434),
Darwin 27.0.0. Rust 1.98.1 / LLVM 22.1.8; default Cargo **release**, no engine changes or extra
counter instrumentation. The benchmark uses `Fast` memory consistency and eager Cranelift `Speed`
for JIT.

`x86jit-bench/examples/machine32_loops.rs` uses the `real16_loops` V4
`proc_pid_rusage().ri_instructions` counter and `Instant` around execution only. Runs use
`scripts/perf_campaign.py::run_process`, ASLR disabled, serially: three fresh processes × three
rotated rounds, 2,000,000 loop iterations per sample. Each VM first warms for 10,000 iterations,
excluding setup, compilation and the initial page-table A/D updates. Each 32-bit sample completes
10,000,001 instructions: five per iteration plus HLT. Counter acquisition adds a negligible fixed
overhead. No devices, IRQs, cycle timing or periodic scheduling slices.

Register: INC EAX; ADD EBX,EAX; XOR EDX,EBX; DEC EDI; JNZ. Memory: MOV EAX,[ESI]; ADD EAX,1; MOV
[ESI],EAX; DEC EDI; JNZ. Call/return: CALL; INC EAX; RET; DEC EDI; JNZ, with a balanced RAM stack.
Protected and Application use identical bytes. Machine is I80386, CPL=0, flat full-limit 32-bit
caches; identity paging uses ordinary RAM tables. Code/data/stack/tables occupy separate pages.
Real16 uses 16-bit equivalents and DI-limited chunks of at most 60,000 iterations, adding 34 HLTs
rather than one. Machine retirement is asserted against the known count; Application JIT retirement
is **derived from the validated finite loop**, because its public retirement counter stays zero.
EAX, DI, SP and memory results are checked.

Reproduce after a normal release build with `machine32_loops 2000000 3`, three processes through the
campaign runner. Diagnostics use one round in each of three fresh processes: `machine32_loops
2000000 1 PATH WORKLOAD`, where PATH is `application-single32`, or WORKLOAD is `register-pad`;
`PATH=components` runs the isolated incremental decoder kernel. Build with `RUSTC_WRAPPER=` if
sandboxed sccache is unavailable; this host also needed the installed LLVM Mach-O linker. Raw
observations: `bench/history/xj-to9-j0.csv` and `bench/history/xj-to9-j0-components.csv`;
host/tool/binary metadata and summaries: `bench/history/xj-to9-j0.json`. These are J0 evidence, not
a regression campaign.

Medians of nine samples; each cell is **M retired/s / host instructions per step**.

| Path | Register | Memory | Call/ret |
|---|---:|---:|---:|
| Machine real16 | 89.93 / 339.9 | 79.48 / 417.1 | 45.83 / 652.5 |
| Machine protected32, PG=0 | 10.19 / 2,807.0 | 8.74 / 3,200.7 | 7.91 / 3,417.1 |
| Machine protected32, PG=1 | 1.92 / 12,384.4 | 1.65 / 14,511.6 | 1.60 / 15,002.9 |
| Application Compat32 interpreter | 94.99 / 316.8 | 98.33 / 330.6 | 62.50 / 489.1 |
| Application Compat32 JIT | 1256.62 / 24.6 | 664.65 / 48.2 | 316.11 / 87.6 |

Register rate ranges were 8.37–10.27 M/s unpaged and 1.66–2.09 M/s paged; frequency/load/scheduling
were not pinned. Register host counts varied by less than 0.04%. Use counted work, not timing
differences between separate runs, for attribution.

### Where the work goes

`sample` could not examine a process inside the sandbox; no sampling percentages are claimed.
Counted probes and the source establish this cost model:

| Probe/component | Host instructions per guest step | Interpretation |
|---|---:|---|
| Normal Compat32 interpreter | 316.8 | Warm five-instruction block reference. |
| Compat32 singleton cached IR | 744.3 | Same code, public `lift_one` prepopulates singleton cache entries; +427.5 exposes lost block amortization, including lookup/dispatch. Not the private Checked hash cost. |
| Incremental decoder alone | 699.3 | Repeats `Decoder::with_ip/decode` for every required prefix length, same 386 options; excludes fetch, segment checks and proofs. |
| Checked unpaged residual | about 1,363 | 2,807 minus those singleton/decode proxies: segment/access checks, Checked execution/checkpoints, allocation and different cache/context handling; not a measured segment-check-only subtotal. |
| Paged versus unpaged register loop | +9,577.4 | No guest operand traffic: fetch walking, source/proof lifecycle, context reconciliation, and transaction preparation/replay. |
| Four redundant CS prefixes per loop | +538.8 PG=0; +1,131.9 PG=1 | Same five instructions/results; Application interpreter remains 316.8. Extra decoder kernel work is 462.9. |

The padding adds 0.8 fetched bytes/step. Its extra paging premium is about 741 host instructions per
extra byte, **including source tracking as well as the walk**. Extrapolated across the baseline's
1.6 bytes/step, this is about 1,186 instructions/step (9.6% of paged work), a marginal proxy, not an
isolated walker measurement or a strict upper bound. A TLB-only 18× claim is unsupported.

`Execution::resolve` / `fetch_bytes` in `vm/legacy.rs` perform one private lookup, 1.6 incremental
decodes and approximately 2.6 CS access checks per register step. `State::resolve` / `fetch` in
`vm/legacy/paging.rs` additionally walk every required byte, validate/watch source, and
replace/register a physical proof even on hits. That is 1.6 walks and 3.2 PDE/PTE reads per register
step after warmup, plus the A-bit checks. `execute_with_retry` in `interp/legacy/paging.rs` prepares
on cloned CPU/task/temp state and executes again to commit, even for these register ops. Their
individual fixed shares are not separable with the available public probes. Memory/call loops add
2,127/2,619 host instructions over the paged register case, including longer fetch, checked
operands, translation plans and stack helpers.

## Proposed compiled identity

Use a distinct Machine namespace in the VM's shared translation cache:

```text
MachineBlockKey {
    physical_pc,             // final post-A20 physical bus address
    decode_ip,               // guest EIP, not interchangeable with physical_pc
    model_semantics_id,      // catalogue/lowering revision, including FPU policy
    context_flags {
        regime, CS_D, SS_B, segment_class,
        CPL, io_permitted, PG, WP_policy, A20_assumption,
    },
    translation_generation, // scoped proof epoch, zero for unpaged execution
    mxcsr_mode,             // MODE_MASK: RC/FTZ/DAZ
}
```

`io_permitted` is the CPL/IOPL permission comparison for protected execution, not the raw IOPL
field. A helper still consults the live TSS I/O bitmap when the comparison denies bypass. V86 is a
different regime and initially falls back; its sensitive-instruction rules must not reuse that
comparison blindly. Any other state folded into generated code must join the identity or be guarded;
CR0 coprocessor controls and descriptor contents may instead remain helper inputs.

`translation_generation` must not be a bare per-vCPU integer: two CPUs with epoch 1 and different
CR3 mappings must not collide. Recommend an opaque, VM-issued, non-reused proof token for a
translation domain/epoch, held by the per-vCPU TLB. Equal-value CR3 reload and explicit invalidation
obtain a fresh token. CR3 identity, model/WP/PG, A20 and physical-map changes belong to this proof's
reconciliation. No vCPU owner ID belongs in compiled identity. Initial conservative token separation
may reduce sharing; it must never allow a stale mapping. Reclaim obsolete entries with bounded
retention, not an ever-growing epoch-keyed cache. TLB eviction that observes a changed mapping must
revalidate the physical source, even inside an unchanged epoch.

Keep `BlockKey`'s public layout, constructors, Application hash byte stream and existing Direct
namespace unchanged. Add a separate Machine key/map alongside the existing cache dispatch maps,
sharing code lifetime, publication and physical coverage/invalidation machinery. Do not wrap the
existing Application key in a tagged union that changes its hash or hot lookup. Private Checked
caches remain the interpreter fallback; matching physical addresses never authorize conversion of
Application or Direct code into compiled protected code.

### Flat32 proof and block boundaries

J2 admits protected CPL 0, PG=0, CS.D=SS.B=1, usable ordinary CS/DS/ES/SS with base zero, full
32-bit limits and the required code/data/stack rights. This is a derived class, validated at
positive entry and after context changes, never a writable promise supplied by an embedder. FS/GS
overrides initially use a checked helper or cause fallback unless their caches satisfy the proof
too. Dynamic segment changes, privilege changes and control writes terminate the block.

Flat32 removes repeated descriptor/base checks; it cannot waive whole-operand limit faults at the
end of the 32-bit address range. Keep a centralized boundary check/slow path for those operands and
stack frames. This preserves the existing `legacy::access` contract (80386 PRM §§6.3.1.1–6.3.1.2,
pp. 6-4–6-6; SDM Vol 3A §§3.4.5.1, 6.3–6.4), rather than assuming Compat32 addressing alone
implements every Machine exception.

A normal compiled block stops at one **linear guest page** and registers its actual physical source
page. A straddling instruction is a singleton with both ordered source spans and both
mapping/permission proofs; fall back initially if the two-page compiled proof is unavailable.
Incremental fetch cannot speculate into an unmapped following page. Cross-page control transfers
return through the dispatcher. Source publication protects against concurrent writes and A/D
updates.

J3 uses the existing translation interface with a small per-vCPU TLB, split by access
privilege/permissions and the proof epoch. Start with the paging design's conservative invalidation
policy. Write hits are unavailable for code-tagged physical pages; slow stores retain physical
SMC/alias invalidation. Read-filled entries cannot become writable until D handling completes. No
host-page mapping substitute or second page walker is introduced. Details and hardware citations
remain in `backlog/docs/design/paging.md` §§1, 3.2–3.6.

## J1 accounting contract

For a block of N ordinary instructions, test remaining steps before entry. If the budget covers N,
reserve/decrement N at entry; finalize only completed units. Otherwise let the **existing Machine
interpreter** finish up to the exact remaining budget at this block's current IP. A zero step budget performs no
fetch, helper, completion consumption or pending-event cleanup. Application accounting is unchanged.

At a fault in zero-based instruction ordinal k, charge the k completed predecessors, refund the
reserved tail N-k, retain the faulting IP and count the faulting instruction zero. A completed
trap/HLT/port handoff at k counts k+1 and wins over coincident budget exhaustion. A device retry
preserves its original instruction/translation ownership and never counts an incomplete transfer
twice. `retired_instructions` advances only for completed whole instructions; it is not the current
approximate Application `executed_instructions` counter.

REP and other variable-unit instructions initially end a compiled block and use the
interpreter/helper's exact ledger: completed REP iterations consume steps, whole retirement occurs
only on completion, and empty REP consumes one. Delivery of a fault or hardware IRQ consumes no
instruction step. Shadow, ready-stop, single-step/trap and host-request boundaries keep
decision-16's ordering. The entry reservation alone cannot justify ignoring an IRQ becoming eligible
inside a block: split or use the exact fallback whenever an observable boundary is due.

J1 witnesses compare full state, memory, exits, steps and retirement across tiers for every budget
around a block boundary, a fault at each ordinal, partial REP, MMIO retries, port completion, SMC
and interrupt/debug boundaries. Run both backends with Machine compilation still gated before J2
admission.

## Recommended order and alternatives

Prefer **one bounded interpreter checkpoint before compiled execution**, reusing the intended
Machine key/proofs and Flat32 lowering. The target register loop needs about 680 host
instructions/step at this run's paged instruction-delivery rate; singleton dispatch alone costs 744,
so dropping segment checks alone is insufficient. Normal interpreted blocks cost 317, leaving useful
headroom. All gains below are engineering estimates for this register loop, not implemented
ablations or speed guarantees; ranges must not be multiplied.

| Order | Deliverable | Estimated result / evidence |
|---|---|---|
| 0 | Reproduce and repair xj-to9.6 PG=0 regression. | Recover the reported ~0.7% affected-cell work; negligible toward 35 M/s. |
| 1 | Shared paging proof/invalidation seam and per-vCPU TLB, retaining Checked singleton execution initially. | TLB alone roughly 1.05–1.25× paged register throughput (~2.0–2.4 M/s); marginal byte cost motivates a modest forecast. Operand-heavy code may gain more. |
| 2 | Interpreter Flat32 plus page-bounded multi-instruction paged blocks; warm hits validate existing proofs without rebuilding/re-registering them. Execute proven nonfaulting register IR once, without transaction preview. Retain existing checked transactions for operands/helpers that can fault. | PG=0 roughly 40–90 M/s (4–9×); PG=1 roughly 25–60 M/s (13–31× current). These require all three savings: fetch/proof reuse, block amortization, and removal of unnecessary preview/replay. TLB or Flat32 alone does not earn this estimate. |
| 3 | J1 exact compiled accounting, then J2 CPL-0 unpaged Flat32 compilation. | J1 is correctness work, with no speedup claimed. J2 roughly 150–600 M/s unpaged, allowing substantial overhead over the 1,257 M/s / 24.6-host-instruction Application JIT reference. |
| 4 | J3 compiled paging using the already witnessed TLB/proofs. | Planning range 60–200 M/s on this register loop; reserve much more work than Application for entry proofs, budgets and helpers. Re-measure memory and call/ret, not just register code. |
| 5 | J4 privilege, then J5 non-flat compiled segments. | Coverage stages, no speedup claimed. V86, real-mode compilation and 286 compilation do not enter early admission. |

Checkpoint 2 is a proposed extension to decision-23's order, requiring maintainer acceptance. It
avoids first implementing backend entry, compiled fault ordinals and new helper ABI plumbing for a
workload the existing interpreter can already execute at 95 M/s. It must reuse ordinary Machine IR,
never borrow Application decode/model semantics or grow a separate emulator. Witness source/mapping
changes, fetch policy, SMC aliases, page straddles, budgets and events before admitting multi-
instruction blocks. A no-fault fast lane requires a proved IR effect classification, not opcode
special cases for this ROM. Register-only results cannot license removing fault preflight from
memory operations. If this checkpoint expands into substantial duplicate machinery or misses 35 M/s,
stop optional interpreter tuning and proceed directly through J1→J2→J3; retain all reusable key/TLB
work. Straight J1→J2 cannot accelerate PG=1 before J3, while TLB-only or Flat32-only singleton
alternatives leave the dominant fixed costs. Do not defer J1's exactness to hit a throughput target.

The known xj-to9.6 approximately **+0.7% host instructions on some PG=0 cells** is an early repair
item, not a benefit claimed for this draft. Reproduce the supervisor's exact baseline/candidate
cells in release and perf; the gate is per-cell +0.5%, with a blocking regression only when over in
both profiles. This J0 measurement is neither that A/B campaign nor evidence that the regression is
repaired. Retain the existing Application hash and dispatch to protect that gate.

## Maintainer questions

| Question | Recommended answer |
|---|---|
| Insert the interpreter checkpoint or go straight to J1/J2? | Accept one checkpoint combining Flat32, cached page proofs and multi-instruction execution; continue to J1→J3 if it misses target or duplicates machinery. |
| Is bypassing paged preview/replay acceptable? | Only for proved nonfaulting ordinary IR; keep current transaction/fault/MMIO machinery for memory and system helpers until separately witnessed. |
| How should identity and sharing work? | Separate Machine map; physical PC + decode IP + model/context + VM-issued translation proof epoch + MXCSR. No owner key and no Application hash/layout change. |
| How broad is first Flat32 admission? | CPL 0, CS.D=SS.B=1, proved ordinary CS/DS/ES/SS; retain end-of-address-space checks and checked FS/GS fallback. Paging comes through the common TLB seam; privilege/non-flat/V86 compilation stays later. |
| What constitutes hitting the target? | ≥35 M retired/s on the embedder's actual paged register ROM with devices and exact 10M-step slices, release ARM64 macOS, 386 and 486; repository RAM loops and host counts are diagnostics, not a substitute. No 35 M/s promise for memory/call mixes. |
| What blocks acceptance before feature expansion? | Repair xj-to9.6 with the established two-profile gate; require exact-step/fault/SMC witnesses and unchanged Application identity at every stage. |

## Consequences and revisit trigger

Revisit the ordering if an early checkpoint misses its projected range, if device or event
boundaries dominate the embedder's ROM, or if J3 fails the 35 M/s target under the embedder's actual
step budget. Re-measure each accepted stage before committing to the next optimization; do not infer
future gains by multiplying independent upper bounds.

## Amendment (one JIT mechanism, key, tiering and stage plan, 2026-11-26, accepted by the maintainer)

Decider: maintainer (2026-11-26), accepting every recommended answer in the
amendment's "Maintainer questions" table (Q1-Q9) and the principle that the library
has one JIT mechanism for every mode. This amendment is the second half of J0 (bead
xj-to9.1). It changes no code. Where it differs from the record above (the compiled key:
the translation token becomes an entry check and WP a TLB-fill input; J3 and J4 merge;
an optional J2b for compiled-to-compiled chaining), the amendment wins; everything else
above stands. No Machine context enters a backend before J1 and J2 are accepted in their
own records (decision-23 §8).

### Context

#### What landed after decision-27

| Change | Commit | What it means for compiled code |
|---|---|---|
| One walk per block on the protected and paged paths | 574bd91d | Multi-instruction Machine IR is executed in one pass. This IR is what a backend would compile. |
| Fast hasher for the engine maps; 80286 segmented class; near JMP on the paged lane | 118b040c, 989a49fc, 08ddd454 | The interpreter's fixed costs went down. Compiled code has to beat a faster baseline. |
| Block links through direct exits, then near RET | 279f910f, 03a3ccbe | A per-vCPU memo (`vm/legacy/links.rs`) stands in for the resolver. It names the context id, proof version, paging id and TLB generation, and the proof generation. |
| Selective proof drops | 6758d3d5 | A drained code write drops only the private proofs whose bytes it overlaps (`cache.rs` written log, `vm/legacy/proof_map.rs`). |
| Code-alias tags stay set | 4712ce9f, e31a5e13 | A page of a declared alias view keeps its code tag once code ran there. Stores through any view take the cold code-write path. |
| Live instruction fields | ef36cb3c | A field patched twice is read at execution (`LegacyOp::LiveField`). A store wholly inside it is *absorbed*: it advances the publication generation but not the proof generation (`cache.rs:1165`, `:1173`, `:1654`). |

#### What the interpreter keys on today

| Path | Private key or identity | File |
|---|---|---|
| Unpaged Flat32 / Segmented proof | `LegacyBlockKey { owner_id, context_id, model_semantics_id, decode_bits, stack_bits, regime, decode_ip, linear_fetch, mxcsr_mode }` | `vm/legacy.rs:93`, built at `vm/legacy/flat32.rs:23` |
| Paged proof | `Key { segment_class, ordinary, decode: LegacyBlockKey, generation (TLB token), cr3, cr0 }` | `vm/legacy/paging.rs:15`, `:154` |
| Context identity | `LegacyContextStamp` (whole `CpuState::legacy` except CR2, selectors, IOPL/VM/RF) → a per-vCPU `context_id` | `vm/legacy.rs:123` |
| Translation token | `Tlb::generation`, from the process-wide issuer `paging_generation()`; fresh when a fetch-used entry is replaced, on CR3/CR0/A20 change | `interp/legacy/paging.rs:397`, `vm/legacy.rs:24` |
| Link | IP, physical, MXCSR, path, run, proof generation, and per path `[context_id, flat version]` or `[context_id, paging context_id, paging version, TLB generation]`, A20 on Direct | `vm/legacy/links.rs:95`, `:267` |
| Shared Application key | `BlockKey { guest_addr, mode, mxcsr_mode }` with the Machine Direct namespace in bits 31/30 | `cache.rs:21`, `:39` |

Every private key carries `owner_id` and `context_id`. Both are per-vCPU. Decision-27
already ruled that compiled identity is physical and not per-vCPU, so these names cannot
move into the compiled key as they are.

#### Measurement (J0, this tree)

§5 holds the J0 measurement decision-23 asked for, re-run at ef36cb3c. On the same
five-instruction bodies, the unpaged Flat32 Machine interpreter costs 426 / 567 / 1050
host instructions per guest step (register / memory / call-ret, release). The Compat32
JIT costs 18.5 / 41.9 / 69.0, a gap of 13.5–23×. Paged, the interpreter costs
595 / 890 / 1491. Since decision-27's baseline (2807 unpaged, 12,384 paged, register),
the interpreter checkpoint has cut 6.6× unpaged and 21× paged. What remains is mostly
the walk itself, and compiling removes it.

### Principle: one JIT mechanism

The library has one compiler and one tier-up path, whatever the profile. Long64,
Compat32 and Machine use the same `Backend` trait (`vm.rs:300-470`), the same Cranelift
backend and IR lowering, the same tier-up policy (`Vm::set_tier_up_after`, the region
threshold T2, `vm.rs:698-719`), the same background request, publish and stale-rejection
path (`TierUpRequest`, `Backend::tier_up_finished`, `TranslationCache::{try_begin_tier_up,
prepare_tier_up, publish_tier_up}`, `cache.rs:1355-1530`), the same shared cache and code
lifetime, and the same helper ABI (`jit_abi.rs` `MemCtx`). Where the profiles differ, the
mechanism takes a parameter. A Machine-only code path is allowed only where the table
below names it as an exception, and each exception is also listed as a question.

| Mechanism | Application (Long64, Compat32) | Machine | Kind |
|---|---|---|---|
| IR and lowering | `IrBlock`, `IrOp` | the same; `IrOp::Legacy(LegacyOp)` lowers to a helper call into the interpreter's op (decision-23 §8 "Machine IR is ordinary IR") | shared |
| Backend, codegen level, compile thread | `JitBackend`, `set_tiering`, one worker | the same worker and queue; Machine requests get no new thread | shared |
| Tier policy | `tier_up_after` (T1), `tier_up_background`, region T2 | the same knobs. Regions are limited to one guest page (decision-23 §8), so a Machine region is a loop inside one page | shared |
| Cache key | `BlockKey` in the Application map | `MachineBlockKey` in its own map of the same `TranslationCache` (§1) | parameter: namespace |
| Source coverage, invalidation, code lifetime | `SpanIndex`, `invalidate_overlapping`, `CodeLifetime` | the same structures. The index stores a unit identity that is either key type, so the Application map and its hash do not change | shared |
| Generation a request carries | publication generation (`cache.rs:1378`) | proof generation, so that an absorbed live-field patch does not cancel the request (§2.4) | parameter |
| Guest memory access | host mapping at `guest_base`, one baked `mmio` window, guard pages | physical bus: an inline probe of the per-vCPU software TLB, a miss helper that calls the existing walk (§3.2) | parameter: memory lowering |
| Self-modifying code on stores | `code_range_ptr` watermark, then the note-write helper (`jit_abi.rs:204`) | the same watermark test on the physical address after the TLB hit (§3.3) | shared |
| System instructions | none | `LegacyOp` system ops through helpers; privilege transfers and control writes end the block | parameter: which ops are helpers |
| Step accounting | off (compiled blocks do not tick `retired`) | exact (J1) | parameter |
| Dispatcher that enters compiled code | `vm.rs` `resolve_dispatch` / run loop | `vm/legacy.rs` `run_dispatch` (`:2837`) | **exception E1** |
| Where hotness is counted | shared `cache.bump_hotness` (`vm.rs:4181`, a locked map) | per-vCPU, beside the private proof and its link | **exception E2** |
| Publishing a finished compile | `cache.upgrade` replaces an `Interpreted` entry (`cache.rs:1502`) | inserts into the Machine map; there is no shared interpreted entry to replace | **exception E3** |

**E1** exists because the Machine dispatcher owns event delivery, the step budget, device
handoffs and the delivery log, and the Application loop has none of them. Merging the two
loops is out of scope. The compiled-block entry ABI is the same: a `CachedBlock::Compiled`
entry called with `CpuState` and `MemCtx`. **E2** exists because a Machine proof hit never
touches the shared cache. That is the point of the private proof maps and the links: the
ceiling study measured the shared lookup at about 190–220 host instructions per block
(performance.md, "The planned items"). **E3** follows from E2: Machine interpreted blocks
are private, so the shared Machine map holds only compiled units.

**Gains the other way.** Live fields could later serve Application compiled code too: the
compiled lowering of `LegacyOp::LiveField` (§3.1) is the part that Application lacks.
Today `absorb_written` refuses to absorb any byte that a shared unit covers. Exact step
accounting (J1) stays a parameter that Application leaves off. Selective proof drops give
Application nothing: its shared coverage is already byte-granular. From Application,
Machine inherits background compile, code-space reclamation, deterministic-failure
suppression (`interpreter_only`), compile-time stats (`BackendTimeStats`, `time_stats.rs`)
and the deferral evidence of `CriticalSectionKind::{Compile, Publish, TierUpWait}`
(`interruption.rs:10-23`).

### 1. The compiled Machine block key

```text
MachineBlockKey {
    physical_pc: u64,          // post-A20, post-translation bus address of decode_ip
    decode_ip: u32,            // EIP; equals the linear address only under Flat32
    model_semantics_id: u64,   // catalogue entry and lowering revision, FPU policy included
    flags: u32,                // regime, CS.D, SS.B, address bits, segment class, CPL,
                               // io_permitted, PG, A20
    mxcsr_mode: u32,           // RC | FTZ | DAZ (SDM Vol 1 §10.2.3)
}
```

For each input the interpreter or decision-27 uses, the table says where it belongs.

| Input | Compiled key | Entry check | Neither (private only) | Why |
|---|:-:|:-:|:-:|---|
| physical PC | ✓ | | | The bytes compiled. Shared coverage protects them |
| decode IP | ✓ | | | Decision-18 §3; under J5 the same physical bytes may run at another EIP |
| model semantics id | ✓ | | | Lowering differs per model |
| regime, CS.D, SS.B, address width, segment class | ✓ | | | They select decoding and lowering |
| CPL, `io_permitted` | ✓ | | | Baked: privilege-sensitive ops are lowered for one CPL. `io_permitted` is the CPL ≤ IOPL comparison, not raw IOPL (decision-27) |
| PG | ✓ | | | Selects the memory lowering |
| A20 | ✓ | | | Unpaged code bakes the A20 mask. Paged code reads it through the TLB, but keeping it in the key is cheap and avoids a second lowering |
| MXCSR mode | ✓ | | | As Application |
| WP | | | ✓ | **Changed from decision-27.** WP is a TLB-fill input (paging design §3.2). The probe checks rights computed at fill, so compiled code does not bake it |
| Translation token (TLB generation) | | ✓ | | **Changed from decision-27**, see below |
| CR3 | | ✓ (through the token) | | As above |
| `owner_id`, `context_id`, Flat32/paged proof version | | | ✓ | Per-vCPU. They guard the private proof that holds the pointer to the compiled unit (§3.1) |
| Proof generation | | | ✓ | Guards private proofs and links. The shared unit is guarded by byte coverage |
| Publication generation | | | ✓ | Only for the request's source-read window (§2.4) |
| Segment bases and limits beyond the class | | (J5) | | Flat32 implies them. Segmented code (J5) reads bases and limits from `CpuState` at run time, as Application reads FS/GS bases |
| Code-lifetime epoch | | ✓ | | A vCPU's cached pointer is checked against `TranslationCache::epoch` before entry |

**The translation token leaves the key.** Decision-27 put a translation proof epoch in the
compiled key. Measured behaviour argues against it. The token is reissued on every CR3
write, every PG/WP/A20 change and every replaced fetch entry
(`interp/legacy/paging.rs:423-500`). A desktop OS reloads CR3 on every process switch, so
with the token in the key every switch would orphan the compiled code of shared kernel
and library pages and compile them again. Compiled code does not need the token:

- a block covers one guest page, so its only mapping dependency is "this linear page maps
  to `physical_pc`'s frame, executable at this CPL";
- the Machine dispatcher already translates the fetch address on every turn to find the
  proof (`vm/legacy.rs:3238-3241` unpaged, the paged resolver otherwise). Finding a key
  with this `physical_pc` *is* that mapping check;
- data accesses inside the block translate at run time through the TLB. Nothing linear
  is baked except intra-page branch targets, which the one-page rule keeps in the same
  frame.

So the token stays where it is today: in the private paged proof key and in the link
identity, which guard what the interpreter itself bakes. The compiled key is what the
bytes and the lowering depend on.

Straddling instructions stay interpreted, as decision-27 allows ("fall back initially").
A compiled block ends before an instruction that crosses its page.

### 2. Tiering

Machine tiering reuses the Application tier-up path (`vm.rs:300-470`, `:3999`,
`:4171-4300`; decision-4), with three kinds of difference:

- **Unchanged:** the T1 / T2 / background knobs, `TierUpRequest` / `TierUpUnit` /
  `TierUpFinished`, the compile worker, `Busy` backpressure, the in-flight slot, source
  registration in `pending_sources`, cancellation by overlapping writes, rejection of stale
  completions, the `interpreter_only` failure suppression, code-lifetime ownership, and the
  stats and interruption kinds.
- **Changed by a parameter:** the unit key (`MachineBlockKey`, so the request echoes a key
  instead of a bare `pc`), the generation snapshot (proof generation, §2.4), the source
  spans (live fields left out, §3.1), the publish target (insert, E3), and where hotness
  is counted (E2).
- **New:** the admission test (§2.1 item 4), the SMC-hot page list and demotion (§2.5),
  the compiled pointer cached in the private proof and the link (§3.1), and the J1
  accounting in generated code (§4).

#### 2.1 When a Machine block is compiled

A private proof (Flat32, Segmented, paged) becomes a compile candidate when all of the
following hold:

1. **Hot.** It ran `tier_up_after` (T1) times. The count lives with the proof and is also
   bumped when a link stands in for the resolver (E2). A loop entirely inside one page
   may become a region at T2, as Application loops do.
2. **Stable.** The count restarts whenever the proof is dropped, so "T1 executions without
   an invalidation" holds by construction: a selective drop (6758d3d5) creates a new proof
   with a new count.
3. **Not SMC-hot.** The page is not on the per-vCPU list of pages where a code write
   dropped a compiled unit (§2.5). The list extends the 16-entry `smc_pages` hint
   (`vm/legacy.rs:584`) with a second bit.
4. **Admissible.** The context is in the admitted class of the current stage (J2: Flat32,
   PG=0, CPL 0; see §6). No fetch policy is installed: links and protected proofs already
   refuse to run under one (`links.rs` module comment), and compiled units would need
   permits checked on every hit. The block is not a retained (self-modifying) twenty-bit
   or Checked block, has no pending continuation and is not a straddle.
5. **Live fields lowered.** A field already live in the proof's IR is compiled as a load
   (§3.1). The compile never bakes a live field.

#### 2.2 Background compilation: the interpreter never waits

The maintainer's requirement: until the JIT has compiled a block, the interpreter runs it,
and the switch is smooth. Application already works this way. Machine uses the same path:

- On reaching T1, the Machine dispatcher claims a slot (`try_begin_tier_up` with a Machine
  key), registers the source spans (`prepare_tier_up`) and submits a `TierUpRequest`
  carrying the proof's already-lifted `Arc<IrBlock>`. It keeps running the interpreted
  proof.
- `TierUpSubmit::Busy` and `Unsupported` mean "stay interpreted". A Machine block is never
  compiled inline on the vCPU in production.
- At the top of each `run_dispatch` turn the Machine dispatcher drains finished compiles,
  as `drain_tier_up` (`vm.rs:3999`) does at the top of the Application loop. It runs under
  `EngineSection` with `CriticalSectionKind::TierUpWait` and `Publish`, so host-request
  deferral evidence stays the same.
- The next turn that resolves that proof, or follows its link, enters compiled code.

**Synchronous mode for witnesses.** With `tier_up_background` off, the Machine dispatcher
compiles inline at T1, as Application does. That gives a deterministic switch point.
Witnesses use it with T1 = 1 ("compilation forced on").

**What background compilation needs for exactness.** Interpreter and compiled code execute
the same IR, and the switch happens only at a block boundary. Results and step counts are
therefore the same whenever publication lands, *provided that* J1 makes compiled blocks
exact. No guest-visible state depends on when the compile thread finishes. That is the
determinism rule of decision-23 §8. It is witnessed by running each Machine witness three
ways: interpreter only, synchronous T1 = 1, and background with a test hook that delays
publication by k turns.

#### 2.3 Races and how the existing path covers them

| Race while the compile is in flight | Covered by |
|---|---|
| A write to the source bytes before span registration | the request's claim generation (`prepare_tier_up`, `cache.rs:1444`) |
| A write after registration | `pending_sources` coverage. An overlapping invalidation cancels the request, and `publish_tier_up` rejects it (`cache.rs:1471-1500`) |
| Live-field promotion over the block's bytes | promotion is a recorded advance over the instruction's bytes (`cache.rs` `advance_for_promotion`). It overlaps the pending sources and cancels the request |
| Context or CPL change | none needed: the flags are part of the key, so the unit is valid under its own key |
| CR3 / TLB change | none needed: the token is not in the key (§1) |
| Code-lifetime reset, unmap, code-space reclamation | the code-lifetime epoch: publication fails, and cached pointers fail their entry check |
| Deterministic compile failure | `interpreter_only` until the bytes change, as Application |

#### 2.4 Which generation a Machine request carries

Application requests snapshot `publication_generation()` (`cache.rs:1378`, `:1457`).
Since ef36cb3c that counter also advances on every absorbed live-field patch. A Machine
request carrying it would be cancelled by every patch in its claim window, on exactly the
pages where live fields were meant to keep code stable. Machine requests therefore carry
`proof_generation()`, the generation the private proofs and links use. Absorbed patches
do not advance it, and every other code write does. The shared map's source registration
must also leave live-field bytes out (§3.1), or `absorb_written` will refuse to absorb
(`cache.rs:1654-1664`: any shared coverage of the bytes refuses) and every patch will drop
the compiled unit.

#### 2.5 Demotion

A compiled unit dropped by a code write that overlaps it adds its page to the vCPU's
SMC-hot list. If the same page drops compiled units **twice** within 2^20 retired
instructions (retired instructions keep the policy deterministic, with no host time),
the page stops compiling until 2^20 retired instructions pass with no drained code write
to it. Its proofs keep running interpreted, with selective drops and live fields. These
numbers are starting values to tune against the SMC benches (`machine32_smc_patch`,
`machine32_smc_live`). They affect cost, never results.

### 3. Interaction with the interpreter's mechanisms

#### 3.1 Private proofs, live fields and the shared unit

- The private proof stays the per-vCPU entry point. After publication it caches
  `(CompiledPtr, epoch)`. A turn that resolves the proof, or follows its link, compares
  the epoch with `TranslationCache::epoch()` (one atomic load), then enters compiled code.
  On a mismatch it looks the unit up again in the Machine map, on a cold path. A link slot
  (`links.rs:95`) gets the same optional pointer, so a linked turn enters compiled code
  without the resolver.
- The shared unit registers its physical source bytes **minus the bytes of live fields it
  reads live**. A store wholly inside a live field is then absorbed as today: no proof drop,
  no compiled-unit drop, no link end. A store that touches any other byte of the
  instruction hits the unit's coverage and invalidates it, exactly as it drops the private
  proof. A field demoted later is still read live by the compiled unit, so the unit stays
  exact.
- `LegacyOp::LiveField` lowers to a load from its bus address. That address lies in
  `physical_pc`'s frame because straddles are excluded, so it needs no TLB probe. The
  interpreter's reading rule holds: one read per execution of the instruction, before the
  instruction's own stores (`live.rs` module comment).
- Selective drops need no change. A drained write that overlaps a private proof drops
  that proof and its cached pointer. If it also overlaps the shared unit, it invalidates
  the unit and advances the epoch, so the other vCPUs' cached pointers fail their entry
  check.

#### 3.2 Memory accesses in compiled Machine code

One lowering is used for every Machine memory access, paged or not: an inline probe of the
per-vCPU software TLB. Unpaged contexts fill identity entries from the region map. The probe
is the interpreter's `Tlb` (`interp/legacy/paging.rs:397`), given a `repr(C)` layout with a
`MemCtx` pointer to it: one translation interface, never a second TLB or walker
(decision-23 §8). A hit checks page tag and rights (U/S, R/W, fast-read, fast-write) and
adds the host offset of the frame. A miss, a page-crossing access, Trap and ROM writes go
to one helper that runs the existing checked access and its fault, MMIO and A20 rules. The
end-of-address-space operand check of decision-27 needs no inline code: an operand that
wraps at 4 GiB crosses a page and takes the helper.

#### 3.3 Stores into code

After a TLB write hit, compiled code applies the Application `code_range_ptr` watermark to
the physical address and calls the note-write helper only inside it. This **refines
decision-23's "write entries exclude code-tagged pages"**. Excluding tagged pages from
write entries would make every `mark_code` shoot down write entries in every vCPU's TLB.
The watermark is already read live, so a page tagged by a compile in flight on another
thread is seen by the next store (`jit_abi.rs:180-204`), and Application already pays it.
Code-alias views keep working unchanged: `note_code_write` expands aliases
(`memory.rs:2168`), and alias pages keep their tag (e31a5e13).

The watermark has one known risk. It is a single contiguous range of 64 KiB granules. On a
PC-class bus that runs code both in low memory and in firmware just below 4 GiB, the range
covers almost every address, and then every store calls the note-write helper. If J2
measures this, the fallback is still the shared tag table, read live. The TLB entry
caches the address of its frame's code-tag word, and a store tests that one bit: one load
and one test, with no index arithmetic, and no shootdown, because the bit itself is live
(Q9). A store that overlaps the
remaining bytes of the *running* compiled block ends the block after the storing
instruction, with that instruction counted (J1; SDM Vol 3A §14.6 p. 14-18, as the
interpreter applies it).

#### 3.4 Exits and chaining

A compiled Machine block returns to `run_dispatch` at its exit IP with a block-result code,
as Application blocks return to theirs. The dispatcher then runs its turn: drain, events,
budget, links. **Compiled-to-compiled chaining is not J2 scope.** Chaining skips the turn,
so it needs J1's budget test at every compiled entry and a dispatch-word poll at every
chain edge (Application chain edges already poll `dispatch_word_ptr`). It also has to end
on any link-invalidating event. It is planned as J2b, inside one page first, and only
after J2's per-turn cost has been measured (§5, Q4).

### 4. Exactness carried into J1

J1 implements these, and J0 does not. They restate decision-27's J1 contract with what
compiled code adds:

- **Step budget.** Test the remaining budget at entry against the block's N instructions.
  If it covers N, reserve N. Otherwise the interpreter runs the block at this IP (the
  existing `remaining` cap in `run_dispatch`). A zero budget enters nothing.
- **Retired count.** Compiled Machine blocks tick `retired_instructions` exactly: whole
  instructions only. REP and other variable units end the block and run through the
  interpreter's ledger.
- **Mid-block faults.** A fault at ordinal k charges k, refunds N−k, keeps the faulting
  IP and leaves the state the interpreter's rollback leaves (`leave_instruction` /
  `roll_back`, 574bd91d). A port, HLT or trap handoff at k counts k+1.
- **Events.** IRQ eligibility, interrupt and SS shadows, TF, NMI and the exit word are
  decided at block boundaries. A block cannot be entered when a boundary is due inside
  it; the interpreter covers that case.
- **Delivery log.** Deliveries are recorded only at the existing delivery sites. Compiled
  code adds no record, and delivery from a compiled fault goes through the same routing
  (`route_legacy_exception`).
- **Time stats and cycles.** Compile time goes to `BackendTimeStats`. With timing level 2,
  a compiled block adds its static cycle sum at entry and its run-time parts where they
  occur (decision-23, "Interpreter and compiled accounting").
- **Device handlers.** In-process device access (decision-28) is reached through the same
  helper the interpreter calls. A handled access continues the block.

### 5. Measurement

**Method.** The tree is main ef36cb3c, unchanged. `x86jit-bench/examples/machine32_loops`
was built in the default `release` profile and in the `perf` profile (codegen-units 1
for core and bench), rustc 1.98.1, Apple M5 Pro, macOS 27.0.1. Each cell is one path and
one workload per process (`machine32_loops 2000000 1 PATH WORKLOAD`), three fresh
processes per cell through `scripts/perf_campaign.py::run_process` with ASLR off, in
alternating path order. A sample is 10,000,001 retired instructions after a
10,000-iteration warm-up. The counter is `proc_pid_rusage` `ri_instructions` around
execution only, as performance.md uses it. The model is I80386, with flat full-limit
segments, CPL 0 unless named, and identity paging in RAM tables. Application paths run
the same bytes; Application JIT is Cranelift `Speed`, eager. The host load average was
9–15 from other work, so wall-clock rates are not reported. Host counts spread under
0.2 % across processes; the brackets give min–max.

**Host instructions per guest step (median of 3 processes, MEASURED):**

| Path | register (release) | memory (release) | callret (release) | register (perf) | memory (perf) | callret (perf) |
|---|---:|---:|---:|---:|---:|---:|
| `machine-protected32` (Flat32, PG=0) | 426.2 [426.1–426.3] | 567.1 [566.9–567.2] | 1050.1 [1050.1–1050.8] | 422.2 | 569.3 | 1039.8 |
| `machine-protected32-cpl3` | 426.1 | 566.8 | 1050.0 | 422.1 | 569.2 | 1039.8 |
| `machine-paged32` (Flat32, PG=1) | 595.0 [594.7–595.2] | 889.5 [889.3–889.8] | 1491.1 [1491.1–1491.7] | 582.4 | 877.0 | 1453.9 |
| `machine-paged32-cpl3` | 598.3 | 892.3 | 1494.6 | 585.5 | 880.1 | 1456.9 |
| `application-interp32` | 276.7 [276.6–276.8] | 299.2 [299.1–299.3] | 472.9 [472.8–472.9] | 280.8 | 308.8 | 468.9 |
| `application-jit32` | 18.5 [18.4–18.5] | 41.9 [41.9–41.9] | 69.0 [69.0–69.0] | 18.6 | 42.1 | 69.5 |
| Machine Flat32 ÷ Compat32 JIT | 23.0× | 13.5× | 15.2× | 22.7× | 13.5× | 15.0× |
| Machine paged ÷ Compat32 JIT | 32.2× | 21.2× | 21.6× | 31.3× | 20.8× | 20.9× |

CPL 3 costs what CPL 0 costs, within 0.6 %. The unpaged Machine interpreter is now 1.5–2.2× the
Application interpreter on the same bytes; it was 8.9× at decision-27.

**Expected compiled Machine range (ASSUMED).** Model:
`per step = J + t·T + a·P + t·E`, where

- J is the Compat32 JIT's per-step cost on the same body (measured above);
- t is dispatcher turns per guest step without chaining: register and memory run one
  block per 5 steps (t = 0.2), and callret runs three blocks, `[CALL]`, `[INC; RET]` and
  `[DEC; JNZ]` (t = 0.6);
- T is the cost of one Machine dispatcher turn whose link hits, entry check included.
  **Assumed 150–400** unpaged and **250–800** paged. This is not isolated by any probe
  today: it is bounded above by the interpreter's 2,130 host instructions per register
  iteration, which includes a five-instruction walk, and the paged−unpaged difference
  (169 per step, 845 per iteration) is assumed to be mostly per-turn reconciliation;
- a is memory accesses per step (0.4 for memory: one load and one store; 0.4 for callret:
  one push and one pop; 0 for register);
- P is one inline TLB probe on a hit. **Assumed 10–20 host instructions**: index hash,
  tag and rights compare, frame add, plus the store watermark compare. The probe is
  charged on unpaged Flat32 too, because J2 uses the same lowering (§3.2);
- E is the J1 entry and exit bookkeeping: **assumed 5–15 per turn**.

| Workload | Unpaged Flat32 compiled | vs interpreter now | Paged compiled | vs interpreter now |
|---|---:|---:|---:|---:|
| register | 50–102 | 4.2–8.6× | 70–182 | 3.3–8.6× |
| memory | 77–133 | 4.3–7.4× | 97–213 | 4.2–9.2× |
| callret | 166–326 | 3.2–6.3× | 226–566 | 2.6–6.6× |

At the register loop's 50–182 host instructions per step, decision-27's 35 M retired/s
target needs only about 1.8–6.4 G host instructions/s, well inside one performance core.
The interpreter today needs 15–21 G at this cost. The ranges are planning numbers, not
promises. J2's first measurement replaces T, P and E with counted values, and chaining
(J2b) removes most of t·T.

### 6. Stage plan, refined

A late-1990s desktop OS runs flat 32-bit code with paging, its kernel at CPL 0 and its
applications at CPL 3. Its hot code is J3 + J4. Firmware, DOS extenders and boot code
are where J2 applies. Its 16-bit protected components and V86 boxes can stay interpreted
(the Segmented class and paged V86 run at about the paged interpreter's cost already).
So J5 moves to last, and J3 and J4 are best built together (Q2).

| Stage | Content | Done when | Witnesses |
|---|---|---|---|
| J1 | Exact accounting in compiled code (§4), with the Machine key, the Machine map, request/publish (§2) and the entry check (§3.1), all behind a test-only admission. | Every J1 witness passes with compilation forced on (T1 = 1, synchronous) and with background publication delayed. No Application cell regresses past the perf gate. | jit==interp lockstep on state, memory, exits, steps, retired, delivery log: budgets 0..N+1 around every block boundary, a fault at each ordinal, partial REP, MMIO retry, port completion, IRQ/TF/NMI boundaries. |
| J2 | Admit Flat32, PG=0, CPL 0 to the backend. TLB-probe lowering with identity entries (§3.2), watermark SMC (§3.3), live fields as loads, links carrying compiled pointers. | `machine32_loops` machine-protected32 register/memory/callret inside the range of §5. SMC, alias and live witnesses pass with compilation forced on. | Re-run with forced compilation: `legacy_one_pass`, `link_witness`, `return_witness`, `legacy_smc_selective`, `legacy_code_alias_smc`, `live.rs` witnesses, threads `same_block_smc`. A compile racing a patch, a promotion and a code-lifetime reset. |
| J2b (optional) | Compiled-to-compiled chaining inside one page. | J2's per-turn cost is the largest remaining item on callret. | Chaining on/off lockstep under budgets and events; every link witness with chaining. |
| J3+J4 | Paged Flat32 at CPL 0 and CPL 3: TLB fill by the existing walk, U/S and R/W rights in the probe, CPL and `io_permitted` in the key, privilege transfers and `INT`/`IRET` through helpers ending the block. | machine-paged32 and machine-paged32-cpl3 inside the range of §5. The embedder's paged ROM reaches decision-27's ≥ 35 M retired/s target. | Paging witnesses (`legacy_paging`) forced compiled: CR3 reload, INVLPG, A/D bits, #PF at each ordinal, WP at CPL 0, U/S faults, a CR3 switch keeping compiled kernel pages (key without token), SMC through a second linear mapping. |
| J5 | Segmented classes (16-bit protected, non-zero bases), V86 later or never. | A measured workload needs it. | Segmented witnesses forced compiled. |

### 7. Code that changes in J1–J3 (for planning; nothing changes in J0)

| Place | Change | Stage |
|---|---|---|
| `vm.rs:191`, `:210`, `:233`, `:254`, `:499`, `:4612`; `x86jit-cranelift/src/lib.rs:4223`, `:4296` | The `legacy_context` guards stay until J2. They then admit IR whose context is in the admitted class, and still reject everything else. | J2 |
| `cache.rs:21` | `BlockKey` unchanged. New `MachineBlockKey` and a Machine map beside `map`; `SpanIndex` and `pending` take a unit identity covering both. | J1 |
| `cache.rs:1355-1530` | `try_begin_tier_up` / `prepare_tier_up` take the generation to check (publication or proof). `publish_tier_up` inserts into the Machine map instead of upgrading. | J1 |
| `vm.rs:390`, `:454` (`TierUpRequest`, `TierUpFinished`) | Carry the unit key, not only `pc`. `drain_tier_up` (`vm.rs:4006`) stops deriving the key from `vm.mode`. | J1 |
| `vm/legacy.rs:2837` `run_dispatch`, `:3283`, `:3405-3470` | Drain at turn start. After resolve or link: entry check, then the compiled call; the interpreted walk otherwise. | J1 |
| `vm/legacy/flat32.rs` `Proof`, `vm/legacy/paging.rs:36` `Proof`, `links.rs:95` `Link` | Hotness count and cached `(CompiledPtr, epoch)`. | J1 |
| `interp/legacy/paging.rs:382-500` `Tlb` | `repr(C)` entries readable from compiled code; fast-read/write bits from region kind. | J2 |
| `jit_abi.rs` `MemCtx` | Pointer to the vCPU's TLB; Machine step-budget slot (fields appended; frozen offsets unchanged). | J1/J2 |
| `vm/legacy/live.rs`, `cache.rs:1654` | Compiled units register sources minus live fields; `absorb_written` unchanged. | J2 |

### Options considered

| Choice | Options | Recommendation |
|---|---|---|
| Tier-up machinery | a Machine-specific tiering path; the shared one, parameterised | **Shared**, with exceptions E1–E3 |
| Translation token | in the key (decision-27); an entry check | **Entry check**: compiled code survives a CR3 reload |
| Compile timing | synchronous at T1; background | **Background** in production, synchronous for witnesses |
| Unpaged memory lowering | a direct physical mapping with region checks; the TLB probe with identity entries | **TLB probe**: one lowering, and J2 measures the cost J3 pays |
| SMC on compiled stores | write entries exclude tagged pages; the watermark after a hit | **Watermark**: no cross-vCPU shootdown, the same as Application |
| Chaining | in J2; later | **Later** (J2b), after the per-turn cost is measured |

### Maintainer questions

| # | Question | Options | Recommendation |
|---|---|---|---|
| Q1 | Take the translation token out of the compiled key (§1)? It amends decision-27. | (a) out, entry check by the dispatcher's own fetch translation; (b) keep decision-27's token in the key | **(a)**. With (b), every CR3 reload orphans shared kernel code. |
| Q2 | Build J3 and J4 as one stage? | (a) one stage, U/S rights in the probe from the start; (b) separate as in decision-23 | **(a)**. The target workload needs CPL 3 at once, and CPL is a key flag and a rights bit. |
| Q3 | Accept the three exceptions to "one mechanism"? E1 separate dispatcher, E2 per-vCPU hotness, E3 insert instead of upgrade. | (a) accept; (b) count hotness in the shared cache too (+190–220 host instructions per block hit); (c) merge the dispatchers first | **(a)**. (b) costs more than the hot code it finds; (c) is a large refactor with no gain for compiled code. |
| Q4 | Chaining scope | (a) J2b after J2's measurement; (b) part of J2; (c) never, links only | **(a)**. |
| Q5 | Unpaged lowering | (a) TLB probe with identity entries; (b) direct physical map with region checks | **(a)**. Revisit if J2 misses its range. |
| Q6 | Is Machine compilation opt-in? | (a) on only with `set_tier_up_after` + background, as Application; (b) on by default for Machine profiles | **(a)** until J3+J4 lands, then reconsider. |
| Q7 | Demotion constants (§2.5): two drops per page within 2^20 retired, 2^20 quiet to recover | (a) as stated, tuned on the SMC benches; (b) per-key instead of per-page | **(a)**. Per-page matches the code-tag granularity. |
| Q8 | Should Application compiled code adopt live fields later? | (a) separate bead after J2; (b) no | **(a)**, low priority: no Application workload has shown the pattern. |
| Q9 | SMC test on compiled Machine stores (§3.3) | (a) the Application watermark, with the live tag-word bit from the TLB entry as a measured fallback; (b) the tag-word bit from the start; (c) write entries never installed for tagged pages (decision-23), with cross-vCPU shootdown on `mark_code` | **(a)**: it is the shared mechanism. Switch to (b) if J2 measures a near-full watermark range on a PC-class layout. (c) misses stores racing a background compile's tagging. |

### Consequences

- J1 starts with no new tiering design: it adds a key type, a generation parameter and a
  Machine call site to the existing path.
- Decision-27's key changes in two fields (token, WP). Its J1 contract, Flat32 admission
  and witness gates stand.
- Every Machine witness gains a forced-compilation variant once J1 lands. The witness
  suites grow, so the slow tier takes the delayed-publication runs.
- Revisit this amendment if J2 misses its range by more than 2×, if the per-turn cost
  dominates callret after J2 (then J2b moves up), or if the embedder's paged ROM misses
  35 M retired/s after J3+J4.
