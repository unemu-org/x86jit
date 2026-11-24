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
