---
id: decision-28
title: 'In-process device access on Machine profiles'
date: '2026-11-25'
status: accepted
---

Decider: maintainer (2026-11-25), accepting the shape below and the recommended
answer to every question in "Maintainer questions". Tracked as xj-to9.17; the
cheaper Exit/resume path is xj-to9.15 and lands first. It amends spec.md §5.1
(the core did not call the embedder in the hot path) and extends decision-23's
device timing seam; decisions 18, 23 and 27 otherwise stand.

# In-process device access on Machine profiles

## Context

Machine port and MMIO transfers currently return through `Exit` and completion/retry APIs. Reported embedder measurements show 79–99% of host time inside `run_steps`, but only 18–59 retired instructions per call: 1.35M video-plane MMIO exits per 49.9M retired instructions, and 3.55M diagnostic-port exits per 79.7M. Throughput is approximately 10 M instructions/s versus 35 M without exits. These are supplied measurements, not newly reproduced results.

The maintainer accepted two complementary approaches in principle: cheaper Exit/resume handling, separately tracked, and optional in-process device access.

This proposal amends the callback prohibition in spec.md §5.1. It preserves the Machine profile separation, exact accounting, physical-memory boundary and compiled-execution requirements of decisions 18, 23 and 27. Devices remain embedder code.

## Decision

### Registration and API

Add a synchronous, object-safe `DeviceIo` trait with one access method receiving a request and a borrowed `DeviceContext`. Register `Arc<dyn DeviceIo + Send + Sync>` objects in a **per-VM** registry. Closures may use an adapter.

Use separate, non-overlapping half-open ranges:

- Ports: byte-granular port numbers within the 16-bit namespace. Lookup uses the addressed port; width remains 1/2/4 bytes.
- MMIO: byte-granular physical ranges within existing `RegionKind::Trap` mappings, after translation, physical masking and A20 gating.

Reject empty, overflowing or overlapping registrations. An MMIO transfer must fit wholly within one registration; otherwise retain its existing Exit. Registration boundaries never introduce additional bus splitting. Unsupported widths fall back.

Registration, replacement, unregistration and relevant mapping changes require all VM execution to be quiescent, with no outstanding handoff. Reject changes during execution or callbacks. Return registration handles for explicit removal.

### Handler contract

Requests contain address space, port/physical address, direction, transfer size, write value, operand width/part metadata, originating instruction/event identity, and a stable VM-local vCPU identifier. Preserve available linear-address evidence separately.

Expose exact counts at the existing handoff point: retired instructions, call-local steps and the unit’s starting position. Provide optional modeled-cycle information when supported. REP progress must not be inferred from retirement alone.

Results are:

- **Handled:** read value or write acknowledgement.
- **Fallback:** issue the ordinary Exit for this transfer.
- **DeviceFault:** return a new typed host-facing stop carrying request/error evidence.

Fallback and DeviceFault require no effects from the current callback, including RAM writes or queued signals. Earlier acknowledged transfers remain effective. DeviceFault preserves the underlying handoff’s counts and pending state; it is not a guest #PF/#GP. Recovery uses explicit completion or abandonment, never automatic redispatch.

`DeviceContext` permits checked physical RAM reads/writes, queuing a vector through existing IRQ semantics, NMI assertion, host-stop requests, and deferred A20/paging-invalidation requests. Device-controller arbitration remains external. RAM writes use normal dirty/SMC notification. No raw pointers, recursive MMIO, CPU-register mutation, mapping changes, registry changes, completion calls or nested execution are permitted.

Callbacks execute synchronously without engine locks held. They must be bounded and non-reentrant. `&self` plus `Send + Sync` leaves device synchronization with the embedder; the engine promises per-vCPU transfer order, not deterministic ordering between vCPUs or new SMP guarantees.

Catch unwinding panics before crossing a compiled helper ABI. Report a terminal handler failure and poison execution: effects may be unknown, so normal resume is forbidden. Abort panics remain process termination.

### Exactness and restart

Dispatch only where the existing engine would issue a real device handoff, after its required segmentation, privilege, paging and transfer preflight. Never call handlers during speculative transaction preparation, dummy reads or probes. Preserve staged frame-read handoffs and existing facility refusals.

Handled transfers enter the existing completion/visit ledger immediately. Replay consumes recorded results without calling the device again. Retained instruction bytes, operand snapshots, translation ownership and abandonment rules remain authoritative.

Preserve current units and accounting:

- Ordinary instructions count once; REP counts iterations and retires the whole instruction only at completion. Empty REP counts once.
- Ports retain handoff-time charging. Supplying IN data and completing INS’s pending store add no step.
- MMIO contributes no step until its instruction/iteration completes.
- Zero-budget calls invoke no handler and consume no completion.
- Service an issued port handoff before testing exhaustion. Finish its required completion without starting another unit; fallback still returns the port Exit even on the last step.

REP INS/OUTS remain one port access per element. INS retains its answer across destination faults/MMIO; OUTS retains source-transfer results. MOVS and other MMIO strings use existing operand/iteration ledgers. Clear an iteration’s ledger before the next element.

Keep existing transfer widths, unaligned splits, A20/page fragments, RAM/device write order, locked read/write sequences and typed refusals. Registration neither merges transfers nor upgrades atomicity. No interrupt is accepted between pieces of an unfinished unit. Existing LOCK#/DMA and cross-vCPU limitations remain.

Device-raised interrupts become pending immediately and are considered at the next eligible instruction or REP boundary, respecting completion holds, IF, shadows, NMI rules, debug handling and budget precedence. Host-stop requests use existing deferral rules.

Delivery logging remains exclusively in the normal delivery sites: identical entered deliveries, order, saved state and retired values; device accesses add no delivery records. Preserve the log’s existing checkpoint and overflow behavior (spec.md, delivery log).

### Engine, time and ownership

A handled access continues execution within the current run. Blocks may retain their suffix when valid; internal redispatch is permitted initially. A handler’s code write or context/proof change forces validation/invalidation before another instruction executes, including compiled regions and chains.

Integrate below checked Machine access/transaction machinery, using `LegacyMemory` and the paging transfer ledger. Do not add a parallel walker or device emulator.

RAM writes retain physical-alias SMC and watched-data tracking. `watch_range` remains page-granular RAM-write observation, not device dispatch; private device storage needs embedder tracking.

Preserve `Fast`/`AcqRel`/`FullTso` access/helper ordering, including failure paths. A callback call alone is not a memory fence.

Provide device-only extra-cycle charging and an optional next-service deadline checked at eligible boundaries. Report unknown timing explicitly where cycle accounting is unavailable; never substitute host elapsed time. This narrowly extends decision-23’s device timing seam, without adding callbacks to ordinary RAM accesses.

Snapshots require quiescent CPUs and completed handoffs, or explicit abandonment. Device state, pending board events and registration reconstruction belong to the embedder. Trait objects and private continuations are not serialized; restore invalidates derived paging/code state.

Application profiles, including the Real16 alias, remain outside this proposal. `x86jit-core` dependencies stay exactly `{iced-x86}`.

## Options considered

| Choice | Recommendation and reason |
|---|---|
| Trait objects, closures, or built-in devices | **Trait objects**, with closure adapters: shared ownership and one future compiled helper contract; no core device models. |
| Per-VM or per-vCPU registration | **Per-VM**, with vCPU identity in requests: one physical device map without duplicated routing. |
| Byte ranges, pages, or individual addresses | **Byte ranges**: supports narrow registers without forcing page-sized devices. |
| Overlap priority or rejection | **Reject overlap**: deterministic routing without hidden precedence. |
| Live registration or quiescent changes | **Quiescent changes**: avoids callback-lifetime and stale-routing hazards. |
| Unrestricted VM access or limited context | **Limited context**: supports RAM/device signaling while preserving transaction ownership. |
| New transfer semantics or existing ledgers | **Existing ledgers**: preserves fault priority, partial effects and exact completion. |
| Mandatory callbacks or optional acceleration | **Optional acceleration**: unregistered and declined transfers retain Exit behavior. |

## Consequences

Frequent device traffic can avoid host returns while retaining a common interpreter/compiled contract. Devices gain explicit restrictions on side effects, re-entry and failure.

Longer runs require explicit deadlines and sufficient delivery-log capacity; embedders can no longer rely on every device access returning control.

Implementation stages:

1. Land cheaper Exit/resume independently. Preserve public accounting, completion and log semantics; do not pre-empt handler ownership, registration, ABI or timing decisions.
2. Add registration and internal handoff dispatch using existing completion machinery.
3. Continue interpreted blocks through handled accesses after exactness witnesses pass.
4. Let admitted compiled Machine paths call the same helpers under decision-27 accounting and invalidation rules.

Release additively: registration defaults empty; existing exits/completion APIs remain supported. Use extensible request/result records and a private compiled bridge, without promising a stable Rust binary ABI.

Witness a census across Machine models/regimes, budgets, scalar/string/locked/split transfers, faults, fallback, interrupts, delivery-stack MMIO, SMC, dirty tracking and restore. Compare handler execution against scripted Exit/completion execution: state, RAM, ordered device effects, summed counts, residual exits and delivery records under equivalent drain schedules. Repeat across execution tiers when compilation is admitted.

## Maintainer questions (answered 2026-11-25)

- **Q1 — RAM access in callbacks?** Accepted: checked physical RAM access from stage 2, with normal tracking and explicit paging invalidation.
- **Q2 — Timing before cycle tables exist?** Accepted: exact instruction/iteration positions now; optional cycles remain unavailable until supported.
- **Q3 — Locked-device atomicity across vCPUs/DMA?** Accepted: retaining current limitations; require a separate arbitration contract before strengthening guarantees.
- **Q4 — Panic recovery?** Accepted: terminal poisoning; recovery requires restoring coherent CPU, RAM and device state.
