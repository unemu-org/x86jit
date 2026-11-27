---
id: decision-20
title: >-
  Reuse legacy IR across protected and big-real context transitions (amends
  decision-18)
date: '2026-11-11 15:21'
status: accepted
---
## Context

The accepted xj-srh.4 acceptance addendum requires frequent protected/real USE16
transitions to retain reusable IR in the other context. Decision-18 §3 originally
keys blocks by a monotonically changing epoch and discards the entire namespace
at every context boundary. This amendment replaces that invalidation policy;
its architectural guard dimensions and interpreter-only execution boundary remain.

The same addendum requires big-real segment reloads to replace the selector/base
while preserving the cached descriptor limit and attributes, including a 4-GiB
data limit. Application Real16 remains the fixed application personality.

## Decision

Keep the monotone context epoch for observing state changes and instruction
boundaries. Intern each complete `LegacyContextStamp` into a vCPU-local stable
context identity and use that identity in `LegacyBlockKey` in place of the epoch.
The stamp retains every field required by decision-18 §3, excluding CR2 and live
IF/shadow/completion state. Returning to an identical stamp reuses its existing
namespace while retained; it does not clear either regime's blocks or relift them
wholesale.

Retain at most **16 contexts per vCPU** (`MAX_RETAINED_CONTEXTS`). The measured
firmware-style real/protected loop uses four contexts; sixteen allows a handful
of additional code/stack configurations while bounding retention under context
churn. Index complete stamps in a hash map for expected O(1) lookup. On a hit,
refresh recency using the context epoch. At capacity, evict the least recently
used context and drop all its instruction blocks before reusing its context ID.
Victim selection scans at most sixteen entries. Returning to an evicted stamp
relifts its instructions; returning to a retained hot stamp preserves its IR.

Revalidate fetched bytes, mappings and fetch policy on each new instruction;
perform segment access checks using current CPU state. Reuse does not cache a
permission verdict. An outstanding MMIO retry retains one instruction IR `Arc`
independently of evictable translations and completes that consumed instruction
when its full context stamp, IP and MXCSR mode match, even after context eviction.
The continuation is released on completion, abandonment or execution-state reset;
the sixteen-context eviction bound is unchanged. Explicit CR3 writes, including
equal-value writes, and epoch rollover still discard derived namespaces. Context
identities are private runtime data, never checkpoint state or a cross-vCPU
sharing contract. Paging remains refused.

Retain the legacy runtime segment-access boundary ahead of the fixed Application
Real16 lowering. After PE clearing, a real reload replaces the base with
`selector << 4` and preserves the cached inclusive limit and descriptor attributes.
The engine validity/reset markers retain their existing reload behavior.

**Authority and narrow reading.** Intel SDM **325462-092US**, Vol 3A §§2.2,
12.9.1–12.9.2 and Vol 3B §23.1.1 describe real-address translation and mode
switching. The **80386 PRM 230985-001**, §§14.1 and 14.5, pp. 14-1 and 14-4
(verified page images), describe ordinary 64-KiB real segments and the recommended
return sequence. Neither explicitly specifies what a later real-mode reload does
to a nonstandard inherited limit/attributes. Base-only reload with those values
preserved is the accepted engine contract; **AMD64 APM Vol 2 §4.1** explicitly
documents retaining both. This is not a claim of complete 80386 model fidelity.

## Consequences

A warmed real/protected USE16 round trip uses four complete context identities
and six instruction blocks. Isolation witnesses require no further lifts across
repeated transitions, unchanged IR ownership, new identities for changed hidden
caches, and byte revalidation for self-modifying code in both regimes. Existing
model, decode-IP, CS mapping, stack-size and permission isolation remains.

Distinct architectural stamps retain interpreter IR in at most sixteen context
namespaces; eviction removes the evicted namespace's blocks. This bounds the
contribution from context churn, not the number of instruction addresses within
one retained context. There is no cross-vCPU interner or new application cache
cost. Churn witnesses populate two blocks per context and assert both bounds;
a hot-context witness asserts IR ownership survives repeated cold insertions.
A rollover witness populates multiple namespaces and verifies removal before
context-ID reuse. Future paging must continue revalidating translation bindings
rather than treating a reused context identity as proof that a mapping is unchanged.

## Measured transition cost

Release interpreter-only `pe_transition_throughput`, rustc 1.98.1 on AArch64
Darwin, default core features, a 64-KiB Flat RAM map, paging off, and USE16 code
with a 16-bit stack. Each six-instruction round trip sets PE, far-jumps into
protected code, clears PE, and far-jumps to real code. After 100 warm-up round
trips, each of three samples executes 100,000 round trips: 200,000 PE changes and
600,000 instructions. The fixture asserts instruction counts, final context,
and zero backend compilation. Each measurement command is limited to 180 seconds.

```sh
cargo --offline nextest run --release -p x86jit-core --lib \
  -E 'test(pe_transition_throughput)' --run-ignored all \
  --test-threads 2 --success-output final
```

| Policy | Median PE transitions/s | Relifts per sample | Blocks retained |
|---|---:|---:|---:|
| Original epoch flushing | 1,281,834 | 600,000 | 0 |
| Full-stamp hash interner trial | 1,205,944 | 0 | 6 |
| Accepted compact stamp table | 1,851,020 | 0 | 6 |

These measurements precede the bounded-retention revision above. The compact
table used full stamp equality and a linear search; the current implementation
uses a full-stamp hash index and a sixteen-context LRU cap. The table therefore
does not establish throughput for the current policy. This microbenchmark
measures warmed engine transitions, not a whole firmware startup or physical CPU
conformance. Samples are sequential before/after measurements on the same host,
not a statistically isolated throughput guarantee. Raw samples, including the
rejected lookup trial, are in
[decision-20-transition-measurements.csv](decision-20-transition-measurements.csv).

## Amendment (context retention and the unpaged identity, 2026-11-27, accepted by the maintainer)

Decider: maintainer (2026-11-27), accepting both items. Item 1 records a change
that landed earlier without an amendment; item 2 lands with xj-to9.56.

1. **Already on main (241c4cba, xj-to9.53).** It raised `MAX_RETAINED_CONTEXTS`
   from 16 to 64, with least-recently-used eviction. This record was not amended
   at the time, so this text closes that gap. A PC-class embedder's
   extended-memory runtime walks through more than 16 distinct stamps on each
   interrupt reflection. The code, data and stack selectors vary, there is a
   real-mode round trip, and IDTR switches to and from the real-mode IVT. With
   16 retained contexts, every reflection minted new context ids. The runtime's
   own blocks were then relifted under new keys, and their proofs flooded the
   proof map. The interrupted program's hot proofs were evicted under unchanged
   keys. With 64 contexts, that embedder's in-game renderer went from 11.8-13.4
   to 20.9-22.1 M steps/s, 126-134 % of its guest clock.

2. **Unpaged identity (xj-to9.56).**
   - **Identity.** An unpaged Flat32 or Segmented context identity names only
     what its lift reads: the model, CR0.PE and CR0.AM, the CPL, A20, the
     reset-fetch and NEC state, and, for Segmented, the descriptor caches. It
     no longer names the selectors, GDTR, IDTR, LDTR, TR, CR3, IOPL, NT, or
     the G and A bits of the flat caches.
   - **Why the reduced identity is exact.** The lifter is a pure function of the
     code bytes, the IP, the MachineContext, the profile and the alignment mask;
     it receives no CpuState. IOPL-sensitive instructions lower to live
     `Privileged` ops. Selector, descriptor-table and task-register reads, and
     segment loads, run in live helpers. Flat32 admission requires a cached
     limit of exactly 0xFFFFFFFF with base 0, so a G=0 cache cannot be admitted
     as flat. Paged contexts, V86 and the Checked class keep the complete stamp,
     as do the retry, REP and validation paths.
   - **Retention.** 256 contexts are retained. Context ids are never reused
     before a translation flush, so evicting a context no longer walks every
     proof.
   - **Proof map.** The Flat32 and Segmented proof map is bounded by size, not
     only by count.
   - **Measured.** On a local reproduction of the runtime's reflection cycle,
     host instructions per step fall from 2345 to 649, and lifts per million
     steps fall from 31,000-60,000 to 0.
   - **Consequence.** Two unpaged stamps that differ only in fields the lift
     never reads now share one namespace. Every such field stays exact because
     it is read live at execution. A witness lifts IOPL-, selector-,
     descriptor-table- and task-sensitive instructions under one state and runs
     them under another, against the Checked class.

## Amendment: paged proofs survive translation-preserving TLB changes (2026-11-27, accepted)

Accepted by the maintainer on 2026-11-27 for bead xj-to9.70. A protected-mode operating
system ran its paged phases with about one proof lift per five retired instructions: every
TLB generation advance (an INVLPG, the flush it caused to repeat, a fill replacing a fetched
entry) dropped every paged proof, and CR0.TS toggles split the paged stamp. The paged
identity changes as follows; everything else in this record stands.

- The paged proof key and the paged context stamp no longer contain the TLB generation, and
  their CR0 omits MP, EM and TS. ESC and WAIT still test those bits when they execute (SDM
  Vol 3A §2.5 CR0), so no lifted op depends on them.
- A paged proof records the linear and bus address of the first fetched byte of each code
  page it covers. On a hit after a generation advance it translates them again as a fetch
  would; the same bus address keeps the proof, another frame or a fault relifts it
  (`ReliftCause::Translation`). The byte proof against the source is unchanged.
- Only a flush drops every paged proof: a CR3 load (decision-32 Q8 stands), a change of the
  paging bits of CR0, A20, the model, or an explicit `Vcpu::invalidate_paging`. INVLPG
  removes only its page's TLB entry, and advances the generation only when a fetch had used
  that entry.
- The paged proof map is bounded at 4096 blocks or 16 MiB of IR, up from 256 blocks.
