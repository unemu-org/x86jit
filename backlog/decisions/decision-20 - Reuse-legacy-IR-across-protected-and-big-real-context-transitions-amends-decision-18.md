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
