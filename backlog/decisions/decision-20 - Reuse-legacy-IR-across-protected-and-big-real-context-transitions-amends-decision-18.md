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
- A TLB flush no longer drops paged proofs (decision-32 Q8, accepted the same day). A guest
  CR3 load, equal values included, a task switch's CR3 load, and a change of A20, CR0.WP,
  CR0.PG or the model still empty the TLB, and they advance the generation, so every proof
  walks its code pages again on its next hit. The proof keys keep naming CR3 through the
  stamp, so each address space keeps its own proofs, and a linear page that another
  directory maps to another frame misses. Only a reset, a mapping change and
  `Vcpu::invalidate_paging` drop every paged proof. The relift diagnostics count guest and
  task CR3 flushes as `TlbAdvance::Cr3Load`, apart from `ExplicitFlush`.
- INVLPG removes only its page's TLB entry, and advances the generation only when a fetch
  had used that entry.
- `Tlb::index` folds the upper page-number bits in shifted left by one, so bit 0 of the
  page number stays unchanged and adjacent pages never share a slot. A block that straddles
  two pages keeps both fetch entries; under the old fold, pages such as 7FFFh and 8000h
  shared a slot, so each hit on such a block walked again.
- The paged proof map is bounded at 4096 blocks or 16 MiB of IR, up from 256 blocks.

## Amendment: reduced identity for paged contexts (2026-11-28, accepted)

Status: **accepted** by the maintainer on 2026-11-28 for bead xj-to9.75. It amends
"Paged contexts, V86 and the Checked class keep the complete stamp" (amendment of
2026-11-27, item 2) for the paged Flat32 class. For every paged class it also changes
how the context table evicts, keys proofs by the frame their code is fetched from
instead of CR3 (replacing "The proof keys keep naming CR3" of the amendment above,
and decision-32 implementation note 13's per-CR3 key), and raises the paged proof
bound.

**Problem.** After the paged proof changes above, a protected-mode operating system's
boot to the desktop still lifted 2.80 M blocks in 266 M retired instructions with
paging on, about 10.5 per 1000. 82 % were First lifts, most of the same flat ring-0
code under a new paged context id with CR3 constant. Two things caused them:

- **Stamp churn.** The OS's 16-bit and V86 transitions reload DS, ES, FS and GS (an
  interrupt from V86 clears all four, SDM Vol 3B §23.3.1.1 step 3). The complete
  stamp names every selector and cache, so the same code ran under many stamps. The
  measured contexts were created by changes to CS, DS, ES, FS, GS, SS, CPL and
  EFLAGS.
- **Fresh ids after eviction.** The stamps outran the 64 retained paged identities.
  An evicted identity lost its proofs, and when the equal stamp came back it got a
  fresh id, since ids were never reused. Its code was lifted again under new keys.
  After the table began evicting one identity at a time (09857cc0), that embedder saw
  1.34 M distinct keys over 64 K addresses, and its context ids grew without bound.

With rules 1 to 3 below, that boot's First lifts fell from 2.31 M to 0.61 M and its
distinct keys to 42 K, but Bound lifts rose from 0.48 M to 1.93 M: proofs no longer
left with every context change, and the working set outgrew the 4096-block bound.
Rules 4 and 5 address that:

- **Per-CR3 keys.** The key named CR3, so a kernel entered from several processes
  needed one proof per block and address space.
- **The bound.** A loop over more blocks than the bound relifts about a quarter of
  them on every pass; the sweep keeps a stable three quarters.

**Rule.**

1. **Paged Flat32 identity.** A paged Flat32 context identity is the reduced identity
   the unpaged path already uses (item 2 above), with CR0.PG added to the CR0 bits it
   names. It names:
   - the model;
   - CR0.PE, CR0.AM and CR0.PG;
   - the CPL;
   - A20, the reset-fetch state and the NEC state;
   - the D/B bits of CS, DS, ES and SS;
   - the class.

   It does not name the selectors, GDTR, IDTR, LDTR, TR, CR3, IOPL, NT, the other CR0
   bits, the FS and GS caches, or the G and A bits of the flat caches. Both the
   dispatcher's identity (`Execution`) and the paged proofs' identity
   (`paging::State`) use it. The paged proof key names CR0 less MP, EM and TS, and
   the frame (rule 4).
2. **Paged Flat32 ids are not interned.** The paged proofs' id of a Flat32 identity is the identity
   itself, packed into one word (`LegacyContextStamp::reduced_word`, bit 63 set). An
   equal state always gives the same key, and these identities never take a slot in
   the context table.
3. **Eviction keeps ids and proofs.** Paged Segmented, Checked and V86 contexts keep
   the complete stamp and the 64-entry table. When a full table evicts its least
   recently used stamp, the stamp's id is kept in a retired map of up to 4096 stamps,
   and its proofs stay in the proof map. When that stamp returns, it takes the same
   id back and finds its proofs. Proofs leave only through the proof map's bound, its
   SMC and translation checks, or when the retired map is full. Then the map forgets
   its least recently used stamp, and only that stamp's proofs leave with it
   (`ReliftCause::ContextEvicted`), one stamp per eviction, so a guest with many
   address spaces sees no cliff. An id still names exactly one stamp until an
   invalidation. The table and the retired map hold at most 4,160 stamps of about
   200 bytes, about 0.9 MB per vCPU, beside the proof map's 32 MiB of IR.
4. **Frame keys.** A paged proof key names the bus address the block's first byte
   translates to, in place of CR3. The resolver translates it on every lookup: a TLB
   entry that permits the fetch, or else the walk the fetch would make (which sets
   A bits and fills the TLB); a walk that faults goes to the lift, which takes the
   fault. A kernel that every directory maps to the same frames has one proof per
   block. A linear address that another directory maps elsewhere is another key, and
   each proof still translates its code pages again after every TLB generation
   advance (CR3 loads included). A page remapped to another frame gets a new proof
   under the new frame's key; the old frame's proof stays, its bytes still guarded
   there, until a write to that frame or the bound drops it, and serves again if the
   mapping returns. Complete stamps (Segmented, Checked, V86) still name CR3, so their
   identities stay per address space. One case costs more than with CR3 keys: a block
   whose first instruction crosses into the next page (the only instruction a paged
   lift lets cross), where every directory maps the first page alike but each maps the
   second to its own frame, has one key for all of them. After each process switch
   its code pages translate again, the second no longer matches, and the block
   relifts; CR3 keys kept one proof per directory. That is correct, only slower.
5. **The bound.** The paged proof map holds 16384 blocks or 32 MiB of IR and
   instruction tables, up from 4096 or 16 MiB: the unpaged map's 8192 blocks with
   room for a kernel's 16-bit and V86 paths, and its byte budget. Measured paged
   blocks of 3 to 10 instructions weigh 1.1 to 1.3 KiB, so the count binds first
   (about 21 MiB at 16384). 1000 blocks of 32 instructions reach the bytes. The
   budget counts IR and instruction tables only (`flat32::weight`), not each proof's
   map entry, page list, `CodeRegistration` or source-index entries, so real use is
   above those figures. The unpaged map's 32 MiB is per vCPU as well: a vCPU that
   fills both reaches about 64 MiB of counted bytes, plus that overhead.

**Why it is as sound as the unpaged reduction.**

- **The lift reads nothing more.** A paged lift reads the same inputs as an unpaged
  one: the code bytes, the IP, the `MachineContext`, the model profile and CR0.AM.
  The only addition is the page walk of its fetch.
- **The fetch is covered without the identity.** The key names the frame the first
  byte translates to under the live CR3 and the CR0 bits that matter, including PG and
  WP. The CPL, which the U/S check of each fetch reads, is in the identity. A proof
  translates its code pages again after any TLB generation advance, so a remap, a
  CR3 load or a downgraded PTE never runs stale code, and a proof found by frame holds
  bytes the source registration guards at that frame.
- **Live reads.** IOPL-sensitive instructions, selector and descriptor-table reads,
  segment loads, task and I/O-permission checks, and CR0.TS for ESC and WAIT run in
  live helpers, as in the unpaged case.
- **FS and GS.** Flat32 IR reaches FS and GS only through checked `Address` ops of
  their live caches.
- **The flat proof.** The class is part of the identity and the key, and is derived
  from the live caches on every reselection. A DS, ES or SS cache that is not flat
  makes the state Segmented, and the Segmented identity names every cache. Within a
  block, an ordinary walk continues past a data-segment load only while the reloaded
  DS or ES is still flat (`reload_continues`).
- **The packed word.** It encodes every field of the reduced identity losslessly
  except the model's semantics id. Every paged proof key names that id, and a model
  change drops every paged proof and link.
- **Kept proofs.** A proof kept for an evicted stamp is valid for that stamp, the
  only stamp its id can name. Its bytes are still guarded by its registration, and
  its pages are revalidated after each TLB generation advance.

**Why paged Segmented keeps the complete stamp.** It could take the reduced identity
by the same argument: the identity would name all six caches, as the unpaged one
does. Complete paged identities, however, carry the derived `MachineContext` and the
dispatcher's identity id (`Execution::reselect_recent`, `paging_selection`,
xj-to9.59). Reducing Segmented was measured to lose those fast paths: the machine32_loops
`farcall`, `segmix` and `farseg` cells on `machine-paged16` and `machine-paged32seg`,
with and without CPL 3, went +8.0 to +13.4 % in host instructions per step. Caching
the context for a reduced Segmented identity would be a follow-up. Item 3 covers its
eviction churn.

**Measured** on `machine32_paged_churn` (release, AArch64 Darwin). The reproduction has
these variants:

- `paged-churn` is the loop as it was.
- `rotate` reloads DS, ES, FS and GS from a 32-entry table at the top of each
  iteration. The table holds flat aliases, byte-granular data segments and null
  selectors.
- `rotate128` uses 128 entries, five of every eight with a non-flat ES. Its
  Segmented phases cycle through about 240 complete stamps.
- `wide:B:K` runs B flat ring-0 blocks, then loads CR3 from K directories that all
  map them to the same frames.

Rules 1 to 3 against 55566a71. Lifts in 20,000 iterations (test-hooks); every lift
was a First lift, and none was a Bound lift, before or after:

| Variant | Lifts before | Lifts after | Distinct keys before | Distinct keys after |
|---|---:|---:|---:|---:|
| `paged-churn` | 42 | 42 | 42 | 42 |
| `rotate` | 544,653 | 277 | ≥ 116,573 (capped) | 277 |
| `rotate128` | 488,912 | 2,147 | ≥ 116,592 (capped) | 2,147 |

The diagnostics cap a site's key set at 4096, so both counts before are lower bounds.
Host instructions per retired step (50,000 iterations, median of 3):

| Variant | Before | After |
|---|---:|---:|
| `paged-churn` | 4,043 | 4,011 |
| `rotate` | 37,272 | 3,799 |
| `rotate128` | 43,489 | 3,915 |

Rules 4 and 5 against rules 1 to 3 alone. Lifts in 200 iterations (test-hooks), and
host instructions per retired step (median of 3):

| Variant | First + Bound before | First + Bound after | Before | After |
|---|---:|---:|---:|---:|
| `wide:1024:8` | 8,273 + 124,133 | 1,035 + 0 | 57,436 | 901 |
| `wide:3000:4` | 12,089 + 438,825 | 3,023 + 0 | 43,721 | 893 |
| `wide:6000:1` | 6,048 + 592,025 | 6,048 + 0 | 29,833 | 892 |
| `wide:12000:1` | 12,091 + 1,794,582 | 12,091 + 0 | 35,257 | 891 |

The `paged-churn`, `rotate` and `rotate128` lifts are unchanged by rules 4 and 5, and
their host instructions per step fall from 4,013 / 3,802 / 3,919 to 3,947 / 3,741 /
3,842: the translation each lookup now makes costs less than it saves. The 12,091
proofs of `wide:12000:1` hold 13.4 MB.

machine32_loops paged cells (49 cells: the interpreted paged paths, the compiled
`machine-paged32-bg` and `machine-paged32-cpl3-bg`, and `cr3switch`; 1 M iterations,
median of 3). Rules 1 to 3 against 55566a71 moved them by −2.7 % (`farcall`, compiled)
to +0.22 % (`farseg` on paged V86) in host instructions per step. Rules 4 and 5 moved
them by −1.3 % (compiled `cr3switch`) to +0.85 % (compiled `farcall` at CPL 3). The
compiled register, memory and callret cells are unchanged at 19.8 / 35.4 / 65.6 (CPL 0)
and 19.8 / 35.2 / 65.4 (CPL 3).

**Consequence.** Two paged Flat32 states that differ only in fields the lift never
reads share one namespace, across address spaces too, and their proofs are shared by
every address space that maps the code to the same frame. A complete stamp evicted
from the table keeps its id, so equal stamps
never fork keys, and the proof bound counts only proofs that some stamp can still
reach. Witnesses:

- `a_paged_identity_names_only_what_its_lift_reads` (core) checks which edits share
  the identity, which share it but rekey the proof (a directory mapping the code to
  another frame, WP), and which split it (AM, A20, CPL, CS.D, class). A directory
  mapping it to the same frame shares both. Segmented and Checked stay complete.
- `a_paged_reduced_identity_reuses_its_proofs_exactly` runs the unpaged reuse cases
  under paging against a fresh vCPU.
- `a_paged_flat32_proof_checks_the_segments_it_reads` covers FS and GS overrides that
  keep their proof through base, limit and null changes. Cases where DS, ES or SS
  becomes non-flat relift, and #GP(0) and #SS(0) are raised on changed limits.
- `paged_context_table_evicts_the_least_recent_identity_and_keeps_its_proofs`
  checks that an evicted stamp returns to its id and proofs without lifting.
- `retired_stamps_leave_one_at_a_time_with_their_proofs` (core) and
  `paged_context_capacity_is_bounded_without_expiring_translations` push more than
  4,160 stamps through: both maps and the proof map stay bounded, every resident proof
  names a held stamp, a retired stamp returns without a lift, and a forgotten one lifts
  again.
- `a_paged_loop_that_reloads_its_data_segments_keeps_its_flat_proofs` runs both
  rotating variants and checks that distinct keys equal lifts.
- `a_kernel_under_many_address_spaces_lifts_each_block_once` runs 512 blocks under 8
  directories and 5000 blocks under one: each block is lifted once and none by the
  bound.
