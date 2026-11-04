---
id: TASK-386
title: >-
  core: a write through a second, writable mapping of code never invalidates its
  translation
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - smc
dependencies:
  - TASK-375
priority: medium
ordinal: 422000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
SMC tracking is keyed by guest VIRTUAL address:
- the `code_page` bitset (memory.rs:517-524);
- the `code_range` watermark (memory.rs:525-528, published to compiled code through `code_range_ptr`, memory.rs:783);
- `mark_code` (memory.rs:687-697);
- `note_write` (memory.rs:811-824).

All of them take the VA of the access. A compiled store tests its VA against the watermark (codegen/mod.rs:2630-2660) and calls the helper, which calls `note_write(VA)`. `handle_smc` drains the dirty pages and invalidates translations overlapping those VAs (vm.rs:551-573). Nothing anywhere relates two VAs that back the same bytes.

THE PATTERN THIS MISSES. Double mapping is the standard W^X layout for a JIT compiler running inside the guest. The runtime maps one shared memory object twice, writes code through an RW view and executes it through an RX view, so no page is ever writable and executable at the same time. Managed-language runtimes and JavaScript engines do this, and so do hardened allocators of executable memory.

Under x86jit the guest's patch lands on the RW page. That page was never marked as code: only the RX view was, when its block was translated. So neither the inline gate nor `note_write` fires. The RX view's translation survives, and the guest runs stale code indefinitely, not one block late. The bytes the lifter would read are already new, because both views share host memory. Only the translation is stale.

The engine cannot discover the alias. The embedder creates it (two host mappings of one object inside the reservation) and is the only party that knows it exists. Writes made outside the store barrier can already be reported with `Memory::note_write`. What is missing is a way to say "this range is an alias of that one".

DESIGN DIRECTION.
- An API on `Vm`/`Memory` to declare `(rw_start, rx_start, len)` pairs and remove them. `Vm::unmap` drops declarations that overlap the unmapped range itself. A runtime may allocate one pair per code chunk, so use a sorted map, not a list scan.
- When `mark_code` tags an RX page that has a declared RW alias, it also tags the alias page in `code_page` and widens the watermark over it. The inline gate does not change. A store to the RW page now reaches the helper. `note_write` finds the page tagged, maps the RW page to its RX page(s), and pushes those to `dirty`, and `handle_smc` invalidates the RX translation. Clearing must be symmetric: invalidate range == clear range, the lesson of task-349.
- Without a declared alias, nothing in the hot path changes.
- MAIN TRADE-OFF. The watermark is a single `[lo, lo+len)` range. An RW view far from its RX view stretches the range over everything in between, and sends ordinary stores there through the helper. task-329 measured that helper path at roughly +10% on `memcpy` when stores fall inside the range. This is why the task depends on TASK-375: its choice of range representation (coarser granule, or per-region watermarks) decides how much this costs, and per-region watermarks would absorb it.
- Concurrency. A pair declared while other vcpus run must be honoured by the next store on any thread. This is the same ordering question task-353 A raises for the watch bit and the count: tag the alias page before the watermark widens, and check the order on aarch64.
- The opposite direction, code written through the RX view by the embedder (a loader), is already covered by `note_write` on that VA.

WHY AN EMBEDDER NEEDS IT. An embedder that runs a full native userspace runs whatever JIT the guest brings. If the guest's runtime chooses dual mapping, which hardened configurations force, every patch it makes is lost on both tiers.

RELATION.
- task-349: unmap untags a shared code page.
- task-352: a guest JIT patching through an alias triggers the same unbounded growth under repeated SMC.
- task-353: watch/mark ordering.
- task-331: with protection-based SMC the RW host mapping would also need protecting, so the alias table is needed there too.

None of these covers aliasing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An API declares and removes RW<->RX alias pairs, documented as the embedder's responsibility; Vm::unmap removes declarations that overlap the unmapped range
- [ ] #2 Test, on both backends: the embedder maps one shared memory object at two guest ranges (mapped twice into the host reservation) and declares the pair. The guest translates and runs code at the RX view, patches it through the RW view with an ordinary compiled store, a rep movs and a locked op, then runs the RX view again and executes the new code (same one-block-late rule as today)
- [ ] #3 The same test with the patch reported by the embedder through note_write on the RW range
- [ ] #4 Without declared aliases, store_write_barrier_stays_within_its_measured_budget is unchanged and memcpy is within noise against bench/baseline.json; with one pair whose views are far apart, the memcpy cost is measured and recorded on this task
- [ ] #5 A pair declared from another thread while a vcpu runs is honoured by that vcpu's next store to the RW view, shown by a test on the aarch64 runner or by an ordering argument written at the code
- [ ] #6 README Known gaps and spec.md §10 state what is covered (declared aliases) and what is not (undeclared ones)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 cargo nextest run (--features unicorn) green, minus fuzz_robustness
- [ ] #2 cargo clippy --all-targets --all-features -- -D warnings clean
- [ ] #3 cargo fmt --check clean (nix-pinned rustfmt)
<!-- DOD:END -->
