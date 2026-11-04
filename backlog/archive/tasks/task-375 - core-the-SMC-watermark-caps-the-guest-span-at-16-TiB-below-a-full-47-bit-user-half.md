---
id: TASK-375
title: >-
  core: the SMC watermark caps the guest span at 16 TiB, below a full 47-bit
  user half
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - smc
dependencies: []
priority: medium
ordinal: 411000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`fresh_code_pages` asserts `span <= MAX_TRACKED_SPAN` (memory.rs:1387-1397). `MAX_TRACKED_SPAN = (u32::MAX as u64) << CODE_PAGE_BITS` (memory.rs:337) is 2^32 pages of 4 KiB, about 16 TiB. The span passed in is `guest_base + backing.len()` (memory.rs:613), so the assert fires for any layout whose TOP lies above 16 TiB, however small the mapping is. A refused span panics at `Memory` construction.

The cap does not come from the bitset. That one is packed and lazily committed (memory.rs:317-321, 358-371), with no limit. It comes from the WATERMARK that every compiled store gates on: one `AtomicU64` holding `(lo_page << 32) | len_pages`:
- declared at memory.rs:525-528;
- widened at memory.rs:718-759;
- layout documented at memory.rs:778-782;
- loaded whole and split with `ushr 32` / `ireduce` in generated code (codegen/mod.rs:2630-2657).

memory.rs:325-337 gives the reason it is a single word: two separate words could be read torn, yield a NARROWED range, and silently skip a page.

WHY AN EMBEDDER HITS IT. The lower canonical half of the x86-64 address space is 2^47 bytes, 128 TiB (SDM Vol 1 §3.3.7.1, Canonical Addressing). An embedder that runs a full native userspace identity-maps the guest's whole user half, so that guest pointers are host-representable and the guest's own mmap hints, ASLR and stack placement behave as on hardware. Guest loaders and kernels put the stack and the mmap base near the TOP of that half, far above 16 TiB. Today such an embedder can either shrink the guest address space, and diverge from what the guest's layout code expects, or not construct the `Memory` at all.

DESIGN OPTIONS. Each must keep the hot path at one load and one compare, and must not reintroduce the narrowing race.

(a) A coarser watermark granule (preferred unless measurement disagrees). The watermark is only a pre-filter; `note_write` is precise (memory.rs:803-824). Count lo/len in 2^16-byte units instead of 4 KiB pages: 2^47 / 2^16 = 2^31 units, which fits the existing packing with room for the one-unit low skew. A store of at most 64 bytes can still spill only into the next unit (memory.rs:712-717). Generated code changes one shift constant, and the bitset keeps 4 KiB pages. The cost: stores up to 64 KiB either side of the code extent take the helper. That is negligible next to an extent that already spans everything from the lowest to the highest code page.

(b) A seqlock-protected two-word range. It adds a sequence load and compare to the hot path.

(c) A 128-bit atomic load. This depends on the host. On an x86-64 host that reports AVX (CPUID.01H:ECX.AVX[28]), one aligned 16-byte load is atomic: Intel guarantees it for MOVAPD, MOVAPS, MOVDQA and their VEX.128 and EVEX.128 (k0) forms (SDM Vol 3A §11.1.1, "Guaranteed Atomic Operations"), AMD for any naturally aligned single load (AMD APM Vol 2 §7.3.2, "Access Atomicity"). An x86 host without AVX needs `lock cmpxchg16b`, a locked read-modify-write on every gated store. aarch64 needs FEAT_LSE2 for a single-copy-atomic aligned LDP, and an LDXP/STXP loop otherwise. The JIT would also have to emit the gate's load as one vector load of such a form: it handles 128-bit values as Cranelift `I128` (codegen/vector.rs:8), which lives in a register pair, so check the emitted bytes rather than assume. The hot path then costs one vector load plus lane extracts, and correctness rests on a host feature (AVX or LSE2) that (a) does not need.

(d) Per-region watermarks. More state and more compares. TASK-386 might want them anyway.

The task picks one and records why, against measurement.

WHAT ALSO BREAKS AT 2^47, and the task must handle:
- Three bitsets are sized over the span: `code_page`, and `WatchPages::watch` and `::dirty` (memory.rs:381-406). Each is 2^35 pages / 64 × 8 B = 4 GiB of address space, 12 GiB in total. `zeroed_words` commits them lazily, which is fine with overcommit and refused under strict overcommit. Indexing the tables from `guest_base`'s page instead of page 0 would make "high base, small backing" cheap. The generated watch probe indexes by raw page (codegen/mod.rs:2689-2703), so it would need the same offset in its cold block. Decide and record.
- `clear_all_code_pages` (memory.rs:795-801) stores 0 into EVERY word of `code_page`. At 2^47 that writes 4 GiB, which commits it and takes seconds. It runs on every Trap-region map (vm.rs:466-470). Bound the walk to the watermark range, or skip words that are already zero.
- memory.rs:335 says `Reserved` "is documented up to 1 TiB". Update it together with the constant.

TEST SHAPE that needs no 128 TiB host mapping (an x86 host with 47-bit user addressing could not provide one): `guest_base = 2^47 - 2 MiB` and a 2 MiB host mapping through `Memory::from_host_ram` (memory.rs:572-598). The span top is then 2^47 and the code page index is above 2^32. Today the constructor panics.

RELATION.
- task-323 (done) removed the 4 GiB CODE_WINDOW, and this is the next limit.
- task-329 (done) introduced the watermark and the budget test `store_write_barrier_stays_within_its_measured_budget` (codegen/mod.rs:4690).
- task-331 (host page protection) would replace the inline gate; nothing here gets in its way.
- task-351 (the codemap truncates a guest RIP above 4 GiB) is a separate high-address defect, in the fault path.
- TASK-386 (aliases) builds on whatever range representation this task chooses.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Memory whose span top is 2^47 constructs with both backends; MAX_TRACKED_SPAN is removed or raised to at least 2^47, with the reason in its doc comment
- [ ] #2 Test: code at a page index above 2^32 (guest_base = 2^47 - 2 MiB) is patched once by a JIT-compiled guest store and once by an embedder write; the interpreter and the JIT both execute the new bytes, under today's one-block-late rule
- [ ] #3 The chosen representation is argued against the torn-read/narrowed-range race in its doc comment, and a concurrent mark_code-versus-store test covers it on the aarch64 runner
- [ ] #4 store_write_barrier_stays_within_its_measured_budget passes with its current bound; memcpy and the other bench workloads are within noise against bench/baseline.json
- [ ] #5 Mapping a Trap region on a 2^47-span Memory does not commit the code-page table; RSS before and after is measured and recorded on this task
- [ ] #6 RSS of a fresh 2^47-span Memory is measured and recorded; the decision on indexing the bitsets from guest_base is recorded with its cost
- [ ] #7 memory.rs:325-337 and :335, and any README or spec text that states a span limit, describe the new limit
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
