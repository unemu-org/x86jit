---
id: TASK-389
title: >-
  elf: static code discovery for ahead-of-time translation (unwind ranges plus
  recursive descent)
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - aot
  - elf
dependencies: []
priority: low
ordinal: 425000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ahead-of-time translation (TASK-385) needs the set of block entries before anything runs. Nothing in the repository finds code today:
- x86jit-elf maps `PT_LOAD` segments of static images and returns the entry point. Its scope note says dynamic linking and relocations "remain the embedder's job" (x86jit-elf/src/lib.rs:1-30).
- The core can lift from any address (`lift_block`, x86jit-core/src/lift/mod.rs:237; `lift_one`, lift/mod.rs:316) but has no notion of which addresses are entries.

On x86 a linear sweep is unreliable. Instructions are 1 to 15 bytes long (Intel SDM Vol. 2A, Chapter 2 "Instruction Format"). Compilers put jump tables, constants and padding inside executable sections. A decode that starts one byte off produces a plausible but wrong instruction stream.

Provide a generic discovery pass, in x86jit-elf or a new crate. It cannot live in the core: x86jit-core's dependency set is fixed at `{iced-x86}` (x86jit-tests/tests/boundary.rs), and goblin must stay out of it.
- SEEDS:
  - `.eh_frame` FDEs, each giving a function's pc_begin and pc_range (System V AMD64 psABI, the unwinding/`.eh_frame` section; LSB Core, "Exception Frames");
  - function symbols in `.symtab`/`.dynsym`, with their sizes;
  - the ELF entry point, and the `DT_INIT`/`DT_FINI` functions with `init_array`/`fini_array` entries.
- RECURSIVE DESCENT from the seeds, using the core's decoder and lifter, along direct jumps, conditional branches and direct calls. Code after a call counts as code at lower confidence, because the callee may not return.
- OUTPUT:
  - entry addresses relative to the module, matching TASK-384, with a confidence for each range: described by unwind data, described by a symbol, or reached only by descent;
  - the executable bytes that were never reached, so the embedder knows how much it covered.

Safety comes from how units are entered, not from discovery being right. Bytes decoded as code are never executed on that basis: a unit is entered only through its entry table (TASK-385), and only when the guest actually transfers to that address. A wrong entry costs space. A missed one costs a JIT compile at run time.

Platform-specific heuristics stay out of this crate: a toolchain's jump-table format, a managed runtime's constant pools or trampolines. Provide a callback through which the embedder adds seeds and marks ranges as data.

Parsing `.eh_frame`: only FDE pc ranges are needed, not the CFI programs. That means either a small parser that handles the CIE's pointer encoding, or `gimli`, which would be a new dependency for this crate and must keep `cargo deny` green. Decide.

Existing work: none.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A discovery API returns entries with a confidence each, plus the unreached executable ranges, for ET_EXEC and ET_DYN images
- [ ] #2 Recall: on x86jit-tests/programs/pthreads.elf, and on one ET_DYN image built for the test with its source kept (per PROVENANCE.md), every block the interpreter executes during a run is in the discovered set, or listed as unreached with the reason recorded. The recall figure is recorded on this task
- [ ] #3 Precision: discovered entries that land inside another discovered instruction are counted. A constructed test with a jump table inside `.text` shows that descent does not decode the table as code
- [ ] #4 The embedder callback can add seeds and mark data ranges, and a test uses both
- [ ] #5 x86jit-core's dependency set is unchanged (boundary.rs green), and `cargo deny` is green
- [ ] #6 Discovery time on the largest test image is measured and recorded
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
