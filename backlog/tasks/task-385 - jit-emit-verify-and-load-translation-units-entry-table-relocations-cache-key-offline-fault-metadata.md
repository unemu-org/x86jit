---
id: TASK-385
title: >-
  jit: emit, verify and load translation units: entry table, relocations,
  cache key, offline fault metadata
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - aot
dependencies:
  - TASK-384
  - TASK-364
priority: medium
ordinal: 421000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
doc-17 B2 (aot-plan.md:71-81) keys a per-block cache on a hash of the lifted IR, so "a hit still pays lift" (line 78). It also hard-codes a policy: `JitBackend::with_aot_dir(path)`, one file per key under `~/.cache/x86jit/` (line 79). The deferred.md entry "AOT / persistent translation cache" (lines 169-173) sets the bar. A cache that is wrong once is worse than none, and the key must be "cheap to check and impossible to get wrong".

Translating whole code ranges ahead of time needs more than B2. For example, a shared library translated once and reused by every process that maps it. The engine needs an API that turns a set of guest entry offsets in a module into one translation unit and loads that unit back. A unit carries:
- AN ENTRY TABLE: guest offset to host offset, sorted. It is the only way into the unit.
- A RELOCATION TABLE: helper addresses, libcalls (if any survive TASK-383), and per-process slots (link slots, IBTC slots, return-stack continuation slots). Each entry is typed, so the loader fills it from the current process.
- FAULT-SITE METADATA, computed offline. For each host instruction that can fault, it records the guest offset, where each guest GPR lives (host register or spill slot), and the state of the flags. This is the data TASK-364 needs at run time, stored so that a signal handler can binary-search it without locks or allocation. Today codemap stores only `(host_off, guest_rip)` pairs of `u32`, registered at run time (x86jit-core/src/codemap.rs:33-44).
- A CACHE KEY that is cheap to check. It is a hash of the module's code bytes (or an identity the caller provides) ⊕ the x86jit version ⊕ every codegen input ⊕ an embedder salt. The codegen inputs are:
  - host ISA target and flags, and opt level;
  - `MemConsistency` tier;
  - guest CPU features and x87 precision (vm.rs:406-407);
  - the MXCSR mode the code assumes (TASK-372);
  - icount, superblock caps and the frame-pointer option (TASK-360).
  `codegen_description` (x86jit-cranelift/src/lib.rs:2919-2928) is a start, but it leaves out the consistency tier, guest features, x87 precision and the memory layout, so it cannot serve as the key as it stands.

At load time the engine hashes the guest bytes the unit was built from and compares them with the bytes now mapped. A mismatch rejects the unit, and the JIT takes over. After a load, the unit's guest ranges are marked as code, so a guest write invalidates its entries through the normal SMC path (spec.md §10; SDM Vol. 3A, "Self-Modifying Code" and "Cross-Modifying Code"). A loaded unit meets the same coherence contract as freshly compiled code.

Policy stays with the embedder: what to translate, when, and where to store it. The engine takes bytes and returns bytes. It opens no files and reads no environment, so it is compatible with TASK-360's feature, and doc-17 B2's `with_aot_dir` becomes an embedder concern. doc-17 B3's two-run test (aot-plan.md:83-89) is the definition of done.

Trade-off: hashing every code byte at load time is linear in the module size. A caller-provided identity such as a build id is constant-time but trusts the caller. Offer both, and verify the bytes by default.

Dependencies: TASK-384 makes units position-independent. TASK-364 does fault recovery and consumes the offline metadata.

Existing work: doc-17 B2/B3 (the per-block IR-hash plan) and the deferred.md AOT entry. Decide whether the per-block IR-hash cache becomes a second tier or is dropped, and say why.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An API builds a unit from (guest range, entry offsets, codegen config) into bytes, and loads (bytes, base) into entries in the translation cache, with no file or environment access
- [ ] #2 doc-17 B3's test passes on x86-64 and on an aarch64 host: two runs with a unit store; in the second, nothing that was loaded is compiled again (counter), aot_hits > 0, the output is byte-identical, and interp == loaded unit
- [ ] #3 Every codegen input is part of the key. A test changes each one in turn (host target, opt level, consistency tier, a guest feature bit, x87 precision, icount, frame pointers, version, salt) and asserts that the unit is rejected
- [ ] #4 A unit loaded over modified guest bytes is rejected and the JIT runs instead. A loaded unit that a guest store then modifies is invalidated through SMC
- [ ] #5 Corrupt, truncated and version-mismatched unit bytes are rejected without a panic. A fuzz target covers the loader
- [ ] #6 Fault-site metadata is looked up from a signal handler with no lock or allocation. A fault inside a loaded unit reports the guest RIP and GPRs that the interpreter reports for the same fault
- [ ] #7 Unit load time and first-run time with and without the unit are measured and recorded
- [ ] #8 deferred.md's AOT entry and aot-plan.md B2/B3 describe what was built
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
