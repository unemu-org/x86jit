---
id: TASK-358
title: >-
  core: MXCSR is not architectural state: ldmxcsr is a no-op, stmxcsr and
  fxsave store 0x1F80
status: To Do
assignee: []
created_date: '2026-11-04 14:00'
updated_date: '2026-11-04 14:00'
labels:
  - fp-fidelity
  - mxcsr
dependencies: []
priority: high
ordinal: 394000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
READ at 96fd5c6: the engine has no MXCSR. `CpuState` (state.rs:313-422) carries the x87 control word, status word and tag state, but no field for the SSE control/status register. Every instruction that touches MXCSR works around the missing field:

- `ldmxcsr` / `vldmxcsr` lift to nothing (lift/mod.rs:1126). The comment above them (1114-1115) says MXCSR is not modelled.
- `stmxcsr` / `vstmxcsr` lift to a 4-byte store of the immediate `0x1F80` (lift/mod.rs:1116-1125).
- `fxsave` writes `0x1f80` at offset 24 and `0xffff` as MXCSR_MASK at offset 28 (x87.rs:387-388). `fxrstor` never reads offset 24 (x87.rs:354-375). The doc comment says so (x87.rs:340-342), and so does the IR doc (ir.rs:308-310).
- The x87 environment doc calls the SSE half of a `fenv_t` "a wider fiction" (x87.rs:552-554).
- The test harness has the same blind spot. The engine snapshot reports `mxcsr: MXCSR_RESET` whatever the guest did (x86jit-tests/src/oracle.rs:254-259). The engine is never seeded with `cpu_init.mxcsr`; oracle.rs:203 seeds only the x87 control word. The native leg is seeded (native.rs:593), and the Unicorn leg writes only FPCW (unicorn.rs:243).

So a guest that saves, changes and restores the SSE mode reads back a value it never wrote. `fesetround(FE_TOWARDZERO)` followed by `fegetround()` reports round-to-nearest, because `stmxcsr` returns RC=00. A `fegetenv`/`fesetenv` or `feholdexcept`/`feupdateenv` pair loses the SSE half. A runtime that sets FTZ and DAZ at thread start and later verifies them sees them cleared.

ARCHITECTURE.
- MXCSR is per-logical-processor state. Bits 15:6 are control (FZ 15, RC 14:13, the six masks 12:7, DAZ 6) and bits 5:0 are the sticky exception flags (SDM Vol 1 §10.2.3, Figure 10-3). The reset value is 1F80H (SDM Vol 1 §11.6.4, Table 11-2, already cited at x86jit-tests/src/vector.rs:108-110).
- LDMXCSR raises #GP(0) on an attempt to set reserved bits (SDM Vol 2A, LDMXCSR). Reserved means clear in MXCSR_MASK (SDM Vol 1 §11.6.6). FXRSTOR raises #GP(0) when the image's MXCSR sets such a bit (SDM Vol 2A, FXRSTOR). The engine already reports MXCSR_MASK = 0xFFFF in its fxsave image (x87.rs:388), so the reserved bits are 31:16, and that has to stay consistent.
- STMXCSR stores the whole register, flags included (SDM Vol 2B, STMXCSR).

WHY AN EMBEDDER NEEDS IT. An embedder that runs a full native userspace, with real POSIX signals and several guest threads, has to:
- save the interrupted thread's MXCSR into the signal frame it builds and restore it on return;
- start each new thread with the value its ABI prescribes, or the value of the thread that created it;
- carry MXCSR across any guest-thread switch it does on a vcpu.

None of this is possible without a per-vcpu value and an accessor. Fixing the guest-visible round trip alone would not be enough.

DESIGN DIRECTION.
- Append `mxcsr: u32` at the END of `CpuState`, like `fpu_empty`/`fpu_sw` (state.rs:385-398). Every existing `#[repr(C)]` offset and `jit_abi::CpuOffsets` stays unchanged. Set it to 0x1F80 in the field-by-field `Default` (state.rs:449ff). That `Default` is written out by hand so that a forgotten field is a compile error (state.rs:436-448).
- Route `ldmxcsr`/`stmxcsr` through a helper, as `fxsave` already is (x86jit-cranelift/src/lib.rs:197). The alternative is an append-only `CpuOffsets` entry for an inline lowering. These instructions are rare (fenv calls, thread start), so the helper is the simpler choice. The interpreter shares the helper body.
- `ldmxcsr`/`vldmxcsr` check the loaded value against the mask. A reserved bit is a fault: `Exit::Exception { vector: 13 }` with RIP on the instruction and nothing committed. That is the fault convention `IrOp::Trap` uses with `advance = 0` (ir.rs:2476-2488), but here it depends on the loaded value, so it cannot be an unconditional `Trap`.
- `fxsave` writes the live value at offset 24. `fxrstor` checks offset 24 BEFORE it commits anything else from the image, because today it commits FCW/SW/tags/ST/XMM in one pass (x87.rs:354-375).
- Add `Vcpu::mxcsr()` / `set_mxcsr()` beside `fpu_cw()` / `set_fpu_cw()` (vm.rs:884-891). Also give the embedder a way to choose the value a new vcpu starts with.
- STORAGE ONLY. Arithmetic keeps ignoring the mode until TASK-372. The sticky flags hold whatever the guest wrote and are never set by execution. This is the "carried verbatim" stance `fpu_sw` had before task-328 (state.rs:388-394).

TRADE-OFF. Until TASK-372 lands, a guest that sets RC, FTZ or DAZ reads back the truth while its arithmetic still runs in the default mode. That is more honest than today, where the guest cannot even read back what it wrote, but README and deferred.md must state the interim plainly ("stored, not yet applied"). Otherwise the next reader will assume the mode works.

RELATION. This reopens deferred.md "MXCSR and vector FP flag semantics" (deferred.md:175-179), into which task-82 was closed. task-325 (done) compares the control half against the native oracle (x86jit-tests/src/vector.rs:118-124, compare.rs:210-211). After this task the engine side of that comparison reports a real value instead of a constant. TASK-372 makes the stored mode govern arithmetic. `xsave`/`xrstor` are not lifted today (no match in lift/). When they are, their legacy region carries MXCSR at the same offset 24.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `CpuState` carries MXCSR with reset value 0x1F80, appended so that every existing `CpuOffsets` offset and its offset-assertion tests are unchanged
- [ ] #2 A test in x86jit-tests, one body run on BOTH backends: `ldmxcsr` of a value with RC=11, FTZ, DAZ and two sticky flags set, then `stmxcsr`, reads back the same 32 bits; the same for `vldmxcsr`/`vstmxcsr`. It fails at 96fd5c6 (reads 0x1F80)
- [ ] #3 `fxsave` stores the live MXCSR at offset 24 and `fxrstor` loads it: the sequence `fxsave` -> `ldmxcsr` (another value) -> `fxrstor` -> `stmxcsr` returns the saved value, on both backends
- [ ] #4 `ldmxcsr` and `fxrstor` with any of bits 31:16 set raise `Exit::Exception { vector: 13 }` with the saved RIP on the instruction and commit nothing; for `fxrstor`, a whole-state snapshot before and after shows no x87/XMM/MXCSR change. Both backends
- [ ] #5 `Vcpu::mxcsr()` / `set_mxcsr()` exist next to the x87 control-word accessors, the embedder can choose the value a new vcpu starts with, and both are documented
- [ ] #6 The oracles seed and read the engine's MXCSR: oracle.rs:203 and 254-259 use the accessors, and the Unicorn leg writes and reads its MXCSR register. A differential vector with a non-default control half agrees with the native oracle on x86-64 Linux and with Unicorn on every host
- [ ] #7 The interim is documented where it is claimed today: README.md:117, deferred.md:175-179, x87.rs:340-342 and 552-554, ir.rs:308-310 and lift/mod.rs:1114-1115 all say that MXCSR is stored and round-trips but does not yet govern arithmetic (TASK-372)
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
