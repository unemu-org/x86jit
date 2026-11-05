---
id: doc-5
title: 'Deferred'
type: guide
created_date: '2026-07-06 11:25'
---

# Deferred

Things **deliberately not implemented yet**. If something seems missing and is listed here, don't add it unprompted — there's a milestone reason (spec.md §12, §14). Each entry: what, why deferred, when to revisit.

## Format

```markdown
### <Feature / behavior>

- **Why deferred:** <one paragraph>
- **Revisit when:** <trigger condition / milestone>
- **Tracked in:** <issue #, if any>
```

## Entries

### fork / execve from a threaded process (go-caddy P2)

- **Now `unemulinux`'s** — the process model left with the Linux userland.
- **Why deferred:** once a process has spawned a thread (`clone(CLONE_VM)`), neither Linux fork semantics (duplicate only the calling thread) nor execve (kill all siblings, replace the image) is modeled. The threaded driver (`x86jit-linux/src/thread.rs`) handles them without ever panicking the host: **fork/vfork/clone-without-CLONE_VM return the guest a real `-EAGAIN`** (fork's resource-exhaustion errno, which every runtime handles), and **execve/wait4/blocking-pipe-read return a fatal typed `ProcError`** naming the op (faking an execve errno would silently corrupt a run). A single-threaded process is unaffected — it still forks/execs through the deferred scheduler (`x86jit-linux/src/proc.rs`).
- **Revisit when:** a real threaded guest needs `posix_spawn`/`system()`/`exec` — then model fork-of-calling-thread and execve-kills-siblings properly.
- **Tracked in:** go-caddy P2.8 (task-21.9 (unemulinux)).

### JIT backend (Cranelift codegen)

- **Why deferred:** interpreter must exist and be correct first — it is the oracle for the JIT (§13). Building both at once removes the reference to validate against.
- **Revisit when:** M4, after the interpreter runs the M2 corpus. Build incrementally: empty "return Continue with new RIP" block first, then `IrOp` by `IrOp` (§8.2.3).
- **Tracked in:** —

### Lazy flags (Variant B)

- **Why deferred:** materialized flags (Variant A, §3.2) are simpler and correct. Lazy flags are a performance optimization that complicates the IR.
- **Revisit when:** profiling attributes a real workload's cost to flag materialization. It already has: task-216 measured ~35 host instructions per block going to flags at block exit, and the design (`cc_op`/`cc_src`/`cc_dst`, ~35 → ~6) is worked out. What is missing is a workload where it moves the number that matters — four earlier micro-optimisations measured well locally and moved the real guest by zero.
- **Tracked in:** was task-220, closed into this entry — the decision not to do it yet is the content, and a task restated it.

### SoftMmu memory model

- **Why deferred:** `Flat` (one contiguous host buffer) is fastest and enough while the guest space is dense (§4.1).
- **Revisit when:** the guest uses sparse / high addresses (e.g. near the top of the 64-bit space) that `Flat` can't back.
- **Tracked in:** —

### SMC (self-modifying-code) invalidation — **SHIPPED, entry kept for the part that is still deferred**

M6 delivered per-page code tracking and invalidation on write, and task-329 extended it
to stores executed by *compiled* code (which had reached no SMC hook at all — the stale
translation ran). What is still deferred:

- **Same-block SMC.** A block writing into the page it is itself executing runs to the
  end of that block on the old bytes. Making it faithful means bounding blocks at
  write-detected boundaries. §10 records it; QEMU makes the same deviation.
- **A zero-cost write barrier.** The current one is an inline range test on every
  compiled store, measured at ~+9 hot instructions per store and ~10% on the store-heavy
  bench. Host page protection (what Box64/FEX/QEMU use) would cost nothing in the steady
  state but needs a fault hook the guest-agnostic core cannot install itself.

- **Revisit when:** a guest whose hot loop is store-bound, or one that JITs (a JVM, V8,
  .NET), makes either cost real.
- **Tracked in:** `TASK-331`. (`TASK-323` closed the multi-vcpu invalidation races.)

### Memory protection (`Prot`) is advisory

- **What:** a guest store into a region mapped `R` or `RX` succeeds and changes the
  bytes, on **both** backends. Measured, not assumed. The engine models no permission
  fault: `MemTrap` distinguishes only mapped/unmapped/MMIO and `Exit` has no
  protection-fault variant. `Prot` tags the region and bounds-checks; nothing consults it
  on access.
- **Why deferred:** not because it is hard in the interpreter — it would be a `region_at`
  consultation — but because the JIT's inlined accesses bound against `MemCtx.size` alone
  and have no region map by design (decision-3). Enforcing there means either a lookup on
  every load and store, or reusing decision-5's guard pages, which requires widening the
  embedder's `protect` hook from `accessible: bool` to a permission. That is a contract
  change across repositories, and it still would not cover a `Vec`-backed `Flat` span.
  Enforcing on the interpreter alone would manufacture exactly the backend divergence the
  project spends its effort removing.
- **Revisit when:** an embedder needs `.text` write-protection as a real guarantee, or a
  guest legitimately depends on `#PF` for a permission violation (a JIT'd guest flipping
  W^X pages is the realistic trigger). Then widen `ProtectFn` and deliver
  `Exit::Exception { vector: 14 }` — the variant already exists, so no new `Exit` is
  needed.
- **Tracked in:** `TASK-330`.

### Leaner ARM memory barriers

- **Implemented:** shared-memory vcpus, translation-cache synchronization, explicit fences and JIT inline-access tiers (§8.2.3). AcqRel currently uses DMB ISH before stores and after loads; FullTso uses DMB ISH after each load and store, with no leading store fence.
- **Why deferred:** STLR/LDAPR lowering, weaker load/store-only barriers and adjacent-fence merging need separate measurements and correctness evidence.
- **Limits:** the interpreter leaves ordinary accesses Relaxed in every tier (xj-0w3.6), helper accesses bypass tier barriers (xj-0w3.7), and aarch64 LL/SC locked-op ordering remains incomplete. These are correctness gaps, not deliberate architectural deviations.

### SIMD (SSE/AVX)

- **Why deferred:** big, self-contained chapter. XMM/YMM state and vector-instruction lift (§3.1, §12 M8+). Real games need it, but nothing on the critical path does.
- **Revisit when:** M8+.
- **Tracked in:** —

### Block chaining / superblocks / traces

- **Why deferred:** performance optimization that stitches blocks without returning to the dispatcher (§12 M5).
- **Revisit when:** M5, after correctness is locked and profiling justifies it.
- **Tracked in:** —

### Background tier-up — deliberate exclusions (bg-tier, doc-22)

Background tier-up shipped (a single compiler thread per `JitBackend`; opt-in via
`Vm::set_tier_up_background`). Three parts were left out on purpose:

- **Compiler-thread pool (one worker only).** The worker holds the same `Mutex<Jit>`
  the foreground `materialize` uses, so N workers can't compile in parallel until the
  `JITModule` is retired. **Revisit when:** FD-AOT B0.2 removes the shared module (§9.1).
  **Tracked in:** doc-22 D3.
- **Background *region* (superblock) tier-up.** Only single blocks tier up in the
  background today; hotness-gated region formation off the vcpu is a separate rung.
  **Revisit when:** BGT-6. **Tracked in:** task-100.
- **Per-span epoch (global epoch today).** A single invalidation epoch means an
  unrelated SMC/map can reject an in-flight compile that then self-heals by
  resubmitting — correct, but wasteful under heavy code-page churn. A per-span epoch
  would scope rejections. **Revisit when:** if the `tier_bg_rejected` counter shows it
  matters. **Tracked in:** doc-22 (risks).

### Threaded virtual clock (VCLK, doc-6 (unemulinux) / decision-2 (unemulinux)) — deliberate exclusions

The mt-mode virtual monotonic clock shipped (rate-controlled value, real host
blocking, idle-only wait credits). Four parts were left out on purpose (doc-6 (unemulinux) M6):

- **No host-time governor ("max virtual speedup" rate-limit).** Nothing asserts
  wall-time pacing, so virtual time is not capped to real elapsed. **Revisit when:**
  a guest legitimately needs wall-clock-correlated time (rate limiters, TLS validity)
  — that reopens the governor alternative (decision-2 (unemulinux) trigger).
- **One clock domain.** `CLOCK_REALTIME` == `CLOCK_MONOTONIC + CLOCK_BASE_SEC`; no
  per-clock drift, no `CLOCK_THREAD_CPUTIME_ID`. **Revisit when:** a guest needs a
  distinct clock's semantics.
- **Blocking host-fd I/O consumes no virtual time.** A real host wait on a real fd
  (a blocking socket `read`/accept) is invisible to the guest clock — bounded risk
  R3. **Revisit when:** a real workload misbehaves timing a host-fd operation (then
  credit blocking fd I/O too).
- **`SYS_POLL` stays instant-ready and time-free.** Go blocks in epoll/futex, not
  poll. **Revisit when:** a real guest needs poll timeouts.

The clock-*discriminator* gate **is built** (task-45 (unemulinux)): a deterministic threaded test,
`thread::tests::busy_periodic_timer_discriminates_cas_from_fetch_max`, replays one
long-span busy-process interleaving (a periodic timer whose every wait is overlapped by
a worker read, ordered by a per-cycle barrier) under BOTH credit rules and shows they
diverge — the idle-CAS gate injects zero wall-coupled time (a virtual deadline holds),
`fetch_max` re-couples the clock and blows it. Deterministic → non-flaky and
load-invariant (verified 30× under CPU load), which the guest-level attempts could not
be. Companion tripwire: `read_metered_deadline_spin_terminates` (the doc-6 (unemulinux) 30 ms
micro-repro).

Still **not built**: the same discrimination through a *real Go guest* under the eager
JIT. A `ReadHeaderTimeout` `go_http` variant was prototyped and dropped — its accept→read
window is too short in guest-progress terms to distinguish the two credit rules (both
pass ≥500 ms), and the deadline-free eager leg passes under either rule and even under a
host-anchored clock (the eager empty-response was a fixture bug, not the clock). A real
long-span-deadline Go workload is also unfit for the AC#2 non-flaky/no-load-sensitivity
bar — the eager JIT leg alone runs minutes. **Revisit when:** a long-span-deadline
workload enters the corpus for another reason and can double as an end-to-end check.

### IRQ injection and icount virtual time (Machine Exit surface)

- **Why deferred:** an `Exit` for asynchronous interrupt injection, and an instruction-count budget that returns control on a virtual-time deadline, are what a *machine* emulator needs. This is a CPU engine; every consumer so far drives it with real syscall traps and wall time, and neither has been asked for.
- **Revisit when:** an embedder emulates a device that raises interrupts, or needs deterministic virtual time across a run (a full-system machine model would).
- **Tracked in:** was task-144, closed into this entry.

### AOT / persistent translation cache

- **Why deferred:** deliberately cut out of the fast-dispatch work. Persisting translations across runs means pinning the guest's code identity, the host target, and every feature flag that changed codegen — a cache that is wrong once is worse than no cache, because the failure is a wrong result rather than a slow start.
- **Revisit when:** startup-dominated workloads matter more than they do now, and there is a way to key the cache that is cheap to check and impossible to get wrong.
- **Tracked in:** was task-84, closed into this entry.

### MXCSR and vector FP flag semantics

- **Deferred:** arithmetic does not set MXCSR's six sticky exception flags, and unmasked SIMD floating-point exceptions (`#XM`) are not delivered. Guest-written flags and masks round-trip unchanged. Host execution masks exceptions; masks and sticky flags are excluded from translation identity. This keeps result-mode support independent of precise SIMD exception reporting and delivery.
- **Implemented:** RC, FTZ and DAZ govern SSE/AVX arithmetic and conversions in every tier. Translations specialise to those three controls, and MXCSR writes end blocks before dispatching in the new mode (SDM Vol 1 §§10.2.3.2–10.2.3.4).
- **Comparison:** native snapshots capture sticky flags, but the general comparator excludes bits 5:0 until arithmetic flag reporting is implemented. Storage witnesses compare all 32 bits.
- **History:** MXCSR storage: xj-pi7.1; numerical controls: xj-pi7.2; earlier deferral: task-82; oracle capture: task-325.

### x87 FP exceptions — NOT deferred, tracked

Listed here only because the MXCSR entry above points at it and a reader looking for
"why does no FP exception fire" arrives at this file. Unlike the SSE side, the x87 half
**is** modelled: arithmetic sets all six status-word flags, the stack-fault flag with C1,
and the condition codes, and an unmasked exception is delivered as `#MF` on the next
waiting FP instruction. It is not a deferral, and the gaps that remain in it are tracked
as ordinary defects rather than listed here.

### x87 legacy environment, save/restore, packed BCD and control forms

- **Why deferred:** keep the 14-byte `fldenv`/`fnstenv` images refused: the
  existing 28-byte path cannot express their different field offsets and 16-bit
  pointers (Intel SDM Vol 1 §8.1.10, Figures 8-9 through 8-12). Both 94-byte and
  108-byte `fnsave`/`frstor` remain refused: implementing their logical stack
  ordering and environment together needs a separate save/restore witness suite;
  the environment-only path must not stand in for a complete register image
  (Intel SDM Vol 2A §3.2 FSAVE/FNSAVE and FRSTOR). `fbld`/`fbstp` need packed
  decimal conversion, range/invalid handling and exception tests, which the
  existing binary integer conversion does not supply (Intel SDM Vol 2A §3.2
  FBLD and FBSTP). `fneni`/`fndisi`/`fnsetpm` are legacy encodings outside the
  supported user-mode FPU control model; they stay rejected (Intel SDM Vol 1
  §8.3.13).
- **Revisit when:** an embedder needs one of these formats or legacy encodings,
  with hardware witnesses for its complete state transfer or conversion.
- **Tracked in:** `xj-pi7.3`. The ordinary arithmetic and stack forms still
  missing from that bead are open implementation work, rather than a deferral.

### Undocumented x87 FFREEP and FSTPNCE encodings

- **Why deferred:** `FFREEP` (`DF C0+i`) and `FSTPNCE` (`D9 D8+i`) have no
  instruction semantics in the pinned Intel SDM or AMD64 APM. A primary source
  is needed before either form lifts. Native-oracle measurements alone cannot
  establish the hardware contract. Both remain precise `Exit::UnknownInstruction`
  exits, witnessed by `x87_rest::undocumented_x87_forms_remain_precise_unknown_instructions`.
- **Revisit when:** a primary-source description is available, with witnesses for
  the documented semantics, exceptions and stack effects.
- **Tracked in:** `xj-562.7`.

### `lock adc` / `lock sbb`, and masked EVEX `vmovss`/`vmovsd`

- **Why deferred:** both are now *rejected* rather than mislifted (task-322) — that
  was the defect, and it is fixed. Supporting them is a different question. An atomic
  carry-propagating RMW needs a CAS-loop IR op that nothing else wants, and masked
  scalar moves need per-element predication including masked memory-fault suppression,
  which the IR cannot express. Neither has a consumer: no fixture and no reported guest
  has executed either form, which is why they sat mislifted for so long without anyone
  noticing.
- **Revisit when:** a real guest traps on one. It will now trap loudly, with the opcode
  bytes, which is the point of rejecting rather than guessing.
- **Tracked in:** task-322 (the rejection); this entry owns the support question.

### Optional hook-based API (alongside return-based)

- **Why deferred:** the core is return-based (`run()` → `Exit`) on purpose (§5.1). Hooks are a possible debugging convenience, not a contract.
- **Revisit when:** after M4, only if hooks prove useful. The return-based core stays authoritative (§14).
- **Tracked in:** —

### Real-mode JIT tier, 286 fidelity, and protected-mode segmentation

- **What ships:** `CpuMode::Long64`, `CpuMode::Compat32` (32-bit
  protected/compatibility mode with flat segments), and `CpuMode::Real16` (16-bit
  real mode). Real16 is **interpreter-only**; an embedder uses it. Real mode itself is not deferred.
- **Validation:** the [SingleStepTests 80286 harness](../../x86jit-tests/tests/ss286.rs)
  executes the hardware-captured corpus on the Real16 interpreter and compares
  final CPU/memory state and in-guest exception delivery. It reports unsupported
  opcodes and divergences rather than enforcing full conformance. Targeted
  [Unicorn comparisons](../../x86jit-tests/tests/cf16.rs) also exercise real-mode
  addressing and control flow.
- **Still deferred:** a Real16 JIT tier; remaining 286 fidelity items exposed by
  corpus failures, traps, and unsupported opcodes; and full protected-mode
  segmentation (segment descriptors and their enforcement). Real16 segmented
  addressing already exists; Compat32 remains flat apart from `FS`/`GS` bases.
- **Revisit when:** a consumer needs real-mode compilation or protected-mode
  segmentation, or a corpus gap requires a focused fidelity fix. Keep mode handling
  concrete; do not add speculative mode traits or abstractions.
- **History:** task-143 was closed into this entry before Real16 shipped.

### Other guest architectures (ARM/MIPS/6502) as a second front-end

- **Why deferred:** the IR already supports this — a new decoder + lift targeting the same IR reuses every backend (§17.1). Not a seam question, just unwritten work; don't scaffold for it speculatively.
- **Revisit when:** there's a concrete second-arch need.
- **Tracked in:** —
