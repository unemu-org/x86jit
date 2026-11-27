---
id: decision-29
title: 'Exact step accounting in compiled Machine blocks (J1)'
date: '2026-11-27'
status: accepted
---

Decider: maintainer (2026-11-27), accepting every recommended answer (Q1-Q8). Drafted 2026-11-27 for bead xj-to9.1, stage J1 of
decision-23 §8. It implements decision-27's "J1 accounting contract" and its amendment §4
and §6 (J1 row), and decision-28 stage 4 for J1's scope. It refines two texts, listed as
questions: decision-27's "reserve N, refund N−k" becomes "test N at entry, charge the
completed count at exit" (Q1), and decision-23's "a block adds its static cycle sum at
entry" becomes a prefix sum charged at exit (Q6). Everything else in decisions 16, 23, 27
and 28 stands. No code changes with this record. No Machine context reaches a backend
outside the test-only admission before J2 is accepted.

# Exact step accounting in compiled Machine blocks (J1)

## Context

J1 is the correctness stage: compiled Machine blocks must give the same results, step
counts, retired counts, exits, deliveries and device positions as the interpreter, at
every budget. Decision-23 §8: "Steps are exact in every tier". No speed-up is claimed.

### What the code does today (main be0b80d9)

| Piece | Where | What J1 relies on |
|---|---|---|
| Machine turn | `vm/legacy.rs:3041-3861` in `run_dispatch` (`:2841`) | In order: budget test (`:3070`), boundary (`:3076-3089`), host-request/NMI poll (`:3093-3135`), service, debug and IRQ delivery (`:3136-3234`), SMC drain (`:3255-3272`), link or resolve (`:3293-3379`), step cap `remaining` (`:3413-3424`), walk (`:3431-3517`), device positions (`:3518-3550`), fast arm (`:3580-3621`), outcome arm (`:3622-3861`, faults at `:3756`) |
| Walk result | `Progress` (`interp/legacy.rs:2794`) | `steps`, `retired`, shadows, `outcome`. Every count, shadow, retry, routing, link and device position after a block is derived from it |
| One-pass walk | `BlockPass` (`interp/legacy.rs:2903`), `commit` (`:3262-3328`) | Counts one step per committed instruction (`:3265-3266`). Stops at the block end, the limit, an exit, STI/SS arming, a nonzero dispatch word (`polled`, `:3276`) or a TLB token change |
| Failing instruction | `fail` → `leave_instruction` → `roll_back` (`:3163`, `:2541`, `:2591-2610`) | Restores the `Checkpoint` image if the instruction has one (`begin`, `:3125-3136`), else only RIP. An instruction without `Checkpoint` has made no architectural write before it fails |
| Flat32 blocks | `flat32.rs:191-274` | ≤ 32 instructions, one linear page; end at any non-`Next` flow or before a non-ordinary instruction (`flat32_ordinary`, `interp/legacy.rs:2689`), which is a singleton |
| Step cap | `devices.cap` (`device.rs:730-745`), `one_step` (`vm/legacy.rs:2738-2744`) | `remaining` already includes device deadlines, INTR with IF=0 before STI/POPF, TF, shadows, a pending IRQ, a pending A20 change and `stop_when_irq_ready` |
| Application compiled ABI | `CompiledFn` (`jit_abi.rs:23`), codes (`:26-81`), `MemCtx` (`:92-220`, append-only, frozen at `:1109`) | Application code defers port I/O, MMIO and unsupported ops: it sets RIP to the instruction, commits nothing of it and returns; the dispatcher runs that instruction on the interpreter (`codegen/control.rs:263-273`, `codegen/mod.rs:910-918`, `vm/legacy.rs:4739-4922`). It does not tick `retired`; `icount_ptr` is an approximate executed count (`jit_abi.rs:144-162`) |
| Time stats | `time_stats.rs:527-531` | Application switches to the compiled segment around its calls (`vm/legacy.rs:4432-4433`) and back (`:4020-4022`). The Machine turn switches no segment today |
| Cycles | — | Timing level 2 is not implemented: no `IrBlock::timing`, no cycle counters in the tree |

### The observation the design rests on

Everything exact about a Machine turn is computed after the walk, from one `Progress`. If
a compiled block hands the dispatcher the `Progress` the interpreted walk of the same
`IrBlock` would have returned, that code runs unchanged and exactness follows from it.
Application already uses the other half: whatever compiled code cannot finish exactly, it
hands to the interpreter at an instruction start.

## Decision

### 1. Principle

A compiled Machine block is a faster producer of the walk's `Progress`. Generated code
completes a prefix of the block's instructions and returns where the next instruction
starts. The dispatcher converts the return into a `Progress`. If the block stopped early,
the interpreter first walks the rest of the same `IrBlock` in the same turn. A compiled
block never builds a Machine `Exit`, delivers an event, touches a step or retired counter,
or writes the delivery log.

### 2. Entry contract

At `vm/legacy.rs:3413-3431`, after `remaining` is computed, the turn enters the compiled
unit only if all of these hold. Otherwise it runs the existing interpreted walk with
`remaining`.

| # | Condition | Why |
|---|---|---|
| E1 | The resolved proof or link carries a unit whose cached epoch equals `vm.cache.epoch()` (decision-27 amendment §3.1) | Code lifetime |
| E2 | `KIND == MACHINE_DISPATCH`, the walk is `BlockWalk::Direct`, and the context is in the admitted class: J1 test admission only, J2 Flat32 PG=0 CPL 0 | Admission |
| E3 | `!boundary.one_step` | Excludes TF, STI/SS shadows, a pending IRQ, a pending A20 change, `stop_when_irq_ready`, one-step budgets (`SINGLE`) and device INTR with IF=1 |
| E4 | `remaining >= ir.icount` | The budget, including the device deadline cap, covers every instruction |
| E5 | `!boundary.instruction_retry`, `!boundary.task_pending`, `!boundary.pending_io`, `event_retry.is_none()` | No retry, task continuation, MMIO retry, port completion or owned event is outstanding |
| E6 | `vm.mem.fetch_policy.is_none()` | As links and protected proofs (amendment §2.1 item 4) |
| E7 | `cpu.mxcsr & MODE_MASK == 0` | J1 units are compiled for the default MXCSR mode only |
| E8 | RF (EFLAGS bit 16) clear | A retiring instruction clears RF (SDM Vol 3B §20.3.1.1; the walk's `complete`); compiled code does not, so a block entered with RF=1 stays interpreted |

When E4 fails, the interpreter runs the block from the current IP with the exact
remaining budget, as it does today. A compiled prefix is not used (Q2). A zero budget
returns at `:3070` before anything is resolved, so it performs no fetch, helper, drain or
completion. That is unchanged, and the background-compile drain is placed after this
test.

### 3. Counting: test at entry, charge at exit

- Generated code never reads or writes `steps`, `Vcpu::retired` or `executed`. The
  dispatcher charges the block's completed count c in the existing merge code
  (`:3581-3583`, `:3622-3624`): `steps += c`, `retired += c`, `executed += c`.
- c is a constant. On the normal exit it is `ir.icount`, which the dispatcher knows, so the
  hot path stores nothing. On an early exit, generated code writes the ordinal baked at
  that exit site to an appended `MemCtx::machine_ordinal`. Spec.md §8.2.1 already keeps
  the `InsnStart` address as a compile-time constant; J1 adds its ordinal.
- This realizes decision-27's "reserve N, refund N−k". Nothing else consumes the budget
  while the block runs, so the entry test is the reservation, and charging c at exit is
  the reservation plus the refund. `Vcpu::retired` stays at its pre-block value during
  the block. A helper that needs a position (decision-28 device helpers, J2) is passed its
  ordinal k as an immediate and reports `retired + k`, exactly what the interpreted walk
  reports (`device.rs:1215-1218`). Decrementing at entry would make every such helper
  subtract.
- Every instruction a compiled block completes is an ordinary one (§4). It retires once
  and costs one step, so c counts both steps and retired instructions. REP and other
  variable units are never compiled in J1: strings are not ordinary, so they stay
  interpreted singletons, and their iteration ledger stays the interpreter's.
- Cycles (when level 2 lands): the compiled unit carries a prefix table
  `static_cycles[0..=N]` from `IrBlock::timing`. The dispatcher adds `static_cycles[c]`
  with the steps. Run-time parts (taken-branch difference, refill, data-dependent counts)
  are added by generated code to an appended `MemCtx` accumulator, only in the level-2
  specialization. With a cycle limit, E4 also requires the remaining cycles to cover the
  block's static sum plus the maximum of its run-time parts; otherwise the block runs
  interpreted. The witness is `jit == interp` on `StepRun::cycles` and
  `final_unit_cycles` (decision-23, "Interpreter and compiled accounting"). J1 adds no
  cycle code, because level 2 does not exist yet (Q6).

### 4. Exits from a compiled block

Three return codes. Two are new and appended to `jit_abi.rs:26-81` (Q7):

| Code | Generated code leaves | The dispatcher builds |
|---|---|---|
| `RET_CONTINUE` | All N instructions done, RIP = successor | `Progress { steps: N, retired: N, outcome: Some(Ok(Outcome { retired: true, ..Default })) }` |
| `RET_MACHINE_DEFER` (17), ordinal c | Instructions 0..c done; RIP = start of instruction c; nothing of c committed | Walks the same `IrBlock` from instruction c (its op index, found by a cold scan) with limit `remaining − c`, then `Progress { steps: c + p.steps, retired: c + p.retired, outcome: p.outcome, shadows: p.* }` |
| `RET_MACHINE_STOP` (18), ordinal c | Instructions 0..c done, the last one completed in full; RIP = start of instruction c | As `RET_CONTINUE` with c instead of N |

Every other code returned by Machine code is an engine bug and panics, as
`vm/legacy.rs:4933` does for Application.

The `RET_CONTINUE` and `RET_MACHINE_STOP` values are exactly what `execute_instructions`
returns when its walk ends normally or on a polled dispatch word with a default outcome.
They therefore take the same outcome arm (`:3622`). J2 may switch them to the fast arm
(`:3580`) after measuring.

**What is compiled in J1.** An instruction is compiled only if:

1. every op is in the J1 lowered set: `ReadReg`, `WriteReg`, the integer ALU and shift ops,
   `Sext`, `Mul`, `GetCond`, `BitScan`, `Bt`, `Bswap`, `SetCf`, `SetDf`, `Lahf`, `Sahf`,
   `Jump`, `Branch`, `Load`, `Store` and `Legacy(Flat32Address)`, with 1-, 2- and 4-byte
   transfers and general registers only. `DoubleShift` is lowered only at 32 bits: a 16-bit
   SHLD/SHRD count above 16 leaves the result undefined (SDM Vol 2B SHRD, "Operation"),
   and the shared lowering's answer differs from the interpreter's, which the lockstep
   harness found;
2. it has no `Legacy(Checkpoint)`;
3. it has at most one memory write, and no fallible op follows that write.

Condition 1 implies that the interpreter's outcome for the instruction is the default
one: only system, port, HLT, trap, interrupt, segment-load and flag-load ops set
`Outcome` fields. Any other instruction is lowered as an **unconditional defer at its
start**, and the rest of the block is not compiled, because nothing after an
unconditional defer is reachable. The lowering therefore never has to be complete, only
exact. Each J2 lowering moves ops into the set (Q5).

**Mid-block causes.**

| Cause at instruction k | Compiled code | Interpreter, then dispatcher | Charged |
|---|---|---|---|
| Fault: limit wrap at 4 GiB (`Flat32Address`), unmapped, `#DE` and similar | Defer at k. Nothing of k is committed (condition 2, plus the lifter invariant that `roll_back` relies on) | Re-executes k from the state before k, raises the fault, rolls back; `Err` arm → `route_legacy_exception` (`:3818`) | k |
| Port I/O, HLT, INT n / INT3 / INTO / BOUND, STI, MOV/POP SS, POPF, IRET, far transfers, control and descriptor ops | Unconditional defer at k. In J1 most of these never reach a compiled block: they are not `flat32_ordinary`, so the Flat32 lift makes each a singleton block, which stays interpreted (E2) | Its own handoff or delivery; a completed port/HLT/trap handoff counts k+1 (decision-16 §2) | k+1 or k, as interpreted |
| MMIO read or write, ROM write | The access helper (below) returns defer | Re-executes k: an MMIO exit (`machine_exit!`, `:2985-3033`), served in process (decision-28) or returned | k (MMIO adds no step until complete) |
| Own store into a code page, not absorbed by a live field | Instruction k finishes, then stop | Next turn sees `DISPATCH_DIRTY`, does not follow the link (`:3293`), drains, drops proofs, relifts | k+1 |
| Own store absorbed by a live field | Continues: live fields are loads (amendment §3.1) | — | N |
| Host request, NMI edge, another thread's code write | Not polled inside the block | Polled at the next turn (`:3093`, `:3255`) | N |

**Why the defer is exact.** On these processors a fault is reported with "the machine
state restored to the state prior to the beginning of execution of the faulting
instruction" (SDM Vol 3A §7.5 p. 7-4; 80386 PRM §9.1 p. 9-2). The interpreter already
implements that, and it is authoritative. A defer hands it exactly that state: the
pre-instruction registers (conditions 2-3), and memory where the deferred instruction
wrote nothing. The interpreter then produces the fault, exit, handoff count, retry ledger,
CR2, delivery and log entry itself. If another vCPU changes memory between the compiled
check and the re-execution, the interpreter's result is a valid later interleaving. The
defer holds whatever made the compiled path stop.

**Access helper (J1).** Every `Load` and `Store` calls one helper, which runs the existing
checked Flat32 access (`LegacyAccess`, with the walk's memory-consistency tier, physical
mask and A20). It first checks that every bus part of the transfer is RAM (or ROM, for a
read), and defers otherwise without calling the walk's `read`/`write`: on a device part
those would already record retry and pending-transfer state, which the re-execution must
own. A store also defers when it splits across bus parts, and in a VM with host-fault
recovery, where its raw write could fault part-way; so no store a deferred instruction
made is ever committed twice. The helper runs with RIP at the block's start: the walk's
retry ledger is keyed by RIP, and E5 guarantees there is none (asserted in debug builds). It returns the value, a defer, or (stores only) "stop after this
instruction". It returns the stop when this vCPU's dispatch word is nonzero after the
store. That is the interpreter's own commit test (`interp/legacy.rs:3276`), evaluated at
the only instructions in a compiled block that can make it true on this vCPU. The note-write,
watch and live-field absorption paths are the interpreter's, unchanged. The helper is the
miss path the inline TLB probe of J2 falls back to (amendment §3.2). J1 has no inline guest
memory access, so it has no host-fault landing.

### 5. Interrupt windows and asynchronous requests

Events are taken only between instructions (SDM Vol 3A §7.6 p. 7-5; 80386 PRM §9.2 p. 9-2).
The dispatcher decides them at turn boundaries in both tiers, in decision-16's order (NMI
before a queued IRQ, then IrqReady). E3 and E4, together with the deferred instructions of
§4, make this invariant hold: **no boundary at which the interpreter would decide
something is due strictly inside a compiled block.**

- **IRQ.** `inject_irq` takes `&mut Vcpu` (`vm.rs:2589`), so it cannot run during a block.
  A pending IRQ forces `one_step`, so E3 fails. A device's INTR is set only inside handler
  calls, which in J1 run in the dispatcher (`machine_exit!` → `devices.access`, then
  `continue`). IF=1 with INTR forces `one_step`, and IF=0 with INTR caps a block that
  holds STI/POPF to one step (`device.rs:741`).
- **TF, STI/SS shadows, NMI blocking.** They change only through deferred instructions.
- **Device deadline.** Capped into `remaining` before E4.
- **NMI edge, host request, another thread's code write.** These arrive through the
  dispatch word from another thread or a signal handler (`NmiLine`, `vm.rs:2066`). The
  interpreter notices one after any instruction of its walk and ends the walk. Compiled
  code notices it at the next turn, at most one block later (≤ 32 instructions, one
  page). Its arrival instruction depends on host timing in both tiers. Every compiled
  outcome is therefore one the interpreter produces for a slightly later arrival, and a
  request raised before the run is seen at the first turn in both tiers. Polling inside
  blocks is not needed before J2b chaining, which polls at chain edges, as Application
  does.

### 6. Delivery log, time stats, compile stats

- **Delivery log.** It is written only at the existing sites: `deliver_legacy_event`
  (`:2161`, records via `record_delivery`, `:2469`) and `record_unit` (`:2531`) in the cold
  epilogue. A compiled block never delivers and completes no software interrupt, so
  `record_unit` finds nothing in its part. A deferred suffix is an interpreted walk, so its
  INT is recorded from its `Progress` as today. The log's checkpoint and stall behaviour
  (`delivery_stalled`, `:2424`) is unchanged.
- **Time stats.** With `vm.time_enabled`, the turn calls `time_stats::compiled()` before
  the call and `time_stats::interpreted()` after it, as Application does. The deferred
  suffix runs in the interpreted segment. The J1 access helpers take the `time_enabled`
  argument of the helper ABI but record no helper time yet. Compile time goes to the
  foreground compile timer (`ForegroundCompileTimer`).

### 7. What the dispatcher re-derives after a compiled block

- **After `RET_CONTINUE` or `RET_MACHINE_STOP`:** what the outcome arm derives for the
  interpreted walk, and nothing more. A compiled instruction changes no context, segment
  cache, CPL, A20, TLB token or shadow (§4), so `context`, `direct_bus`, `sync_class`, the
  proofs and the links need nothing. Code written by the block is handled by the next
  turn's dirty test, which also keeps the link from being followed (`:3255`, `:3293`).
  `link_from` is set as for the interpreted walk (`:3680-3684`), except after a
  `RET_MACHINE_STOP`, whose RIP is not the block's exit and which links nothing.
- **After `RET_MACHINE_DEFER`:** the suffix walk's outcome drives everything as today:
  context change and translation invalidation (`:3698-3717`), A20 (`:3687-3697`), fault
  routing and retry slices of the full `ir` (`:3756-3858`), and device positions from the
  merged `Progress` (`:3518-3550`).
- **Around every call:** `fp_mode.compiled(mxcsr)` before and `fp_mode.default()` after
  under a non-default MXCSR, as `:4451-4462` does. (MXCSR writes defer in J1.) The cached
  unit pointer is checked against the epoch at each entry (E1), not after the call.
  `MemCtx::dispatch_word_ptr` is set for Machine runs too (`:2889-2890`), for the helpers.

### 8. Code that changes in J1

| Place | Change |
|---|---|
| `jit_abi.rs:26-81`, `:219` | `RET_MACHINE_DEFER = 17`, `RET_MACHINE_STOP = 18`; append `machine_ordinal` and `machine_value` (out) and, later, the level-2 cycle accumulator. The frozen-offset test (`:1109`) still passes |
| `vm/legacy.rs:2886-2898` | Machine `MemCtx` gets `dispatch_word_ptr`, for the store helper |
| new `jit_abi/machine.rs`, `interp/legacy/machine_jit.rs` | The classifier, the access helpers, the suffix walk |
| `vm/legacy.rs:3413-3517`, new `vm/legacy/compiled.rs` | E1-E8, then the compiled call and the `Progress` conversion (§4); the interpreted walk otherwise |
| `interp/legacy.rs:2818`, `:2993-3001` | `execute_instructions` and `BlockPass::run` take a starting op index for the deferred suffix |
| `vm.rs` `Backend` | `materialize_machine`, default `None`: a synchronous compile of one Machine block, not published to the shared cache |
| `x86jit-cranelift/src/codegen/mod.rs:895` (`op`), `codegen/control.rs:243-273`, `codegen/mod.rs:910-918` | A Machine parameter of the shared translator: ordinal tracking at `InsnStart`, the J1 lowered set, an unconditional defer for everything else, the access helper calls, no Application exit codes |
| `vm.rs:192`, `:211`, `:234`, `:255`, `:500`, `:4613`; `x86jit-cranelift/src/lib.rs:4223`, `:4296` | Unchanged: the Application paths still reject Machine IR. Machine IR reaches a backend only through `materialize_machine`, which the dispatcher calls only under the test-only `MachineJitMode` (settable only with `test-hooks`) |

**J1 units are per vCPU.** J1 keeps each compiled unit in a per-vCPU map keyed by the
`Arc<IrBlock>` it was compiled from, which the map also holds. The unit is valid whenever
the resolver or a link returns that block, because it is a function of the block's IR
alone: every input that is not in the IR (A20, regions, the dispatch word) is read live by
the helpers. The Machine key, the shared Machine map, background compilation with
proof-generation requests, hotness in the proofs and links, and the cached pointers in
proofs and links (amendment §2, §3.1) move to J2, where sharing across vCPUs matters.

Each J1 unit is registered with the backend's code lifetime (spec §9.1, decision-17) and
owned by its vCPU, which retires it when only the map still holds its block (a sweep
every 64 new units), when the map reaches 512 units, when the VM's code lifetime changes,
and when the vCPU is dropped. Every compiled call runs inside the vCPU's execution
participation, entered just before the call, so retired code is not reclaimed while it
runs. E1 is that participation's check: a unit whose epoch is not the guard's is called
only if the lifetime still admits it.

### 9. Testing J1 before J2

J1 is tested before J2 exists in four modes, on the x86-64 and ARM64 hosts:

1. **Interpreter only:** the reference.
2. **Forced, synchronous:** `MachineJitMode::Forced`. Every eligible Flat32 PG=0 CPL-0
   block compiles at its first execution.
3. **Delayed:** `MachineJitMode::Delayed(k)`, k = 1 and 3: a unit compiled at the first
   execution is entered only after k more eligible turns, so a witness switches tiers in
   the middle. This stands in for background publication until J2 has the worker path.
4. **Scripted units (`test-hooks`):** a seam at the compiled call substitutes a scripted
   unit for generated code. The unit completes instructions 0..c on the interpreter, then
   returns `CONTINUE`, `DEFER(c)` or `STOP(c)` for **every** c in 0..N of every block.
   Deferring or stopping is legal at any instruction start, so this checks the
   dispatcher's half of the contract on every program, independently of what the J1
   lowering covers. One restriction mirrors the store helper: a prefix after which the
   dispatch word is set stops; it never defers into the rest of a block whose bytes may
   have changed.

Modes are selected with `Vm::set_machine_jit_for_test` or, in `test-hooks` builds, with
`X86JIT_TEST_MACHINE_JIT=forced|delayed:K|scripted:SEED`. The existing witness files build
their machines on non-compiling backends, so they run in mode 4 through the variable;
modes 2 and 3 run in the lockstep harness, which builds its machines on the Cranelift
backend. Three interpreter cost probes (`legacy_one_pass` warm-block walk counts,
`legacy_smc_selective` lift counts, the `vm/legacy/live.rs` relift censuses) count turns and
lifts, which compiled units change without an observable difference; they select `Off`.

**Lockstep harness** (new, `x86jit-tests/tests/legacy/machine_j1_lockstep.rs`). It pairs
each witness program as reference and subject, as `legacy_one_pass.rs:264` does already.
Both run the same budget schedule: constant budgets 1, 2, 7 and 64, and from every block
start each budget 0..=N+1, with operands from a seeded generator. After each `run_steps`
it compares the full `CpuState` (flags and legacy state included), the RAM the programs
write, `StepRun` and `retired_instructions()`, and all RAM and ROM at the end of a run;
it answers MMIO and port exits identically on both machines. Its programs include every
lowered op family and width (each asserted to run with no defer), read-modify-write
stores and stores into code, whose repetition after a wrongly committed defer, or a
DEFER in place of a STOP, shows in state (both mutations were checked to fail), splits
across RAM, MMIO and ROM, A20 off with the 1 MiB wrap, and the refusals E3, E7 and E8,
asserted not to enter a unit. A further witness checks that no unit outlives its vCPU. Delivery logs, device-request
transcripts and NMI/IRQ boundaries are checked by `legacy_delivery_log`,
`vm/device/tests.rs`, `legacy_boundary_events` and `irq_shadow` in mode 4. Budget 1 never
enters compiled code (E3). It still checks that the tiers interleave correctly across
calls.

### 10. Done when, and witnesses

J1 is done when:

1. The witnesses below pass in all four modes on both hosts.
2. Each forced witness reports compiled-retired instructions > 0 and an exit-reason census
   that covers every row of §4's cause table (test-hooks counters), so no pass is
   vacuous.
3. Application is unchanged: the frozen ABI test, the `BlockKey` hash test, and no
   Application cell past the two-profile perf gate.
4. With compilation off, no Machine cell is past the gate: the entry check costs one test
   of the resolved slot.
5. Production guards still reject Machine IR, and `x86jit-core`'s dependencies stay
   `{iced-x86}` (`x86jit-tests/tests/codegen/boundary.rs`).

**Witnesses**, run in modes 2-4 (only Flat32 PG=0 CPL-0 contexts compile; the rest are
the regression set):

- `legacy/`: `legacy_one_pass`, `legacy_flat32`, `legacy_rollback_sweep`,
  `legacy_smc_positions`, `legacy_smc_selective`, `legacy_code_alias_smc`,
  `legacy_boundary_events`, `legacy_delivery_log`, `legacy_delivery_liveness`,
  `legacy_tf_loop`, `legacy_debug_test`, `legacy_census_386`, `legacy_i486`,
  `legacy_contract`;
- `exits/`: `retired_count`, `run_steps`, `irq_shadow`, `exit_request`, `deferred_exit`;
- in-crate: `vm/legacy/{link_witness, return_witness, live, exit_census}.rs` and
  `vm/device/tests.rs` (the decision-28 census, with in-process devices);
- new: the lockstep harness, whose programs are the defer census of modes 2-3: register,
  memory and call/return loops; HLT, #UD, #DE, #GP and #SS at the end of the linear
  space, and IN, each at two positions in a block; MMIO reads and writes; a store into
  the next instruction and a store into another block, each with TF clear and set. Mode 4
  defers and stops at every ordinal of every block of every witness above, with and
  without in-process device handlers (`vm/device/tests.rs`).
- moved to J2, with the background worker and the shared Machine map: Machine analogues
  of `threads/same_block_smc.rs` and `threads/bg_tier.rs` (a compile racing a patch, a
  promotion and a code-lifetime reset), a device-position witness in modes 2-3 (J1's
  compiled code calls no device handler), and an absorbed live-field store compiled (J1
  defers every `LiveField` instruction).

## Options considered

| Choice | Options | Recommendation |
|---|---|---|
| Budget not covering N | (a) the interpreter runs the block; (b) a compiled prefix with a fuel count per instruction; (c) N entry points; (d) lift blocks to the budget | **(a)**: no per-instruction cost and no new code shape. A short budget is rare in production slices |
| Counting | (a) test at entry, the dispatcher charges c at exit; (b) generated code increments `retired` per instruction; (c) decrement at entry and refund | **(a)**: nothing in the hot path, and helpers see the interpreter's positions |
| Mid-block exits | (a) defer to the interpreter at the instruction start; (b) compiled code and helpers build Machine exits, ledgers and routing; (c) a helper runs the whole instruction on the interpreter and returns its `Outcome` | **(a)**: the Application JIT's existing mechanism, exact by construction, no second fault or ledger implementation. (b) duplicates the interpreter; (c) needs temps marshalled out of SSA |
| Own code writes | (a) stop after any own store that leaves the dispatch word set; (b) stop only when it overlaps the running block (amendment §3.3) | **(a)** in J1: it is the interpreter's rule. (b) is a later refinement, since a turn boundary with nothing due is unobservable |
| Asynchronous requests | (a) at turn boundaries; (b) poll the dispatch word at every instruction | **(a)**: no load and branch per guest instruction, and no observable difference (§5) |
| Testing before J2 | (a) a minimal Cranelift lowering plus scripted units; (b) scripted units only; (c) wait for J2 | **(a)**: real generated code on every witness, plus an exhaustive check of the contract at every ordinal |
| Cycles | (a) a prefix sum at exit; (b) the static sum at entry with a refund | **(a)**: the same shape as steps; helpers see the pre-block value |

## Risks

- **The classifier drifts from the interpreter.** A lifter change could give an op in the
  lowered set a non-default `Outcome`, and compiled code would then continue where the
  interpreter stops. Mitigation: one classifier function next to `flat32_ordinary`, a
  debug assertion that the set's ops never set `Outcome` fields, and the census in
  done-when item 2.
- **Re-executed RAM stores.** Condition 3 keeps a deferred instruction from having
  written. If it is relaxed later (multi-store instructions such as PUSHA or ENTER), a
  re-execution writes RAM twice. Another vCPU can observe that, and it notifies SMC
  twice. Relaxing it needs probe-before-first-store or a compiled rollback image (J2).
- **Vacuous coverage.** With a small lowered set, most blocks defer early. Scripted units
  and the compiled-retired counters keep the witnesses honest; J2's lowering shows the
  speed.
- **Latency of host requests.** Up to one block (32 instructions) later than the
  interpreter. Tests that raise a request from another thread at a precise instruction
  would be flaky by design. None is known; the `exit_request` witnesses raise before the
  run.
- **Cost with compilation off.** E1 adds a load and a test per turn, and E2's hotness count
  an increment. Both are measured under done-when item 4.
- **Paged obligations (J3+J4).** A TLB token change inside a block must stop it after the
  instruction (`interp/legacy.rs:3286`). A/D-bit writes of a deferred instruction must be
  idempotent. Both belong to J3's record, through the same STOP and DEFER codes.

## Maintainer questions

| # | Question | Options | Recommendation |
|---|---|---|---|
| Q1 | Realize decision-27's "reserve N, refund N−k" as "test N at entry, charge the completed count at exit"? | (a) yes; (b) decrement at entry and refund | **(a)**: the same counts, with nothing in generated code |
| Q2 | A budget below N: interpreter for the whole block, never a compiled prefix? | (a) interpreter; (b) prefix with per-instruction fuel | **(a)** |
| Q3 | Defer to the interpreter at the instruction start as the only way a compiled block ends early, with stop after own code writes? | (a) defer and stop; (b) compiled exits built in generated code | **(a)**: one mechanism, shared with Application |
| Q4 | Host requests, NMI edges and other threads' code writes noticed only at turn boundaries? | (a) yes; (b) poll per instruction | **(a)** until J2b, whose chain edges poll |
| Q5 | J1's lowered set as listed, with every other instruction an unconditional defer (Checkpoint, multi-store, Div, near transfers and RET included)? | (a) minimal, widened by J2; (b) a broader set in J1 | **(a)**: J1 proves the contract, J2 earns the speed |
| Q6 | Cycles as a prefix sum charged at exit, amending decision-23's "at entry"; no cycle code in J1? | (a) yes; (b) keep "at entry" with a refund | **(a)**, implemented with level 2 in both tiers |
| Q7 | New return codes `RET_MACHINE_DEFER` / `RET_MACHINE_STOP` and an appended `machine_ordinal`, rather than reusing Application codes? | (a) new codes; (b) reuse `RET_MMIO_DEFER` and the others with an ordinal | **(a)**: the Application dispatch stays untouched, and one Machine meaning per code |
| Q8 | Require `!one_step` at entry, so an N=1 block under TF, a shadow or a pending IRQ stays interpreted? | (a) yes; (b) allow N = 1 under `one_step` | **(a)**: simpler; revisit if a workload runs long with IF=0 and an IRQ pending |

## Consequences

- J1 adds no second fault, retry, delivery or device path. Compiled Machine code is
  correct whenever the lowered prefix agrees with the interpreter, which the lockstep
  harness checks instruction for instruction.
- J2 becomes a widening of the lowered set (inline TLB probe, live-field loads, near
  transfers, Checkpoint instructions), each under the same witnesses. The accounting does
  not change again.
- Revisit if defers dominate J2's measured turns (then lower the deferring op), or if a
  real workload depends on instruction-precise latency of host requests.
