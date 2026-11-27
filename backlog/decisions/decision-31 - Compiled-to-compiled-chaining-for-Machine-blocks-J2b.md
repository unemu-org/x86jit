---
id: decision-31
title: 'Compiled-to-compiled chaining for Machine blocks (J2b)'
date: '2026-11-27'
status: accepted
---

Decider: maintainer (2026-11-27), accepting every recommended answer (Q1-Q9).
Drafted 2026-11-27 for bead xj-to9.54, stage J2b of decision-27's amendment of 2026-11-26
(§3.4, §6 J2b row). Decision-30 Q8 starts it: after J2 (b409168b, in review) a linked compiled
turn still costs about 530 host instructions, and register and callret miss decision-30's
done-when 2. It builds on decision-29 (J1) and decision-30 (J2) and changes four accepted
texts, each listed as a question: decision-27's and decision-30 §6's "inside one page first"
(Q3), decision-29 Q7's "every other return code panics" (Q4), decision-30 §6's "Machine regions
(T2) come with J2b" (Q5), and decision-29 §4's "the hot path stores nothing" for the edges
only (Q7). Everything else in decisions 27, 29 and 30 stands. No code changes with this record.

# Compiled-to-compiled chaining for Machine blocks (J2b)

## Context

### What J2 left (b409168b)

| Piece | Where | State |
|---|---|---|
| Machine turn | `vm/legacy.rs:3148` `run_dispatch`; per turn: device boundary, limit, drain (`:3407`), boundary refresh, exit/NMI poll, service, SMC drain (`:3588`), link or resolve (`:3631`), `remaining` (`:3750`), compiled call (`:3783`), fast arm (`:3796-3815`) | Every compiled block returns here |
| Compiled entry | `vm/legacy/compiled.rs:380` `machine_progress`: E2-E8, memo, participation (`:481`), call (`:549`), DEFER suffix (`:592`) | Kept for a chain's first block |
| Exits in generated code | `codegen/machine.rs:277-353`: LOOP, JMP and Jcc store RIP and return `RET_CONTINUE`; CALL and RET likewise | Become chain sites (§3) |
| Successor links | `vm/legacy/links.rs:324` `follow_link`: per vCPU, IP-keyed, validated by proof generation, context id, version, run | Unchanged; serve the turn that starts a chain |
| Application chaining | `chain_or_link` (`codegen/mod.rs:5553`): a filled slot returns `RET_CHAIN` with `next_entry`, an empty one `RET_LINK` with `link_slot`; a shadow return ring (`:5644`, `:5671`, `jit_abi.rs:266`); the host-side chain loop polls the dispatch word and the epoch at each edge (`vm/legacy.rs:4825-5347`, poll at `:5334-5347`) | The mechanism J2b parameterizes |
| Application regions | in-code loops with a fuel gate and an atomic dispatch-word poll per sub-block (`codegen/mod.rs:5736`) | Not in J2b (Q5) |
| Code lifetime | slots registered with `register_slot` are zeroed by every retirement before the epoch advances (`code_lifetime.rs:172-190`, `:229-250`); IBTC descriptors retire after execution grace | Shared |

### Measured, and what it implies

`machine32_loops`, release, ARM64, host instructions per guest step (decision-30 implementation
notes, MEASURED): J2 background (T1 = 50) 124.5 / 138.5 / 370.3; Compat32 JIT on the same bytes
18.4 / 41.8 / 68.8; gates 102 / 133 / 326. With t turns per step (0.2 / 0.2 / 0.6), the excess
per turn over the Application JIT's chained edge is (J2 − J) / t = **532 / 484 / 503**
(DERIVED). The profile spreads it over the turn body (~30 %), `machine_progress` (~25 %),
`follow_link` (~12 %), the per-turn `Arc` (~6 %) and the SMC poll (~5 %); generated code is
~8 %. No item exceeds a tenth of the turn, so no single trim reaches the gates. Removing the
turn between two compiled blocks does. The Compat32 figures already pay one Application chain
edge per block, so they measure what a chained block plus a host-side edge costs on these bytes.

### What a turn decides, and what a chain edge must do instead

A chain edge skips the turn. It is exact if every decision the turn makes there is either made
at the edge or provably has nothing to decide. The fast arm already relies on the second half
for the boundary (decision-30 §6 item 2): a compiled block changes no boundary input.

| Turn step | Can a compiled block change its input? | At a chain edge |
|---|---|---|
| Port-string completion (`:3375`) | No: IN/INS defer | Nothing |
| `devices.boundary` (`device.rs:669`): deadline, INTR with IF=1, queued invalidation | Deadline: steps advance. INTR and invalidation: only handler effects, which defer (decision-30 §4) | Deadline: the edge budget test (§4.1). INTR with IF=1 and invalidation force `one_step`, so no chain starts. INTR with IF=0 decides nothing (`device.rs:698-713`); compiled code never sets IF |
| Step limit | Steps advance | The edge budget test |
| Machine drain | No | Skipped. Publication latency only |
| Boundary refresh | No (E3-E5 inputs) | Kept, as the fast arm keeps it |
| Exit request, NMI edge (`:3426`) | Other threads | Poll of the whole dispatch word |
| Service (`:3469-3557`) | `handoff_service` changes only through handoffs (deferred); numeric reconciliation (`state.rs:590`) is idempotent while x87 state is unchanged, and x87 and FWAIT defer | Nothing |
| SMC drain (`:3588`) | Another thread's code write sets `DISPATCH_DIRTY`; an own one STOPs | Poll (same word) |
| Link or resolve | No context, A20, MXCSR or mapping change | The successor slot plus a flags compare (§2) |
| `remaining` and `devices.cap` (`device.rs:730`) | The STI/POPF clause never applies: compiled blocks hold neither | `remaining − chained` |
| E1-E8 (`compiled.rs:394-414`) | E1 by retirement; E3, E5-E8 no; E2's class no | E1 epoch test, E2 flags compare, E4 budget test; E3, E5-E8 invariant |
| In-block device hook (`:3781`) | — | Re-armed with the successor's block and position (§4.4) |
| Fast-arm counts (`:3796-3815`) | — | Charged once when the chain ends, prefix first (§2.3) |
| Cold epilogue (delivery log, NMI blocked, A20 pending) | — | No chain starts (Q6) |

## Decision

### 1. Principle

A chain is a sequence of J1 entries with no turn between them. Each entry satisfies E1-E8,
either tested at the edge or invariant since the chain's first entry (table above). The
dispatcher charges every completed instruction when the chain ends, so decision-29's counting
is unchanged: nothing in generated code touches steps or retired counts. Events are taken only
at instruction boundaries (SDM Vol 3A §7.6 p. 7-5; 80386 PRM §9.2 p. 9-2), and a chain skips only
boundaries at which the turn would decide nothing. A boundary with nothing due is unobservable,
which the interpreter's own budget-cut walks and J1's STOP already rely on.

### 2. Mechanism: a host-side Machine chain loop over shared successor slots

**2.1 Generated code.** Every static exit of a Machine unit (§3) calls the shared
`chain_or_link` with a slot from `alloc_slot`, as Application's direct exits do. A filled slot
returns `RET_CHAIN` with `MemCtx::next_entry` = the slot's value; an empty one returns
`RET_LINK` with `MemCtx::link_slot`. Near CALL pushes `(return address, continuation slot)` on
the vCPU's shadow ring (`emit_ret_push`), and near RET predicts through it (`emit_ret_predict`):
`RET_CHAIN`, `RET_LINK`, or `RET_RETURN_MISS` on a mismatch or underflow. In Machine code all
three mean "all N instructions completed, RIP holds the successor" (Q4). The ring only gates a
chain; a hit requires the popped EIP to equal the predicted return address. The one Machine
parameter: the slot is read with an acquire load (one `ldar` on ARM64), because its value is a
descriptor pointer, not an entry.

**2.2 The slot holds a target descriptor.** A filled Machine slot points to an immutable
`MachineTarget { entry: CompiledPtr, ir: Arc<IrBlock>, icount: u32, flags: u32, link_exit: bool }`,
allocated and retired through the code lifetime as IBTC descriptors are (`cache.rs:1125`),
published with Release. `icount` serves the edge's budget test, `flags` (the target's
`MachineBlockKey::flags`) the edge's class test, and `ir` the device hook and a DEFER suffix.
`ir` is the block the filling turn entered, another lift of the same bytes under the same key,
which decision-30 §5 makes interchangeable.

**2.3 The chain loop** runs inside `machine_progress`, under the one participation guard the
first call already enters:

```text
first block A: J2 entry (E1-E8), remaining R, chained = 0, cur = A, flags = A's key flags
loop {
  code = call(cur.entry)                       // ctx.machine_ordinal = 0 before
  if code == RET_CHAIN {
    chained += cur.icount; t = next_entry as &MachineTarget
    if word != 0 || lifetime.epoch() != guard.epoch() || t.flags != flags
       || R - chained < t.icount   { end: before t, RIP = t's start }
    if hook armed { re-arm(t.ir, retired + chained, steps + chained) }
    vcpu.chained += 1; cur = t; continue
  }
  end: cur's result (RET_CONTINUE / RET_LINK / RET_RETURN_MISS / STOP(c) / DEFER(c))
}
```

When the chain ends, the dispatcher first charges the prefix (`steps`, `retired`, `executed`
+= chained; device `unit_start` advanced; `instruction_ip` = cur's start), then handles cur's
result exactly as J2 handles a turn's only block, with `ir` = cur's IR and `remaining` =
R − chained: the fast arm for completion and STOP, J1's same-turn suffix walk for DEFER (Q7),
`link_from` from cur's `link_exit` (never after STOP). Every J1 and J2 path therefore runs
unchanged on the last block. A chain that ends before t is a completion of cur at RIP = t.

**2.4 Filling a slot.** On `RET_LINK` the turn keeps `fill = (slot, target, flags, epoch)`, the
epoch read before anything is resolved. The next turn publishes a descriptor for the unit it
enters, through `publish_entry_if_current` (`code_lifetime.rs:207`), only if `linked` survived
that turn's boundary (no event, service, exit or drain), the unit's IP is `target` and its key
flags equal `flags`. Any other outcome drops `fill`. A retirement in between fails the epoch
test, so a slot a retirement cleared is never refilled with a dead entry (Application's
`RET_LINK` rule, `vm/legacy.rs:4880-4930`). `Delayed(k)` is honoured: a slot is filled only by
an actual entry.

**2.5 Why one fill serves every vCPU.** A slot belongs to a shared unit A, so another vCPU may
follow it. In the J2 class (Flat32, PG = 0, CPL 0) the successor's key is a function of A's key
and the target: `decode_ip` is the target; `physical_pc` is the target through the bus mask and
A20, both fixed by A's model id and flags; `model_semantics_id` is A's; `flags` are A's, because
a compiled block changes no context, A20 or CR0; `mxcsr_mode` is 0 (E7). No per-vCPU translation
exists without paging. The flags compare in the loop is redundant for direct slots and needed
for the ring, whose entries outlive a context change between CALL and RET (an A20 toggle in the
callee). A vCPU that follows a slot before its own T1 count was reached only gains speed:
results are the interpreter's in every tier (decision-23 §8).

### 3. Which exits chain

| Exit | J2b | Note |
|---|---|---|
| JMP rel8/rel32 | chain | |
| Jcc, both arms | chain | Two slots |
| LOOP, LOOPE, LOOPNE, JECXZ, both arms | chain | `codegen/machine.rs:277-322` |
| Near CALL direct, to the callee | chain | Ring push first |
| Fall-through: a block cut at the lifter's bound or before a non-ordinary instruction | chain | The target is static. A singleton successor is never compiled, so its slot stays empty and costs one load per exit |
| Near RET, RET imm16 | chain through the ring | A miss returns to the turn, whose IP-keyed link serves it (`links.rs`, xj-to9.30) |
| Near JMP/CALL r/m (indirect) | turn | Q9 |
| Every deferred instruction (far transfers, INT, IRET, HLT, I/O, strings, x87, segment loads, POPF/STI/CLI, LOCK) | ends the chain with DEFER | J1 suffix in the same turn |
| An own store into code | ends the chain with STOP | Links nothing (decision-29 §7) |

**Cross-page edges (Q3).** Unpaged, the target page's physical address is the target through
the bus mask and A20 (§2.5); no translation can change inside a chain, and regions change only
between runs (`Vm::map` takes `&mut Vm`). J2b therefore chains across pages. **Rule for J3+J4:**
paged, a compiled data access may refill or replace a fetch-used TLB entry inside a chain, so a
cross-page edge must validate the target page's fetch translation (or stay within its page),
and J3+J4's record decides which.

### 4. Exactness at chain edges

**4.1 Budget.** R is the turn's `remaining`, which already holds the step limit and the device
deadline cap (`device.rs:730-745`). An edge enters t only if `R − chained ≥ t.icount`: J1's E4 at
every chained entry, so a chain never runs past the limit or a deadline. Each compiled
instruction adds one step and one retired instruction, so the cap's step and retired-instruction
deadlines both stay exact under one subtraction. A budget short of t's N ends the chain before
t; the next turn runs t interpreted with the exact remainder (decision-29 Q2). Budget 1 never
enters compiled code (E3), budget 2 never enters a five-instruction block, budget 7 enters one
and stops, budget 64 chains twelve.

**4.2 Counts at any exit.** Charged = Σ(N of completed blocks) + {cur's N on a completion, c on
STOP(c), c + suffix on DEFER(c)}. Generated code still stores only `machine_ordinal` at an early
exit; the edge's counting is in the loop (Q7). Retired, steps, executed and the device positions
therefore equal the interpreter's at every exit.

**4.3 Asynchronous events.** Each edge loads the dispatch word once and ends the chain on any
nonzero value: `EXIT_REQUESTED`, `NMI_EDGE`, `DISPATCH_DIRTY`, `CODE_SPACE_EXHAUSTED` and the
child-dirty bits. This is decision-27 §3.4's "chain edges poll", as Application's chain loop does
(`vm/legacy.rs:5342`). Cost: one load and one test per edge. Latency: at most one block
(≤ 32 instructions) after arrival, J1's bound (decision-29 §5), now per block instead of per
turn. A cross-modifying store by another vCPU is seen at the next edge; executing at most one
more block of the old bytes is an interleaving the processor allows: compliant cross-modifying
code has the executing processor run a serializing instruction before the new code (SDM Vol 3A
§11.1.3 pp. 11-5–11-6), and such instructions defer, ending the chain. The epoch test (§5) closes
the rest. IRQs: a pending IRQ, `stop_when_irq_ready`, TF and the STI/SS shadows force
`one_step` at the chain's first entry (E3), and none can arise inside a chain: `inject_irq` needs
`&mut Vcpu`, TF and IF change only through deferred instructions (single-step traps follow the
instruction after the one that sets TF, SDM Vol 3B §20.3.1.4 p. 20-11, and POPF defers).

**4.4 Devices.** The in-block hook (`device.rs:1046`) is armed for the chain's first block. At
each edge with a hook armed the loop re-arms it with `t.ir`, `retired + chained` and
`steps + chained`, so a handled MMIO access in the k-th chained block reports the position the
interpreter reports (decision-30 §4) and `sole_access` reads the block that runs. A handler
result kept in `Run::decided` makes the helper DEFER, and J1's suffix walk of cur consumes it in
the same turn, before any boundary: the handler is never called twice (Q7). `devices.unit_start`
advances once, with the prefix.

**4.5 Delivery log, time stats, counters.** No chain starts under the cold epilogue: an enabled
delivery log, a blocked NMI (`nmi_state ≠ 0`) or a pending A20 change (Q6). Compiled blocks
deliver nothing and record nothing (decision-29 §6), so this is cost only. With `time_enabled`
a chain is one compiled segment: `time_stats::compiled()` before the first call and
`interpreted()` after the last, as Application brackets its chain loop. Chained transfers add to
`Vcpu::chained`, flushed into `TranslationCache::add_chained` per run, so `executed / chained`
keeps its meaning. The witness counters (entries, continues, compiled-retired) count chained
entries too, in test and test-hooks builds only.

**4.6 Other entry conditions.** MXCSR mode stays 0 (MXCSR writes defer), RF stays 0 (E8 holds at
the first entry and compiled code sets no RF), the fetch policy and mapping are fixed for the
run (E6), and no retry, task continuation or pending I/O can begin without a deferral (E5).

### 5. Invalidation of chain edges and code lifetime

| Event | What ends or prevents the edge |
|---|---|
| Own store into a code page | STOP: the chain ends; the next turn drains |
| Another vCPU's code store | `DISPATCH_DIRTY` in every vCPU's word (`memory.rs:2205-2215`): the next edge ends; the drain invalidates the unit, which zeroes every registered slot and advances the epoch |
| Absorbed live-field store | Nothing: unit, slot and descriptor survive, and compiled code reads the field live (decision-30 §5) |
| Selective proof drop | Nothing to do: slots name shared units, never private proofs. A drop overlapping a unit invalidates the unit too |
| Context change, CS load, CPL, CR0, A20, MXCSR | Only deferred instructions do them, ending the chain. The next chain starts with its own flags; a stale ring entry fails the flags compare |
| Unit retirement, reclamation, code-lifetime reset | Slots zeroed before the epoch advances (`code_lifetime.rs:229-250`). An edge that loaded a slot just before is caught by the epoch test. The participation guard pins the old epoch, so neither code nor descriptors are reclaimed while a chain may still hold them |
| Map, unmap, Trap map | Between runs; overlapping units are invalidated (`vm.rs:1617-1700`) |
| `invalidate_translations`, context eviction | Private state only; units and slots survive (decision-30 §7) |
| Fork, snapshot, restore, register edits | Between runs. Every run's first block is a full turn with E1-E8 on the current state, and every chain takes its flags from that state |
| Shadow ring | Its continuation-slot pointers are discarded whenever this vCPU observes a new execution epoch, as Application's `fast_clear` does (`vm.rs:2947-2956`) |

### 6. Code that changes in J2b

| Place | Change |
|---|---|
| `x86jit-cranelift/src/codegen/machine.rs:277-353`, the Machine CALL/RET and fall-through paths | Static exits call `chain_or_link`; CALL pushes the ring; RET predicts; indirect exits keep `RET_CONTINUE` |
| `codegen/mod.rs:5553`, `:5644`, `:5671` | A Machine parameter: the acquire slot load. Application code is byte-identical |
| `jit_abi.rs:27-93` | Documentation only: `RET_CHAIN`, `RET_LINK` and `RET_RETURN_MISS` are legal from Machine code. No new code, no new `MemCtx` field; the frozen offsets test passes |
| `vm/legacy.rs:3203-3228` | The Machine `MemCtx` gets `ret_stack` |
| `vm/legacy/compiled.rs:380-590` | The chain loop; `Compiled` reports the prefix, the last block's IR and a pending fill |
| `vm/legacy.rs:3783-3830` | Charge the prefix, then the J2 handling of the last block; publish a pending fill at the next entering turn |
| `code_lifetime.rs`, `cache.rs:1125` | `MachineTarget` descriptors beside IBTC descriptors, retired after execution grace |
| `vm/device.rs:1046` | `arm` re-entered per edge |
| `x86jit-bench/examples/machine32_loops.rs` | A test-only chain-off variant of `machine-protected32-bg`, and its sliced (`run_steps(64)`) form |

### 7. Expected cost and the gate

Model: per step = J + t·ΔE + a·ΔP, with J the Compat32 JIT (measured, chain edges included), t
and a as decision-30 §9, ΔE the Machine edge's cost over Application's (descriptor and flags
loads, budget test, re-arm only with devices: **ASSUMED 5-20**), and ΔP the probe's cost over
Application's inline access. J2's memory figure leaves no measurable ΔP (484 versus 532 per
turn), so it is **ASSUMED 0-6**.

| Workload | J2 bg | J2b, RET chained | J2b, RET to the turn | ΔE = 50 (revisit trigger) | Gate |
|---|---:|---:|---:|---:|---:|
| register | 124.5 | 19-22 | — | 28 | 102 |
| memory | 138.5 | 43-48 | — | 54 | 133 |
| callret | 370.3 | 72-83 | ≈ 170-180 | 99 | 326 |

These are planning numbers. A sliced run (`run_steps(64)`) adds one full turn per slice, about
8-10 per step at R = 64; it is recorded, not gated. **The gate** stays decision-30's done-when 2,
inside decision-27 §5's range (Q8). If a cell lands above the ΔE = 50 column, the edge is
profiled before J3+J4, which inherits this loop.

### 8. Witnesses and done-when

**Modes.** Every Machine witness runs as in decision-30 §10 (interpreter only; Forced;
Delayed(1), Delayed(3); background at T1 = 1, 3 and 50; Scripted), on x86-64 and ARM64 hosts, plus
a test-only **chain-off** switch (as `links::off()`) so that each compiled mode runs with and
without chains. **Scripted chains:** Scripted(seed) also ends each scripted block with a seeded
`RET_CHAIN` to the resolver's next block, `RET_LINK` or `RET_CONTINUE`, so the dispatcher's
half of the loop meets every end reason after every block and DEFER/STOP at every ordinal of
every chained block, whatever the lowering covers.

**Lockstep** (`machine_lockstep.rs`): J1's full comparison after each `run_steps` (CPU state, all
RAM and ROM, `StepRun`, retired counts), interpreter versus chained versus chain-off, with:

- budgets 1, 2, 7 and 64, and from every chain position each budget 0..=ΣN+1 over chains of 1-12
  blocks, including the twelve-block self-loop and the CALL / RET / JNZ triangle;
- device deadlines in steps and in retired instructions at every position inside a chain;
- every chained exit kind of §3, cross-page edges, a fall-through into a singleton, RET through
  the ring with recursion deeper than its 64 frames, a rewritten return address, a stack switch
  and a `longjmp` past pending returns, RET imm16;
- a callee that toggles A20 and returns, and two vCPUs with A20 on and off sharing units: the
  flags compare must end the chain;
- the delivery log enabled and an NMI handler running: no chained transfer, same results.

**Chains broken mid-run.** A test hook raises `EXIT_REQUESTED`, `NMI_EDGE`, `DISPATCH_DIRTY` or
`CODE_SPACE_EXHAUSTED` at the k-th edge of a chain. The witness asserts that the run stops within
one block of the raise and equals the interpreter run with the same request raised at the
retired count the chained run observed. Own and foreign stores into the chained loop's code, each
with TF clear and set; an IRQ injected between `run_steps` calls; INTR asserted by a handler
effect from a chained block (DEFER, then the boundary).

**Devices** (`machine_j2_devices.rs`): handled MMIO in the first, second and twelfth block of a
chain; positions and transcripts equal to the interpreter's; Fallback, DeviceFault, effects, INTR
and a deadline from a chained block each call the handler once.

**Threads.** Two vCPUs chaining through the same shared slots; one rewrites code the other is
chained through (seen at its next edge); a unit retired and reclaimed while another vCPU chains
(forced through the code-lifetime test hooks): no entry into a unit after its epoch advanced, no
descriptor freed before the chaining vCPU's participation ends; a slot filled by one vCPU and
followed by the other.

**Existing witnesses** under every mode with chains on: decision-30 §10's list, `link_witness`,
`return_witness`, `live.rs`, `legacy_smc_selective` with demotion, `legacy_boundary_events`,
`legacy_delivery_log`, `irq_shadow`, `exit_request`, `retired_count`, `run_steps`.

**Done when:**

1. The witnesses pass in every mode on both hosts. Each compiled witness reports chained
   transfers > 0 and a chain-end census covering every reason (budget, word, epoch, flags, empty
   slot, return miss, unchained exit, DEFER, STOP), so no pass is vacuous.
2. `machine-protected32-bg`, release and perf profiles: register ≤ 102, memory ≤ 133,
   callret ≤ 326 host instructions per step; the edge census and the sliced run are recorded in
   performance.md.
3. No Application cell, no interpreted Machine cell and no J2 device cell is past the two-profile
   perf gate; Application code generation is byte-identical; the frozen ABI and `BlockKey` hash
   tests pass.
4. Production guards still reject Machine IR on Application entries; `x86jit-core` depends only
   on `{iced-x86}`.
5. The embedder census (decision-30 §9) is repeated with chain lengths and end reasons. It is
   not a gate.

## Options considered

| Choice | Options | Recommendation |
|---|---|---|
| Chaining mechanism | (a) host-side Machine chain loop over shared slots (Application's `chain_or_link`, ring, `RET_CHAIN`); (b) in-code tail jumps through the slots; (c) a per-vCPU successor memo in the dispatcher, no generated-code change; (d) patched direct branches | **(a)**: one mechanism with Application; every edge test in Rust, next to J1's. (b) saves perhaps 20-30 host instructions per edge but needs a tail calling convention, fuel, poll, epoch and hook state in generated code; revisit if the edge census says so. (c) fills per vCPU and duplicates `links.rs`'s validation. (d) needs writable code, instruction-cache maintenance and concurrent-modification rules on ARM64 hosts |
| Slot content | (a) a `MachineTarget` descriptor; (b) the bare entry, with N and IR looked up per edge | **(a)**: the edge needs N, flags and IR; one dependent load |
| RET | (a) the shadow ring; (b) RET returns to the turn | **(a)**: callret ≈ 75 against ≈ 175 |
| Chained DEFER | (a) J1's same-turn suffix on the descriptor's IR, after charging the prefix; (b) end the turn at the deferred instruction | **(a)**: (b) would break `Run::decided` (a decided handler result must be consumed before any boundary) |
| Page scope | (a) cross-page unpaged; (b) one page | **(a)** for PG = 0; J3+J4 decides paged |
| Regions (T2) | (a) later, J2c; (b) in J2b | **(a)** |

## Risks

- **Descriptor and slot lifetime.** A use-after-free would be silent. Mitigation: slots are
  zeroed before the epoch advances, the participation guard pins code and descriptors, and the
  retirement-while-chained witness runs under the code-lifetime test hooks.
- **A shared slot wrong for some vCPU.** Mitigation: §2.5 derives the successor key from the
  predecessor's; the flags compare covers ring entries; the A20 two-vCPU witness.
- **Device positions in long chains.** Mitigation: re-arm per edge; transcripts at every chain
  position.
- **Publication latency.** A chain skips the drain; new units publish when it ends. Cost only,
  bounded by slices and exit requests; visible in the background census.
- **The edge costs more than assumed** (call overhead of `call_block_in_mode`, the loop's register
  pressure). Mitigation: the revisit column of §7; option (b) stays open.
- **Ordering on ARM64.** The descriptor is published with Release and the slot read with acquire;
  a relaxed read would let a weakly ordered host see the pointer before its fields, the IBTC
  lesson (`cache.rs:1115-1124`).
- **J3+J4 inheritance.** Paged chains need the fetch-translation rule of §3; decision-29's "TLB
  token change inside a block stops it" extends to edges.

## Maintainer questions

| # | Question | Options | Recommendation |
|---|---|---|---|
| Q1 | Chaining mechanism (§2) | (a) host-side chain loop over shared slots with target descriptors; (b) in-code tail jumps; (c) per-vCPU successor memo; (d) patched branches | **(a)** |
| Q2 | Chain near RET through Application's shadow ring? | (a) yes; (b) RET returns to the turn | **(a)**: the ring only gates; a hit needs the popped EIP |
| Q3 | Chain across pages in unpaged Flat32, amending decision-27's and decision-30's "inside one page first"? | (a) yes, J3+J4 decides paged; (b) one page | **(a)**: without paging no translation can change inside a chain |
| Q4 | Machine code returns `RET_CHAIN`, `RET_LINK` and `RET_RETURN_MISS`, amending decision-29 Q7? | (a) reuse, meaning "all N completed"; (b) new Machine codes 19-21 | **(a)**: same meaning, shared emitters, no ABI growth |
| Q5 | Machine regions (T2), which decision-30 §6 put in J2b | (a) later, as J2c, if the census shows tight loops whose edge dominates; (b) in J2b | **(a)**: chaining alone is expected 3-5× under the gates |
| Q6 | No chain under the cold epilogue (delivery log, NMI blocked, A20 pending)? | (a) yes; (b) chain under the delivery log too | **(a)**: cost only, and keeps `record_unit` per unit |
| Q7 | The edge's counting lives in the chain loop, and a chained DEFER runs J1's same-turn suffix on the descriptor's IR? | (a) yes; (b) end the turn at the deferred instruction | **(a)**: generated code still stores only the ordinal; `Run::decided` stays safe |
| Q8 | Gate | (a) decision-30 done-when 2 binding, the ΔE = 50 column a revisit trigger; (b) tighten the gate to the estimates | **(a)** |
| Q9 | Indirect JMP/CALL | (a) unchained; an IBTC lowering in a later bead if the embedder census shows them; (b) IBTC in J2b | **(a)** |

## Consequences

- Compiled Machine code reaches the Application JIT's per-block cost on the same bytes, within
  ΔE per edge, and the dispatcher turn becomes the cold path it is for Application.
- J3+J4 inherits the chain loop, the descriptors and the ring, and adds the paged edge rule.
- Revisit if a cell exceeds §7's ΔE = 50 column (profile the edge; option (b)), if the embedder
  census shows chains ending mostly on the word poll (find the writer), or if indirect transfers
  dominate its turns (Q9).

## Implementation notes (2026-11-27, proposed record diff)

Recorded while implementing J2b (bead xj-to9.54). None changes an answered question.

1. **Where the ring lives (§2.1, §6).** J2 gave `MemCtx::ret_stack` its Machine meaning, the
   compiled TLB view (decision-30 implementation note 5). The vCPU's Machine shadow ring is a
   field of that view (`MachineTlb::ring`, at `jit_abi::machine::RING_OFFSET`), so Machine
   CALL and RET reach it through `ret_stack` plus a constant: no `MemCtx` field, the frozen
   offsets unchanged. The ring is emptied when the participation epoch it was used in
   changes, and when the vCPU follows another code lifetime.
2. **Which CALL and RET.** Machine IR has no `IrOp::Call` or `IrOp::Ret`: near CALL and RET
   are a push or a load, the ESP write and a `Jump`. `jit_abi::machine::machine_exit`
   classifies the block's last instruction: a stored immediate equal to the next
   instruction's offset with an ESP write is a CALL (direct or indirect: both push the ring,
   only a direct one chains), and an indirect `Jump` of a block whose `IrBlock::link_exit`
   flag holds is a near RET: the flag holds for a direct exit or a near RET, so with an
   indirect jump only a near RET sets it. This bears on cost only.
3. **The fill condition (§2.4).** A `RET_LINK` leaves its fill in the vCPU's Machine state
   with the vCPU's retired count after the chain. The next compiled entry takes it, and
   publishes it only when the retired count is unchanged, the block starts at the target and
   its key flags equal the fill's; the epoch is tested under the lifetime's lock
   (`publish_entry_if_current`). A turn that delivered an event or ran a service step moves
   the IP or nothing that the successor key depends on (§2.5); each run starts without one.
4. **Descriptors.** `code_lifetime::MachineTarget` holds the entry, the block, its count and
   its key flags; the block's `link_exit` is read from the block. Descriptors are retired with
   the IBTC descriptors when slots are cleared and collected by the same grace rule.
5. **The chain's result.** `machine_progress` returns the last block's result with the chain
   prefix and the last block; the dispatcher charges the prefix and goes on with that block
   as the turn's block. A chained DEFER's suffix runs inside `machine_progress`, after the
   guard, with the device hook pointed at a clone of the last block (`InBlock::advance`), so
   `Run::decided` is consumed before any boundary (Q7).
6. **Census and test hooks.** `MachineJitCounters` gains `chained` and `chain_ends` (nine
   reasons, `CHAIN_END_NAMES`), counted in test and test-hooks builds only; a `RET_CHAIN`
   with no chain started (chains off, or the cold epilogue) counts as `unchained`. The
   vCPU's `chained` counter, flushed into `TranslationCache::chained`, now counts Machine
   chain transfers too, in every build, beside the Application block links. The chain-off
   switch is `Vm::set_machine_chains_for_test`, and `Vcpu::raise_at_chain_edge_for_test`
   raises an exit request, an NMI edge, a code write, a retirement or exhausted code space at
   a chosen edge (`ChainEdgeAction`). Scripted chains end each completed scripted block with a
   seeded chain to the block the resolver returns, a link or a continue.
