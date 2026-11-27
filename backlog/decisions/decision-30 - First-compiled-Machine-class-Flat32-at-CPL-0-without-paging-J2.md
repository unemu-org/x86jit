---
id: decision-30
title: 'First compiled Machine class: Flat32 at CPL 0 without paging (J2)'
date: '2026-11-27'
status: accepted
---

Decider: maintainer (2026-11-27), accepting every recommended answer (Q1-Q10).
Drafted 2026-11-27 for bead xj-to9.2, stage J2 of decision-23 §8. It implements decision-27's
amendment of 2026-11-26 (§§1-3, §5, §6 J2 row) on top of decision-29 (J1, landed in 2816c7c2),
and decision-28 stage 4 for the Flat32 class. It changes two accepted texts, each listed as a
question: the order of decision-27 Q9 (the code-tag bit from the start instead of the
watermark, Q2) and, for unpaged code, the shape of the TLB probe of decision-27 §3.2 (a
direct-mapped compiled view of the same TLB, Q3). Everything else in decisions 20, 23, 27, 28
and 29 stands. No code changes with this record.

# First compiled Machine class: Flat32 at CPL 0 without paging (J2)

## Context

J1 made compiled Machine blocks exact and kept them behind a test-only admission. J2 turns
them on for embedders, for one class: PE set, PG clear, CPL 0, CS.D = SS.B = 1, and
CS/DS/ES/SS ordinary expand-up caches with base 0 and limit 0xFFFFFFFF
(`legacy::flat32_cache`, `legacy.rs:433`). J2 is the first stage that has to be fast.

### What J1 left (main 2816c7c2)

| Piece | Where | State |
|---|---|---|
| Entry contract E1-E8, charge at exit | `vm/legacy/compiled.rs:174-326` | Production-ready. J2 keeps it |
| Return codes, `machine_ordinal`/`machine_value` | `jit_abi.rs:84-92`, `:232-236` | Kept |
| Units | `compiled.rs:90-157`: a per-vCPU `EngineMap<usize, Unit>` keyed by `Arc<IrBlock>` address, looked up every turn | Replaced by the shared Machine map (§2) |
| Compile | `Backend::materialize_machine` (`vm.rs:388-405`), synchronous, `Shared::compile_machine` (`x86jit-cranelift/src/lib.rs:4336`) | Kept for the Forced test mode; production goes through the worker |
| Lowered set | `interp/legacy/machine_jit.rs:11-73` | Widened (§3) |
| Memory | Every `Load`/`Store` calls `machine_load`/`machine_store` (`jit_abi/machine.rs:60-123`), which runs the checked access `machine_jit::access` (`machine_jit.rs:108-177`) | The miss path behind an inline probe (§4) |
| Exits | `RET_CONTINUE` and `STOP` take the outcome arm (`vm/legacy.rs:3969`) | Fast arm (§6) |
| Admission | `MachineJitMode` Off / Forced / Delayed(k) / Scripted(seed), test-hooks only (`compiled.rs:19-57`) | Production admission added (§2); the test modes stay |

### Measured, and what it implies

J1, `machine32_loops`, release, host instructions per guest step, interpreted / forced
compiled (2816c7c2 commit record; a supervisor re-run gave 177 / 285 / 1113):

| Workload | Interpreted | J1 forced | Compat32 JIT (decision-27 §5) |
|---|---:|---:|---:|
| register | 429 | 170 | 18.5 |
| memory | 571 | 278 | 41.9 |
| callret | 1063 | 1091 (near CALL and RET defer) | 69.0 |

**DERIVED, not measured:** the register loop runs one five-instruction block per turn
(t = 0.2) with no memory access. If the compiled body costs about what the Compat32 JIT's
does (18.5), a compiled Machine turn costs about (170 − 18.5) / 0.2 ≈ **760 host
instructions**. Decision-27 §5 assumed 150-400. The memory loop then puts one J1 helper access
at about (278 − 0.2·760 − 30) / 0.4 ≈ **240**, against an assumed 10-20 for an inline probe.
Those are the two numbers J2 has to move. With the J1 turn, no lowering of the body reaches
decision-27's range for register or callret: the register loop would stay at about 170.

## Decision

### 1. Principle

J2 changes how fast a compiled block produces the walk's `Progress`, not what it produces.
Decision-29's contract (E1-E8, test at entry, charge at exit, DEFER at an instruction start
with nothing of it committed, STOP after an own code write) is unchanged. Each widening in
§3-§5 is a new way to complete an instruction in generated code, and each is exact for the
same reason J1 is: an instruction is either completed in full or deferred before its first
commit, and the interpreter re-executes a deferred one from the state before it (SDM Vol 3A
§7.5 p. 7-4; 80386 PRM §9.1 p. 9-2).

### 2. Turning it on: admission, key, map, worker, threshold, demotion

**Admission (decision-27 Q6).** A Machine VM compiles Flat32 blocks when all hold:

1. `Vm::set_tier_up_after(Some(T1))` and `Vm::set_tier_up_background(true)` (`vm.rs:1185`,
   `:1198`). No new knob. With background off, Machine code stays interpreted: decision-27
   §2.2 forbids an inline compile on the vCPU in production. The synchronous switch point
   witnesses need stays `MachineJitMode::Forced`, test-hooks only.
2. The backend says it compiles Machine IR: a new `Backend::compiles_machine(&self) -> bool`,
   default `false`, `true` for `JitBackend`. That is additive. A wrapping or external backend
   that asserts `legacy_context.is_none()` in `tier_up_async` is then never handed Machine IR.
3. The VM's memory does not recover host faults (`ROUTED == false`). Inline accesses (§4)
   would need the Application JIT's fault landing pads, and J2 does not build them for Machine
   code (Q7). A routed VM stays interpreted, as J1's store helper already does for stores.
4. Decision-29 E1-E8 at every entry, plus J2's class test (E2): `Flat32`, `!paging_enabled`,
   CPL 0, `BlockWalk::Direct`.

The production guards stay: `vm.rs:199`, `:218`, `:241`, `:262`, `:524`, `:4666` and
`x86jit-cranelift/src/lib.rs:4223`, `:4296` keep rejecting legacy IR on every Application
entry. Machine IR reaches the backend only through the Machine request below and through
`materialize_machine` (Forced).

**Key.** `MachineBlockKey` exactly as decision-27 §1 defines it: `physical_pc` (post-A20),
`decode_ip`, `model_semantics_id`, `flags` (regime, CS.D, SS.B, address bits, segment class,
CPL, `io_permitted`, PG, A20, CR0.AM), `mxcsr_mode`. CR0.AM joins the flags because the lift
reads the alignment mask (decision-20 amendment item 2). It is constant at CPL 0, where
`I486Gate` is never emitted (`lift/legacy.rs:1587-1593`), but the key names every lift input
so that J3+J4 inherits it unchanged. In J2 `mxcsr_mode` is always 0 (E7).

**Shared map (exception E3).** `TranslationCache` gains `machine: RwLock<HashMap<MachineBlockKey,
CompiledPtr>>` beside `map` (`cache.rs:695`). Application's `BlockKey`, its hash stream and
`DispatchMap` are untouched. The source and pending indexes become generic: `SourceIndex<K>`
(`cache.rs:423`) gets a second instance for Machine units under the same `spans` lock, and so
do `pending` and `interpreter_only` (`cache.rs:436-444`). `invalidate_range` (`:1858`),
`absorb_written` (`:1741`) and the unmap and Trap-map flushes (`vm.rs:1617-1700`) query both.
The Machine instance's empty-root test short-circuits, so an Application VM pays one branch
per invalidation (Q10). J1's per-vCPU units map (`compiled.rs:90-157`) is removed.

**Private proofs hold the pointer (decision-27 §3.1).** `flat32::Proof` (`flat32.rs:69`) and
`links::Link` (`links.rs:95`) gain `hot: u32` and `compiled: Option<(CompiledPtr, u64 /*epoch*/)>`.
The resolve hit (`flat32.rs:140-147`, `get` becomes `get_mut`) and `follow_link` (`links.rs:324`)
bump `hot` and return the slot by value with the block. A turn that holds a slot whose epoch
equals `cache.epoch()` enters it with no lookup. On a mismatch, it looks the key up once in
`machine` (cold) and refreshes or clears the slot. Hotness lives with the proof (exception E2),
and a dropped proof restarts its count ("stable", decision-27 §2.1 item 2).

**Request and publish: the Application path, parameterized.**

- At `hot == T1` the turn computes the key and looks in `machine` first. Another vCPU may
  already have published that unit. If the unit is absent, `try_begin_tier_up` claims a
  Machine pending slot.
- `prepare_tier_up` registers the sources: the block's physical span minus each
  `LegacyOp::LiveField`'s `[bus, bus + bytes)` (decision-27 §3.1). The request carries **the
  proof generation that this turn's proof map reconciled to** (`flat32.rs` `reconcile`), not
  the current one. `prepare_tier_up` and `publish_tier_up` (`cache.rs:1530`, `:1557`) take
  the generation source as a parameter: `publication_generation()` for Application,
  `proof_generation()` for Machine (decision-27 §2.4). An absorbed live-field patch in the
  claim window then cancels nothing, and any other code write does.
- The request is an ordinary `TierUpRequest` with `TierUpUnit::Block(ir)` whose
  `legacy_context` is `Some`. `pc` is `physical_pc`, `epoch` is the request id. No public type
  changes. The Cranelift worker's `define_request` (`lib.rs:5073`) routes Machine IR to the
  Machine translator. The access configuration (physical mask from `context.generation`, order
  from `consistency`) is derived from the request.
- `Busy` and `Unsupported` mean "stay interpreted", and the proof keeps counting. A finished
  compile is drained at the top of the Machine turn, after the budget test (`vm/legacy.rs:3389`),
  under `CriticalSectionKind::TierUpWait` and `Publish`, as `drain_tier_up` does (`vm.rs:4053`).
  The drain maps the echoed request id to its Machine key, so `TierUpFinished` needs no new
  field. Publication **inserts** into `machine` (E3). It is rejected exactly as Application's
  is: a superseded id, a cancelled source, a generation moved before registration, or a
  code-lifetime change. A deterministic compile failure enters the Machine `interpreter_only`
  set until the bytes change.
- The draining vCPU stores the pointer into its own proof and link. Other vCPUs see the unit at
  their own T1, or through `cache.machine_published()`, a counter that a proof in the
  "requested" state compares at each hit (one load) before it does a cold lookup.

**Threshold.** T1 is the embedder's `tier_up_after`. Recommended: 50, as `x86jit-run` ships.
No Machine regions (T2) in J2: a region skips turns, as chaining does, so it belongs to J2b.

**Demotion (decision-27 §2.5, Q7 accepted).** The 16-entry `smc_pages` hint (`vm/legacy.rs:796`)
becomes `{page, compiled_drops, window_start, demoted_until}`, counted in this vCPU's retired
instructions. A selective drop of a proof that held a compiled slot counts one drop for its
page. Two drops within 2^20 retired instructions stop T1 requests for that page until 2^20
retired instructions pass with no drained code write to it. This affects cost only.

**Test modes.** `Forced` compiles synchronously at T1 = 1 through `materialize_machine` and
inserts into `machine` with the same generation check. `Delayed(k)` becomes background
compilation whose drain a test hook holds for k turns. `Scripted` is unchanged. All three stay
behind test-hooks.

### 3. What J2 compiles

The classifier (`machine_jit.rs:11-73`) keeps J1's form: an instruction is compiled if every
op is lowered and the **compiled-fallibility rule** holds. Otherwise it is an unconditional
DEFER at its start. The rule replaces J1's conditions 2-3:

> Counting as fallible only the ops whose compiled lowering can defer (`Load`, `Store`,
> `Flat32Address`, `Div`), no commit (register write, flag write, store) precedes a fallible
> op, and the instruction has at most one store.

`Legacy(Checkpoint)` is then admitted. The lifter inserts it whenever a commit precedes any op
it counts as faultable (`rollback_required`, `lift/legacy.rs:1060-1117`), and it counts `Jump`
and `Branch`. In Flat32 compiled code neither can fail: targets are masked to the operand size
and the CS limit is 0xFFFFFFFF, so no target lies beyond it (SDM Vol 2A JMP p. 3-506 and
LOOP/LOOPcc p. 3-571, "#GP(0) if the offset ... is beyond the limits of the CS segment"). The
lifter stays unchanged (Q5): the interpreter keeps its checkpoint, and compiled code needs none
because nothing fallible follows a commit.

| Instruction family | IR in Flat32 (lifter) | J2 | How it stays exact |
|---|---|---|---|
| Near CALL rel/r/m | `Flat32Address(CS, Execute)`, push (`Flat32Address(SS)`, `Store`), `WriteReg ESP`, `Jump` with `Checkpoint` (`lift/legacy.rs:2080-2090`, `:1464`) | **compiled** | The target check and the stack check precede the store. A push whose dword would pass FFFFFFFFh (ESP 1-3) defers at `Flat32Address`, and the interpreter raises the stack fault. Nothing is committed before the store |
| Near RET / RET imm16 | `Flat32Address(SS)`, `Load`, `Flat32Address(CS)`, `WriteReg ESP`, `Jump` (`:2104-2116`) | **compiled** | Load and checks before the only commit |
| PUSH/POP reg, imm, mem; LEAVE | generic stack ops | already compiled by J1 (PUSH/POP r) / **compiled** (mem forms) | One store, last fallible op |
| LOOP/LOOPE/LOOPNE (`L::Loop`) | count decrement and branch, `check_target` | **compiled** | Infallible in Flat32 (above); `count_bits` and the ZF test as in the interpreter's op |
| DIV/IDIV (`IrOp::Div`) | operands, `Div`, `WriteReg` | **compiled** | The shared `div` helper (`codegen/mod.rs:3076`) reports #DE; the Machine lowering takes a HELPER_DEFER there instead of `RET_EXCEPTION`, and the interpreter raises #DE. CF, OF, SF, ZF, AF and PF are undefined after DIV (SDM Vol 2A DIV "Flags Affected" p. 3-265): the lockstep must show the lowering leaves the interpreter's values, the SHLD lesson of decision-29 |
| `LegacyOp::LiveField` | one per live immediate or displacement | **compiled** (§5) | A load, never baked |
| `LegacyOp::ReadSelector` (MOV r, Sreg) | | **compiled** | A load of the `CpuState` selector, read live (decision-20 amendment) |
| String ops, REP | `L::String`, a singleton block | defer (interpreted) | Variable units; decision-29 keeps the iteration ledger in the interpreter |
| x87 (`L::Coprocessor`), FWAIT | singleton | defer | FPU axis, numeric-error routing; a later stage (Q4) |
| PUSHA/POPA, ENTER, PUSHF/POPF, CLI/STI, IN/OUT, INT, IRET, far transfers, control and descriptor ops | not ordinary or multi-store | defer | As J1 |
| Segment loads, FS/GS operands (`L::Address`), CS writes | live checks | defer | Context changes stay the interpreter's (`reload_continues`) |
| LOCK, XCHG with memory, CMPXCHG, XADD | atomic ops | defer | Cross-vCPU atomicity and `LockedTransfer` device rules |
| 16-bit SHLD/SHRD | | defer | As J1 |

The alignment check needs nothing at CPL 0: #AC requires CPL 3 with CR0.AM and EFLAGS.AC
(SDM Vol 3A §7.15, Event 17, p. 7-53; `alignment_regime`, `interp/legacy.rs:216`).

### 4. Memory accesses in unpaged Flat32 code

**The compiled view of the TLB.** The vCPU's `Tlb` (`interp/legacy/paging.rs:397`) gains
`fast: [FastEntry; 256]`, `repr(C)`, with `MemCtx::machine_tlb` (appended) pointing at it:

```text
FastEntry { read_tag: u32, write_tag: u32, addend: u64, frame: u32, _pad: u32 }   // 24 bytes
index = (linear >> 12) & 255; a tag is linear page + 1 (0 = invalid)
```

It is a projection of that one TLB, not a second one (decision-23 §8). It is written only by
`Tlb::insert`, `remove` and `invalidate`, and, with PG = 0, by an identity fill. `reconcile`
(`:423-437`) already invalidates on CR0.PG/PE, A20 and the model's flags mask, so a context
change empties it. A changed `Memory::mapping_identity` empties it at the turn's proof
reconciliation, as `flat32.rs:128` does for proofs. It is direct-mapped on the low page bits
instead of the interpreter's folded 64-slot index (`:415-420`): the fold protects fetch
entries' generations, which identity entries do not have, and 256 slots bound conflict misses
on unpaged data sets of 1 MiB per vCPU at 6 KiB (Q3).

**Identity fill (PG = 0).** On a miss the helper fills the slot for the linear page:
physical = page masked by `physical_mask(generation)` and A20 (the 1 MiB wrap with A20 off,
the 386SX 24-bit bus). It sets `read_tag` for RAM and ROM, and `write_tag` for RAM only.
`addend` is host base − guest base. The fill happens only when the whole frame is one region
inside the backing and the memory model is not SoftMmu. Trap pages, holes, mixed frames and
SoftMmu fill nothing, so every access to them takes the helper.

**Inline load (size s, linear a, after `Flat32Address`):** page = a >> 12; e = fast[page & 255];
if (a & 0xfff) > 0x1000 − s or e.read_tag ≠ page + 1 → helper; else value = load(e.addend + a)
with the consistency tier's ordering, as the shared `emit_load`. Assumed 9-11 host instructions.

**Inline store:** the same probe on `write_tag`, then the raw store with the tier's ordering,
then the store barrier of the shared `note_store` (`codegen/mod.rs:3782-3910`) with one change.
Its code gate tests the frame's live code-tag bit instead of the `code_range` watermark (Q2):
`word = code_page[(frame − guest_base page) >> 6]`, `bit = frame & 63`, through an appended
`MemCtx::code_tags_ptr`. The watch gate (live `watch_count`, then the page bit, the helper on
a page straddle) is the Application code unchanged, so `watch_range` sees compiled Machine
stores. A set tag bit calls the cold `machine_note_store` helper, which runs
`Memory::note_write` with the interpreter's observer: alias expansion, `absorb_at_store` for
live fields (`live.rs:157`), and the queued dirty range. It returns `ACCESS_STOP` when the
vCPU's dispatch word is then nonzero, as `machine_store` does. Assumed 16-20 host instructions
when untagged and unwatched.

**Why the tag bit and not the watermark.** The watermark is one range over 64 KiB granules
(`memory.rs:1986-2010`), and it only widens. A 386 or later processor fetches its first
instruction at physical FFFFFFF0H (SDM Vol 3A §12.1.4 p. 12-5) and keeps fetching from the top
64 KiB until its first far transfer (80386 PRM §10.2.3 p. 10-3). On a PC-class board that boots
from reset, the reset page is tagged, and the watermark then spans from the lowest code page
to 4 GiB for the rest of the VM's life: every compiled store calls the note helper.
Decision-27 §3.3 named exactly this case as the trigger for its fallback, and the trigger holds
by construction, so J2 starts with the fallback. The bit is read live, so a page tagged by a
compile in flight on another thread is seen by the next store, as the watermark is (decision-27
Q9's objection to option (c) does not apply).

**The miss helper** is J1's `machine_jit::access` (`machine_jit.rs:108`), plus the fill:

| Access | Inline | Helper |
|---|---|---|
| RAM, one frame | yes | fills on a miss, completes |
| ROM read | yes | fills on a miss |
| ROM write | no write tag | DEFER; the interpreter discards the write (`RegionKind::Rom`) |
| Crosses a page | no | completes when every part is RAM (ROM for reads), as J1; a store split across bus parts DEFERs (decision-29) |
| Trap with an in-process handler for this run | no | the decision-28 in-block contract (below) |
| Trap otherwise, hole | no | DEFER; the interpreter's exit or fault |
| Code-tagged RAM | yes, then the note helper | — |
| Operand reaching past 4 GiB | `Flat32Address` DEFERs first | — |

**In-process MMIO (decision-28 stage 4).** For a Trap access, the helper calls the existing
`device::in_block_access` (`vm/device.rs:1161`) with the baked instruction IP. The dispatcher
arms the `InBlock` hook (`device.rs:955`) around the compiled call with the proof's `ir`, as it
arms the interpreted walk, extended from real-mode shared walks to compiled Flat32 calls, with
`linear_address = Some(a)`. The device position is the hook's `retired + k`, where k is the
ordinal of that IP (`sole_access`). A debug assertion checks it against the baked ordinal.
`Handled` with no effects continues the block. Anything else returns `None`, and the result
stays in `Run::decided`. The helper DEFERs, and the interpreter's re-execution issues the
ordinary handoff from `decided`: the handler is never called twice for one transfer. A
handler that asserts INTR, moves a deadline or writes RAM has effects, so it always takes this
path. No boundary therefore becomes due inside a compiled block (decision-29 §5).

### 5. Live fields in compiled code

`LegacyOp::LiveField { dst, bus, bytes, extend }` lowers to one unaligned load of `bytes` from
host base + (`bus` − guest base), then the decoder's extension (`legacy::live_field_value`).
It is read once per execution, where the op stands, which is before the instruction's own
stores, as the interpreter reads it (`live.rs` module comment). `bus` lies in `physical_pc`'s
frame (no straddles), and that frame is RAM or ROM. A remap of that frame invalidates the
unit, because `Vm::map` (Trap) and `unmap` invalidate overlapping coverage (`vm.rs:1617-1700`),
and the unit's coverage includes the frame's opcode bytes. The unit registers its sources minus
its live fields (§2), so `absorb_written` still absorbs a store wholly inside a field: the unit,
the proof and the link all survive. A unit compiled from a lift that baked a field covers that
field's bytes, so a patch there is not absorbed and drops the unit, which is exact. Units under
one key are therefore interchangeable whichever lift they came from.

### 6. The turn, links and chaining

At the J1 cost (≈ 760, derived above), the turn dominates every compiled workload. J2 makes the
compiled linked turn cheaper and leaves chaining to J2b:

1. **No units lookup.** The slot comes from the proof or the link (§2). The epoch test is one
   load, against J1's hash lookup, `Option` take and put of the participant, and unit
   bookkeeping per turn.
2. **Fast arm.** `RET_CONTINUE` and `RET_MACHINE_STOP` return `outcome: None` with the
   counts and take the fast arm (`vm/legacy.rs:3912`), which refreshes the boundary only when
   it must (`:3924`). J1's outcome arm recomputes `machine_boundary` (`:3016`) on every turn.
   A STOP still links nothing.
3. **Links carry the slot.** A link made from a proof copies its slot. Publication updates the
   link slot for that IP when it names the same block. A linked turn enters compiled code
   without the resolver (decision-27 §3.1).
4. **Census first.** Before the lowering work, J2 measures the compiled linked turn with
   throwaway switches, as the ceiling study did (performance.md, "Interpreter ceiling study"):
   epoch and participation, eligibility, link validation, the SMC and exit-word polls, device
   bookkeeping, time stats, and the merge. That census replaces the derived 760.

**Chaining (J2b, decision-27 Q4).** Compiled-to-compiled transfers inside one page, with J1's
budget test at each chained entry and a dispatch-word poll at each chain edge, as Application's
chain edges poll. They end on any link-invalidating event, and Machine regions (T2) come with
them. J2b starts if the census shows that a linked compiled turn cannot get below 400 host
instructions, or if callret misses its range after J2 (Q8).

### 7. Reduced identity and 256 contexts

The unpaged Flat32 identity names only what the lift reads (decision-20 amendment item 2), and
the compiled key is that set's projection plus `physical_pc`, `decode_ip` and `mxcsr_mode`.
Two contexts that differ only in selectors, GDTR, IDTR, LDTR, TR, CR3, IOPL or NT share a
proof, and so a unit. That is exact because compiled code reads none of them. `ReadSelector` is
a live load, and every IOPL-, descriptor- or task-sensitive instruction defers. Each of the 256
retained contexts holds its own proofs and slots. An evicted context's slots go with its
proofs, and since context ids are not reused before a translation flush, no slot is reached
under another context. `invalidate_translations` drops private state only: units are physical
and survive it. A compiled block changes no context (decision-29 §7), so the 256-context
retention and its least-recently-used order see the same sequence of contexts in both tiers.

### 8. Code that changes in J2

| Place | Change |
|---|---|
| `cache.rs:423-460`, `:695`, `:1441-1620`, `:1741`, `:1858` | `MachineBlockKey`, the `machine` map, generic `SourceIndex<K>` with a Machine instance, Machine `pending` and `interpreter_only`, generation source as a parameter, insert-publish, `machine_published()` |
| `vm.rs:300-470` | `Backend::compiles_machine` (default false). `TierUpRequest` / `TierUpFinished` unchanged |
| `vm/legacy.rs:3184-3191`, `:3389`, `:3612`, `:3750-3762`, `:3912-3933` | Machine drain after the budget test; the slot from resolve or link; T1 request; `machine_progress` without a units map; CONTINUE and STOP on the fast arm |
| `vm/legacy/compiled.rs` | Units map removed; production admission (§2 items 1-4); test modes on the shared map |
| `vm/legacy/flat32.rs:69`, `:140-147`; `links.rs:95`, `:324` | `hot` and the compiled slot; bump on hit and follow |
| `vm/legacy.rs:796` | The SMC-hot list with demotion |
| `interp/legacy/paging.rs:383-500` | `Tlb::fast`, identity fill, invalidation of the fast slot in `insert`/`remove`/`invalidate` |
| `jit_abi.rs:103-236` | Appended `machine_tlb`, `code_tags_ptr`; the frozen offsets test (`:1066-1228`) still passes |
| `jit_abi/machine.rs`, `interp/legacy/machine_jit.rs:11-177` | Classifier rule (§3), the fill in the miss helper, `machine_note_store`, the in-block device call |
| `vm/device.rs:955-1240` | `InBlock` armed around compiled Flat32 calls, with a linear address |
| `x86jit-cranelift/src/codegen/machine.rs:130-233`, `codegen/mod.rs:3782-3910` | Inline probe and store barrier with the tag-bit gate; `Div`, `Loop`, `LiveField`, `ReadSelector`, `Checkpoint` lowerings |
| `x86jit-cranelift/src/lib.rs:4336`, `:5073`, `:5354` | Worker routes Machine requests to the Machine translator; `compiles_machine` |
| `x86jit-bench/examples/machine32_loops.rs` | A `machine-protected32-bg` path (T1 = 50, background), beside J1's forced path |

### 9. Targets

Model (decision-27 §5): per step = J + t·T + a·ΔP, where J is the Compat32 JIT on the same
bytes, t is turns per step (0.2 / 0.2 / 0.6), a is accesses per step (0 / 0.4 / 0.4), and ΔP is
the probe's cost over Application's inline access, **ASSUMED** 8-15. T is the census result,
**ASSUMED 250-400** after §6.

| Workload | J1 forced | J2 expected (T = 250-400) | Decision-27 §5 range | If T stays ≈ 760 |
|---|---:|---:|---:|---:|
| register | 170 | 69-99 | 50-102 | ≈ 170 |
| memory | 278 | 95-128 | 77-133 | ≈ 200 |
| callret | 1091 | 222-315 | 166-326 | ≈ 530 |

These are planning numbers. They multiply no upper bounds, and the census replaces T.

**The embedder's workload.** A PC-class embedder's 486-33 game already runs at about 150 % of
its guest clock interpreted (supplied, after the decision-20 amendment). J2 is not needed to
make it playable. Its value is headroom: host time per guest second falls for its Flat32 CPL-0
code, and that buys margin for device emulation, lower host load, or a faster guest clock. It
is also the first compiled class on the way to the late-1990s 32-bit targets, which need
several times a 486's instruction rate. Which of those targets run flat at CPL 0 without paging
depends on their DOS extender and its host (raw, XMS, VCPI or DPMI). Those running at CPL 3 or
with paging need J3+J4, and J2 does not guess which. J2's embedder step is a **census, not a
gate**: run the game in a test-hooks copy of the embedder's tree (as the ceiling study did) and
record the share of retired instructions completed in compiled code, the defer census by
cause, the shares of strings, x87 and port singletons, host instructions per retired
instruction before and after, and turns per step. The next lowering is chosen from that
census.

### 10. Witnesses and done-when

**Modes.** Every Machine witness runs in: (1) interpreter only; (2) Forced, synchronous on the
shared map; (3) Delayed(1), Delayed(3), background with the drain held; (4) **background
production configuration**, T1 = 1 and 3 with the real worker, whose publication timing varies
from run to run while results must not; (5) Scripted. All five run on x86-64 and ARM64 hosts.

**Lockstep.** `machine_j1_lockstep.rs` grows into `machine_lockstep.rs`. It keeps J1's full
comparison at every `run_steps` call (CPU state, all RAM and ROM, `StepRun`, retired counts) and
J1's budget schedule, and adds:

- each newly lowered family (near CALL/RET direct, register- and memory-indirect, RET imm16,
  PUSH/POP mem, LOOP family, DIV/IDIV with seeded operands including #DE and the undefined
  flags, `ReadSelector`), each asserted to run with no defer;
- #SS from a push at ESP 1-3 and a pop at ESP FFFFFFFDh (ESP 0 pushes legally at FFFFFFFCh),
  #GP at the 4 GiB end, #DE, each at two ordinals;
- the fast path: RAM, ROM read and write, Trap, hole, page-crossing loads and stores, a
  RAM/ROM split, A20 off across the 1 MiB wrap, the 24-bit bus, more than 256 data pages
  (conflict misses), and a map or unmap between runs;
- stores into code: a tagged page, a page of a declared alias view, the running block's own
  next instruction, another block, and an absorbed live-field store, each with TF clear and
  set. A compiled store must STOP or invalidate exactly as the interpreted one does;
- `watch_range`: compiled stores to watched and unwatched pages and straddles report the same
  dirty ranges as the interpreter's (`take_dirty_ranges`);
- the refusals: a routed VM, background off, a backend without `compiles_machine`, each
  asserted to enter no unit.

**Threads** (the J2 items decision-29 moved here): Machine analogues of
`threads/same_block_smc.rs` and `threads/bg_tier.rs`. Two vCPUs share units. A compile races a
patch of its source, a live-field promotion over it, and a code-lifetime reset; the publication
must be rejected or the unit invalidated, and both vCPUs' results must equal the interpreter's
with the same cross-vCPU schedule. A store by one vCPU into code another runs compiled is seen
at its next turn.

**Devices.** `vm/device/tests.rs`'s decision-28 census in modes 2-4 with in-process handlers.
Handled MMIO completes inside compiled blocks, with positions `retired + k` equal to the
interpreter's. The device transcripts (requests, order, positions) are identical in every mode.
Fallback, DeviceFault, effects, INTR and a deadline each call the handler once, and the turn
then sees the same boundary.

**Existing witnesses** under modes 2-5: the J1 list (decision-29 §10), plus `link_witness`,
`return_witness`, `live.rs`, `legacy_code_alias_smc`, `legacy_smc_selective` (with
demotion: a page dropped twice stops compiling and then recovers, with results unchanged), and
the three interpreter cost probes in Off.

**Done when:**

1. The witnesses pass in every mode on both hosts. Each forced or background witness reports
   compiled-retired instructions > 0 and a defer census covering every cause row of decision-29
   §4 plus J2's new causes, so no pass is vacuous.
2. `machine32_loops` `machine-protected32-bg` (T1 = 50, background), release and perf profiles,
   is inside decision-27 §5's range: register ≤ 102, memory ≤ 133, callret ≤ 326 host
   instructions per step. The census of §6 is recorded in performance.md.
3. With compilation on, no Machine cell that does not compile (real16, paged, Segmented, V86)
   is past the two-profile perf gate. With it off, no Machine cell is. No Application cell is,
   and the `BlockKey` hash and frozen ABI tests pass.
4. Production guards reject Machine IR on every Application entry, and `x86jit-core`'s
   dependencies stay `{iced-x86}`.
5. The embedder census (§9) is recorded. It is not a pass/fail gate.

## Options considered

| Choice | Options | Recommendation |
|---|---|---|
| Enablement | (a) `tier_up_after` + background, as decision-27 Q6; (b) a new Machine switch; (c) on by default | **(a)**, plus the backend capability bit |
| Units | (a) shared `machine` map and slots in proofs and links; (b) keep J1's per-vCPU map | **(a)**: decision-27 §3.1; (b) costs a hash per turn and compiles once per vCPU |
| Index structures | (a) generic `SourceIndex<K>`, a second instance; (b) one enum unit key | **(a)**: Application types and hashes untouched |
| Request API | (a) `TierUpUnit::Block` with Machine IR, gated by `compiles_machine`; (b) a new unit variant or fields | **(a)**: no public type changes |
| SMC test on compiled stores | (a) watermark; (b) live tag bit from the start | **(b)**: the reset vector makes (a) near-full on every 386+ PC-class board |
| Unpaged lowering | (a) the TLB's compiled view, direct-mapped, 256 slots; (b) the interpreter's 64-slot folded index; (c) a per-page attribute table, no TLB | **(a)**: J3 fills the same view; (c) would be a second translation structure |
| Checkpoint instructions | (a) the compiled-fallibility rule; (b) stop emitting `Checkpoint` before infallible jumps | **(a)**: the interpreter is unchanged |
| In-process MMIO | (a) the decision-28 in-block contract from the helper; (b) always defer | **(a)**: pixel and register loops stay compiled; one ledger |
| Routed VMs | (a) stay interpreted; (b) Machine landing pads in J2 | **(a)** until a routed Machine embedder exists |
| Chaining | (a) J2b after the census; (b) in J2 | **(a)**, as decision-27 Q4 |

## Risks

- **The turn does not get cheap enough.** At ≈ 760 the register loop gains nothing over J1.
  Mitigation: the census comes first; done-when 2 is a gate; J2b is the planned next step.
- **Classifier drift and undefined results.** DIV flags are undefined (SDM Vol 2A DIV p. 3-265),
  as 16-bit SHLD counts were. Mitigation: the lockstep compares full state for seeded operands;
  any divergence keeps that form interpreted, as SHLD's does.
- **TLB conflict misses.** A data set over 256 pages, or two arrays whose pages collide,
  misses into a ~240-instruction helper. Mitigation: the >256-page witness measures the miss
  cost; the slot count is a constant to tune.
- **Epoch churn.** Every invalidation that drops any unit advances the epoch, and every vCPU
  then pays one cold lookup per slot it enters. Demotion bounds this on SMC-heavy pages.
- **Code space.** The Machine map has no per-entry bound. It shares Application's code-space
  accounting and reclamation; exhaustion leaves code interpreted, never wrong.
- **Device re-entry.** A handler called from compiled code must never be called again by the
  re-execution. `Run::decided` already guarantees this for interpreted walks; the transcript
  witness checks it for compiled ones.
- **Embedders that already set tier-up knobs on Machine VMs** start compiling after upgrade.
  That is the opt-in decision-27 Q6 chose. The release note names it, and
  `set_tier_up_after(None)` keeps the old behaviour.

## Maintainer questions

| # | Question | Options | Recommendation |
|---|---|---|---|
| Q1 | Production enablement as decision-27 Q6, plus `Backend::compiles_machine`, with background off meaning "interpreted"? | (a) yes; (b) a new explicit switch; (c) inline compile when background is off | **(a)** |
| Q2 | Use the live code-tag bit on compiled stores from the start, changing decision-27 Q9's order? | (a) tag bit (Q9 (b)); (b) watermark first, then measure | **(a)**: the reset vector (SDM Vol 3A §12.1.4; 80386 PRM §10.2.3) makes the watermark full by construction |
| Q3 | Unpaged accesses through a 256-slot direct-mapped compiled view of the `Tlb`, refining decision-27 §3.2's probe? | (a) as proposed; (b) the 64-slot folded index; (c) a page attribute table | **(a)** |
| Q4 | J2's op set as §3, with strings, x87, segment loads, LOCK forms and PUSHA/POPA/ENTER deferring? | (a) yes, the embedder census picks the next; (b) add REP MOVS/STOS over RAM; (c) add x87 | **(a)**. x87 after J3+J4 under the FPU axis (decision-23 §4), unless the census says otherwise |
| Q5 | Admit `Checkpoint` instructions through the compiled-fallibility rule, lifter unchanged? | (a) yes; (b) change `rollback_required` | **(a)** |
| Q6 | Compiled in-process MMIO through `in_block_access` and `Run::decided` (decision-28 stage 4)? | (a) yes; (b) always defer | **(a)** |
| Q7 | VMs with host-fault recovery stay interpreted in J2? | (a) yes; (b) build Machine landing pads now | **(a)** |
| Q8 | J2b trigger: start chaining if a linked compiled turn stays above 400 or callret misses its range? | (a) yes, J2b before J3+J4; (b) proceed to J3+J4 regardless | **(a)**: decision-27's revisit trigger, made concrete |
| Q9 | T1 guidance for embedders: 50, no Machine regions in J2? | (a) yes; (b) lower T1 for Machine | **(a)**; revisit with the background census |
| Q10 | Machine units in a second `SourceIndex` instance rather than an enum key? | (a) second instance; (b) enum key | **(a)** |

## Consequences

- Compiled Machine code becomes available to embedders for one class, with the interpreter as
  the reference and the fallback for everything that defers.
- J3+J4 inherits the key, the shared map, the request path, the slots, demotion, the tag-bit
  store barrier and the TLB's compiled view. It adds paged fills, rights in the tags and the
  CPL-3 key.
- Revisit if the census puts the turn above 400 after §6 (J2b moves up), if conflict misses
  dominate memory workloads (resize or re-index the view), or if the embedder census shows that
  strings, x87 or port singletons dominate its compiled-class time (lower that family next).

## Implementation notes (2026-11-27, proposed record diff)

Recorded while implementing J2 (bead xj-to9.2). Items 1, 3 and 4 were reviewed by the
supervisor; the rest await review. None changes an answered question.

1. **Where a vCPU caches a unit (§2, §6).** The compiled entry, the hotness count and the
   request state live in a per-vCPU direct-mapped memo of 256 slots keyed by the block a proof
   or a link returns (`vm/legacy/compiled.rs`), not in `flat32::Proof` and `links::Link`. Both
   paths return the same `Arc<IrBlock>`, so one memo serves both, and neither structure
   changes. A conflict costs a recount and one cold lookup of the shared unit.
2. **Machine units in the cache (Q10).** As accepted: `MachineBlockKey`, a Machine map
   beside `map`, and a second `SourceIndex` instance (made generic over its key) with Machine
   pending and suppression sets, under the one spans lock (`cache.rs`). Application's
   `BlockKey`, hash stream and dispatch map are unchanged.
3. **Test modes.** `Delayed(k)` keeps J1's meaning: a synchronous compile, entered k eligible
   turns later. Real background publication is a separate witness mode: the production
   configuration (`tier_up_after` + background) at T1 = 1, 3 and 50.
4. **Device requests from compiled code.** `linear_address` stays `None` for unpaged
   requests, which is what the exit path reports with paging off, so transcripts are equal
   in both tiers. `sole_access` now accepts `Flat32Address` before the access, the Flat32
   whole-operand check that faults before any call.
5. **MemCtx.** No field is added: `MemCtx` stays 176 bytes and the frozen-offset test is
   unchanged, as §8 said. The Machine meaning of two existing fields carries what §4 needed:
   `ret_stack` holds the compiled TLB view (Machine code has no shadow return stack, and the
   Machine dispatcher never set it), and `machine_value` (J1's helper output) also carries
   the instruction IP a compiled block stores before an access helper call. A first cut
   appended two fields (176 to 192 bytes). Every run of every dispatcher then initialised
   16 more bytes, and the Machine loop kept two more live flags. Both showed in the
   interpreter perf gate with compilation off. The code-tag word and bit are in the view's
   entry. `FastEntry` is 32 bytes, pinned by `fast_entry_layout_matches_generated_code`.
6. **Selectors.** The Flat32 lift reads CS/DS/ES/SS with `ReadReg` and FS/GS with
   `IrOp::ReadSelector`. Both are lowered as live loads, besides `LegacyOp::ReadSelector`.
7. **The block's bus address.** The dispatcher's per-turn fetch address is computed as
   `(CS << 4) + EIP` on every path, the real-mode form. The links compare it consistently,
   but it is not the Flat32 bus address. The compiled key and the unit's sources therefore
   use the block's EIP through the bus mask and A20 (`compiled.rs`). The smc-page hint still
   uses the dispatcher's form; that is cost only.
8. **Demotion trigger.** A drop is counted when a turn's drain dropped shared units, for the
   page of the EIP the vCPU then fetches from (the page the compiled key names, through the
   bus mask). It approximates "a code write dropped a compiled unit of this page", and affects
   cost only.
9. **Counters.** The hot-path counters (entries, continues, compiled-retired) exist only in
   test and test-hooks builds.
10. **Admission (§2 items 2-3).** Sparse backing never compiles (spec §4.1, as Application
    already enforces), nor does a backend whose code space is exhausted. This holds in the
    Forced and Delayed test modes too; Scripted runs no generated code.
11. **Retry.** A slot waiting for a unit asks again every threshold's turns as well as after a
    publication, so a refused or rejected request does not stall it.
12. **Interpreter cost with compilation off.** Every Machine check on the shared cache's
    code-write paths (coverage, victims, unit tests, absorb, tag release) is behind one flag
    that is set by the first Machine request, and its bodies are out of line. The Machine
    map and spans are boxed, so the Application fields keep their layout. The Machine
    dispatcher decides compilation once per run with two tests, and reads `ctx.ret_stack`
    in place of a flag inside its loop.
13. **The compiled-fallibility rule.** The lockstep's commit-before-fault programs (POP to
    MMIO and ROM, LEAVE from MMIO, RCL and ADC on ROM) stay exact with the rule disabled.
    The lifter defers flag writes past an instruction's store and writes ESP last, so no
    lifted Flat32 instruction is known to reach the rule. A unit test witnesses it on the
    shapes it refuses.

**Turn census (§6 item 4), MEASURED.** `machine32_loops`, release, ARM64, host instructions
per guest step (one process per cell):

| Path | register | memory | callret |
|---|---:|---:|---:|
| interpreted | 421.6 | 563.2 | 1039.8 |
| J1 forced (2816c7c2) | 176.8 | 284.8 | 1112.8 |
| J2 forced (test-hooks) | 124.8 | 140.3 | 375.7 |
| J2 background, T1 = 50 (production build) | 124.5 | 138.5 | 370.3 |
| Compat32 JIT | 18.4 | 41.8 | 68.8 |

A linked compiled turn cost about 790 host instructions in J1 and about 530 now. The CPU
Profiler (34,000 samples, forced register loop) puts the remainder in the dispatcher's turn
body (about 30 % of all samples), `machine_progress` (about 25 %: eligibility, memo,
participation, call and merge), `follow_link` (about 12 %), the block's `Arc` reference taken
by each linked turn (about 6 %), the SMC poll (about 5 %) and the generated code (about 8 %).
No single item is larger than about a tenth of the turn. Keeping the previous turn's
reference across turns to skip the `Arc` count measured no change. The turn stays above 400,
so Q8's trigger for J2b holds: the remaining gap to done-when 2 (register ≤ 102, callret
≤ 326) is per-turn overhead that only chaining removes. Memory is at 138, next to its gate.
