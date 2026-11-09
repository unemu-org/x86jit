---
title: Instruction-exact step budget and interrupt acceptance for the Real16 interpreter
date: '2026-11-09'
status: accepted
---

# Instruction-exact step budget and interrupt acceptance for the Real16 interpreter

## Context

A Real16 embedder needs bounded execution for device scheduling and an interrupt
acceptance boundary at which it can acknowledge its interrupt controller. The request
comes from **decision-15 (unemupc98)**; that is the embedder's record, distinct from our
[decision-15](decision-15%20-%20Allocation-free-child-step-for-shared-address-space-children.md).
This record describes an additive API; the maintainer accepted it on 2026-11-09 with the recommended choices below, after the requesting embedder checked it against its own design record (no conflicts).
Source inspection is against `34ceea8`; no execution measurements were made.

Today `Vcpu::run(vm, Option<u64>)` budgets dispatch blocks, including interrupt delivery
and some retry work, rather than instructions (`x86jit-core/src/vm.rs:1885–2050`).
Real16 already bypasses compilation and region formation even with a compiling backend,
and caches interpreted blocks by physical fetch address and mode (`vm.rs:2710–2755`).
`step_instruction` calls `interp::step_one`, which lifts one instruction afresh; it is
not a run loop with equivalent IRQ polling and completion handling (`vm.rs:1652–1720`,
`x86jit-core/src/interp/mod.rs:58–113`). Neither API bounds REP iterations.

The block walk retires the previous instruction at `InsnStart` and infers final
retirement from changed RIP (`interp/mod.rs:296–375,441–460`). **xj-5c2** is expected
to replace that inference: completed instructions, including `jmp $`, retire;
fault delivery does not. **xj-d7d.2** adds the SS-load shadow alongside STI's shadow.
These are prerequisites, not changes claimed by this document. Current STI tracking
resets at instruction start and the dispatcher samples readiness only between blocks;
an exact API must preserve and inspect the incoming shadow before that reset.

Completion paths differ. OUT, IN, HLT and syscall advance RIP and currently retire at
the exit; IN leaves `pending_port_in`, and `complete_port_in` writes the accumulator
without replay (`interp/control.rs:402–453`, `vm.rs:1638–1651`). MMIO leaves RIP on the
instruction and uses answers, acknowledgements and an owner-keyed retry ledger before
the instruction can retire (`vm.rs:1605–1636`, `state.rs:662–748`). REP helpers commit
indices/count/flags per successful iteration and retain the instruction RIP on an early
stop (`interp/mod.rs:7640–7824`). Their `StrMem` accesses do not consume scalar MMIO
answers (`interp/mod.rs:7297–7351`): bounded REP must not assume that replay already
works for every device access.

### Primary-source sections

Checked in the pinned `oracles/intel/sdm.pdf`, order **325462-092US, June 2026**,
using `pdftotext -layout`. Use this edition's section numbers directly; its interrupt
chapter is 7, where older editions used 6. Named instruction entries below are sections
within the indicated instruction-reference section; page numbers disambiguate them.

| Authority | Fact used by this proposal |
|---|---|
| SDM Vol 3A §7.6, Program or Task Restart | Faults restart at the faulting instruction; traps resume after the completed instruction or at its branch target. Interrupts can be taken after a REP iteration with restart registers for the next iteration. |
| SDM Vol 3A §7.8.1; Vol 2B §4.3, STI, Description, pp. 4-673–674 | IF masks maskable hardware interrupts. STI starting with IF=0 inhibits them at the immediately following boundary; execution of the next instruction or delivery of an event ends that inhibition. STI with IF already set does not create that inhibition. |
| SDM Vol 3A §7.8.3; Vol 2B §4.3, MOV, Description, p. 4-29; POP, Description, p. 4-399 | Successful MOV SS and POP SS inhibit interrupts at the following boundary, ending after the next instruction or exception delivery. Only the first of consecutive SS loads is guaranteed to inhibit events. LSS does not have this inhibition. |
| SDM Vol 2B §4.3, REP, Description/Operation, pp. 4-563–564; REPE/REPZ, pp. 4-565–566; REPNE/REPNZ, pp. 4-567–568 | Repetition tests the count, commits iterations, and can service interrupts between iterations. Restart IP names the string instruction; indices/count describe the completed prefix. |
| SDM Vol 2A §3.3, HLT, Description/Operation, p. 3-439; Vol 3A §7.8.1 | HLT stops execution; an enabled interrupt can resume it, with saved IP after HLT. A masked maskable vector is insufficient to wake it. |

These sources license hardware behavior. Budget units, exit precedence, and the port
handoff accounting below are engine contracts, not claims about hardware cycle counts.

## Decision

Recommend the following contract, subject to the maintainer's choices below.

### 1. Public API and scope

Keep `run` and `step_instruction` signatures. Add these Rust API shapes (constructors,
accessors and non-exhaustive types are sketched, not full definitions):

```rust
pub struct StepOptions { /* private; Default: stop_when_irq_ready = false */ }
impl StepOptions {
    pub fn with_stop_when_irq_ready(self, enabled: bool) -> Self;
}
pub struct StepRun { /* private: steps: u64, stop: StepStop */ }
impl StepRun {
    pub fn steps(&self) -> u64;
    pub fn stop(&self) -> &StepStop;
    pub fn into_stop(self) -> StepStop;
}
#[non_exhaustive]
pub enum StepStop { Exit(Exit), IrqReady }
#[non_exhaustive]
pub enum StepRunError { UnsupportedMode, PendingPortIn }
impl Vcpu {
    /// Run at most max_steps accounting units on the Real16 interpreter.
    /// One unit is a completed instruction, or a completed REP iteration;
    /// empty REP and port handoffs follow the rules below. Never begin unit N+1.
    /// Returns completed units and the existing Exit, or an IRQ-ready boundary.
    /// Zero budget has no guest effects, including delivery or completion effects.
    pub fn run_steps(&mut self, vm: &Vm, max_steps: u64, options: StepOptions)
        -> Result<StepRun, StepRunError>;

    /// Whether a Real16 maskable IRQ could be accepted at this boundary.
    /// Independent of whether a vector is queued; does not acknowledge or deliver.
    /// False outside Real16. Pending device completion also prevents acceptance.
    pub fn irq_ready(&self) -> bool;
}
```

`StepStop::Exit(Exit::BudgetExhausted)` means the **step** limit was reached in this
API; it retains its block meaning in `run`. `IrqReady` is a separate stop reason,
not `Exit::Interrupted` and not a new guest exception. No callback or completion-only
resumption is added. Normal exit metadata, syscall evidence, host-interruption evidence,
SMC handling, memory consistency and host-fault recovery retain their existing contracts;
evidence accessors must document `run_steps` as another run entry.

Refuse Long64/Compat32 with `UnsupportedMode`, before changing guest state, even for
N=0. Accept a Real16 VM configured with a JIT backend: the existing Real16 interpreter
restriction still applies. Do not identify backend types or silently interpret other
modes. As with `run`, the vCPU must belong to the supplied VM.

### 2. Budget, retirement and exits

For supported calls, `0 <= steps <= N`. N=0 immediately returns budget exhaustion with
zero steps: no fetch, instruction, interrupt/exception delivery, completion consumption,
shadow change or retry cleanup. An existing host request or eligible vector stays pending.
For positive N, check before starting any instruction/iteration; once the Nth unit has
completed, stop without any effect of the next unit. An incomplete unit can perform the
partial effects already permitted by its ordinary exit; the limit does not make faults
or multi-transfer device operations atomic.

| Outcome of the current instruction | Contribution and resume rule |
|---|---|
| Ordinary completion, including taken/self branch | One; next RIP is the actual successor, even if unchanged. |
| Non-REP string instruction | One after its single operation completes. |
| Nonempty REP | One per completed iteration; no extra unit for final instruction retirement. |
| REP with entry count zero | One for completing the empty instruction, preventing an unmetered stream of empty repeats. |
| OUT / IN `PortIo` handoff | One, matching current interpreter retirement at advanced RIP. OUT's device action is the embedder's obligation. IN's value is still pending; see below. |
| HLT / an existing syscall exit | One if the instruction completed; return that exit, with RIP past it. This introduces no new Real16 syscall instruction support. |
| Fault, fetch failure, unsupported instruction, MMIO request | Zero for the incomplete instruction/iteration; retain all earlier completed units. |
| Completed software INT / trap-class event | One for the completed instruction, zero additional units for delivery (SDM Vol 3A §7.6). A fault-class gate is not a completed INT. |
| Hardware IRQ or fault-class exception delivery | Zero, including internal Real16 IVT delivery; handler instructions count normally. Delivery failures retain the ordinary fault exit. |

**Attribution of the returned count to the stopping instruction.** An embedder
time-stamps a device access at the stopping instruction's start, so the record fixes
which counted step, if any, belongs to it:

| Stop | Stopping instruction's share of `steps` |
|---|---|
| `Exit::PortIo` (OUT, IN), `Exit::Hlt`, an existing syscall exit | The **last** counted step; it started at step `steps - 1` of the call. |
| MMIO request, fault, fetch failure, unsupported instruction | **Zero**: every counted step is a completed predecessor; it started at step `steps`. |
| A REP instruction that exits mid-iteration (MMIO, fault) | Its completed iterations are counted; the exiting iteration contributes zero. |
| `Exit::BudgetExhausted`, `StepStop::IrqReady` | Nothing is in flight; the next instruction starts at step `steps`. |

**Port handoff is an explicit accounting convention.** IN is counted on `PortIo` even
though the accumulator is incomplete. `complete_port_in` adds no unit and must run before
another positive-budget call; otherwise return `PendingPortIn` without guest changes.
N=0 remains harmless with pending IN. This preserves existing retirement accounting and
avoids charging an instruction to a later scheduling slice merely for supplying its value.
It is not a claim that IN's architectural result is complete at that exit.

Keep `retired_instructions` as the completed-whole-instruction counter, with xj-5c2's
completion classification, including the existing port-handoff convention. No cumulative
step counter is required: the exact per-call count lives in the stepped run context.
For a successful call, **steps = delta(retired_instructions) + I - C**, where I is the
number of completed REP iterations in this call and C the number of nonempty REP
instructions completed in it. Empty REP contributes one to both steps and retirement.
A REP interrupted after k iterations contributes k steps and no whole retirement until
it finishes. Compute the ledger directly, not by RIP movement or count-register subtraction.
This relation is call-local; it does not reinterpret `executed_instructions` or impose
new per-iteration accounting on the block API.

An instruction's own exit wins even if it consumes the last unit: return OUT/IN/HLT/
syscall/fault rather than hiding it behind budget exhaustion. At other boundaries use
this order: exhausted step budget; existing host-exit request when completion permits;
delivery of an already queued eligible IRQ; optional ready stop; next guest unit. Host
requests stay deferred across pending completion; a budget can still prevent replay.
Thus budget and readiness
coinciding return budget exhaustion, and the embedder can still query `irq_ready()`.
Internal fault delivery follows the existing Real16 policy; neither it nor handler entry
is a new step. A step limit is not a wall-clock bound on repeated zero-retirement faults;
the existing host-exit request remains the escape, with no new zero-progress yield.

### 3. Exact block walk and REP resumption

Use an internal **const-specialized stepped walk** over ordinary cached IR blocks.
Before a later `InsnStart`, first finalize the preceding instruction and its shadows,
then check the boundary, then enter the next instruction. A cut installs that marker's
16-bit guest IP; it must not leave RIP at the block entry or use block-end fall-through.
Terminator/exit returns also carry explicit completion classification. The stopped marker
has performed no register, flag, memory, trace-entry or shadow-consuming effect. Temporary
IR values can be discarded because the cut precedes all ops of the next instruction.

Give the stepped REP helper a remaining-unit limit and explicit completed-iteration count
plus completion/stop status. Check before an iteration's first access; charge only after
its accesses and index/count/flag updates complete. If count exhaustion or REPE/REPNE's
condition ends repetition on the last permitted iteration, finish the instruction and
advance RIP then, without another step. Otherwise retain its IP and updated restart
registers. Do not test the next iteration by reading its memory. This follows the restart
boundaries in SDM Vol 3A §7.6 and Vol 2B §4.3, REP/REPE/REPNE, Description/Operation.

Resume through the normal Real16 resolver at the current CS:IP; a cached suffix block
may be reused or a new suffix lifted. This can add cache entries for cut positions but
requires no persistent IR cursor. Handle SMC before resolving again. Read-ahead decoding
does not authorize an exit or guest event from beyond the stopped boundary; only execute
the completed prefix and report a later decode/fetch problem when that instruction is due.

MMIO replay remains ordinary execution with a positive budget: reserve room for its unit,
consume the answer/ack, and charge only when the instruction or current iteration finishes.
Further MMIO/fault exits can return zero new units with the retry owner intact. Pending
means an outstanding request **or** an answered-but-unfinished replay, including recorded
parts even when the scalar answer slots are empty. N=0 must preserve all of it. Clear a
REP iteration's device ledger when that iteration finishes, even though instruction RIP
is unchanged, so equal-address later iterations cannot reuse a prior device transaction.
The stepped helper needs an answer-aware transfer path; the current raw `StrMem` loop
alone is insufficient. This does not add unsupported string-I/O opcodes or general MMIO
RMW support. Preserve decision-14's exit-owned metadata and partial-effect qualifications;
zero charged steps is not evidence of zero memory/device effects.

### 4. Interrupt readiness and shadows

For Real16 define `irq_ready = IF && !sti_shadow && !ss_shadow && !completion_pending`.
The last term includes pending IN and the entire MMIO retry described above, before and
after an answer is supplied. Readiness neither tests `has_pending_irq` nor examines an
embedder's interrupt controller. Use the same predicate for the stepped dispatcher's
delivery gate. It must not inject between sub-transfers of one unfinished unit.

Carry shadow state across `run`, stepping, and `run_steps` on `Vcpu`; do not reinitialize it
from a zeroed block-local `RetireInfo`. Successful STI beginning with IF=0 arms STI's
shadow; successful MOV SS/POP SS arms SS's shadow. Check readiness before starting the
protected successor; merely entering an instruction marker, cutting, or replaying an
incomplete device operation must not expose a ready boundary. The next completed unit
consumes the incoming shadow. For a REP successor, interpret the first completed iteration
as that unit: this combines the iteration boundary in SDM Vol 3A §7.6 with §§7.8.1/7.8.3;
it must have a dedicated witness rather than treating every budget cut as shadow expiry.
Actual exception/event delivery ends inhibition under those sections; a host-side map/MMIO
exit without delivery does not. Do not arm a shadow for a faulted SS load or for LSS.

For consecutive SS loads, recommend consuming an incoming **SS** shadow without rearming
it from the protected SS load: a deterministic engine policy within §7.8.3's guarantee
for only the first load. Consume old state before arming new state. For mixed sequences,
STI(IF=0); MOV SS replaces STI blocking with SS blocking; MOV SS; STI(IF=0) replaces SS
blocking with STI blocking; MOV SS; STI(IF=1) leaves neither. These are explicit model
resolutions to witness, not a guarantee that every shadow-producing sequence is atomic.
This proposal covers maskable IRQ acceptance, not SS loads' wider debug/NMI rules.

With `stop_when_irq_ready`, return `IrqReady` at the first eligible boundary, including
entry with zero steps if already ready and N>0. This is a level check; failure to inject
can produce the same ready stop again. With a shadow active, execute the protected unit,
then stop before the following unit. Also check between REP iterations. The embedder
acknowledges its controller, calls `inject_irq(vector)`, then resumes with positive budget;
the queued vector is delivered before another ready stop or guest instruction. Delivery
itself is not a step. If state is changed so readiness is lost, keep the vector queued.

Starting with IF=0 and enough budget, `sti; hlt` completes both before returning HLT;
HLT wins over a simultaneous ready/budget stop. A subsequent eligible IRQ saves the
post-HLT IP (SDM Vol 2A §3.3, HLT).
With IF=0 a pending maskable vector stays queued and cannot wake HLT (§7.8.1). Keep the
existing embedder-owned halt policy: after HLT the embedder parks the CPU until a modeled
wake event is eligible; reentering is permission to resume. This proposal adds no persistent
halt latch or NMI/reset mechanism. A raw rerun after masked HLT is not a hardware wake test.

### 5. Cost and layout constraints

The existing block-budget `run` uses the unstepped specialization: no step countdown,
ready polling, per-iteration step increment, new counter store or runtime "stepping?"
branch in its walk/helpers. Select stepped and stop-when-ready specializations at API
entry. Share semantics, not runtime budget checks; keep existing public helper signatures
as wrappers around their unbounded specialization. Other modes' generated code is unchanged.
The SS shadow/retirement prerequisite fixes are independent of adding this API.

Keep the budget ledger local; append private SS/shadow-continuation state to `Vcpu` as
needed. Do not reorder `CpuState`, add stepping offsets to `jit_abi`, or change existing
ABI fields/return codes. Any unavoidable state addition there must be append-only.
Inspect generated unstepped loops and compare interleaved dispatch/REP benchmarks before
landing: static elimination guarantees no added checks, but code-size/layout effects
still require measurement. Acceptance requires no block-run performance regression.

## Alternatives considered

| Choice | Assessment |
|---|---|
| Per-`InsnStart` checks and bounded REP only in stepped specializations | **Recommended.** Reuses normal IR/cache, amortizes dispatch over a block, and can stop after any completed unit. More interpreter variants and explicit completion bookkeeping. |
| Force single-instruction blocks | Simple instruction cut, but increases dispatch/cache entries and changes translation policy; still needs a bounded REP helper. Reject as the default. |
| Loop over `step_one` / `step_instruction` | Useful reference for tests; repeatedly decodes/lifts/allocates and lacks the run dispatcher's full IRQ/completion protocol. REP is still unbounded without helper work. Reject for production. |
| Runtime optional countdown in every block walk | Less generated Rust code, but taxes the existing API on each instruction/iteration. Reject. |
| Cumulative `Vcpu::steps()` | Convenient virtual-time query, but counting every block-run REP iteration would add work to the old API; counting only stepped calls needs another scope contract. Prefer the documented retirement relation and exact per-call result. |
| Count IN only when its value arrives | More literal architectural completion, but changes retirement or requires deferred accounting credit and a different completion API. Viable only if the maintainer rejects the explicit handoff convention. |
| Refuse all JIT-configured VMs / interpret every mode | Backend refusal contradicts existing Real16 dispatch; broader fallback hides unsupported scope and cost. Prefer mode-based refusal. |

## Consequences

An embedder can partition an instruction stream into deterministic slices and acknowledge
IRQs at acceptance boundaries without a callback. Instruction retirement remains distinct
from REP scheduling units and from device completion. Small slices can create suffix
cache entries and incur extra dispatch; stepped specialization increases code size. Device
replay and shadows must survive cuts and exits, so this is more than a loop around `step`.
Implementation is tracked by the epic xj-d7d (xj-d7d.1 step-budget run, xj-d7d.3 readiness query, xj-d7d.4 stop-when-ready); xj-5c2 and xj-d7d.2 are its prerequisites.

## Witness tests required before implementation acceptance

1. Cut N=7 inside a cached 100-instruction block: exactly seven effects/retirements,
   RIP at instruction eight, and no eighth trace entry, fault, or device access.
2. Partition the same stream with N=1, 7, 100 and mixed slices, including zero calls;
   match registers, flags, memory, aggregate steps/retirement and serviced device events.
3. `jmp $`: any positive N completes N units/retirements at unchanged RIP; fault-class
   IVT delivery at changed RIP retires none. Include completed INT/trap and self-target IRET.
4. Cut/resume REP MOVS/STOS and conditional CMPS/SCAS: first/middle/final/zero iterations,
   DF directions, count wrap boundaries, conditional early termination, source/destination
   faults and host requests. Assert no next-element access; last iteration advances IP.
5. MOV SS and POP SS followed by stack-pointer update: slice after every unit, including
   a trapped successor, a faulted SS load, repeated SS loads and STI/SS overlap. Test STI
   starting with IF=0 and IF=1, `sti; nop`, `sti; cli`, and shadowed REP's first iteration.
   Assert the first ready stop is after the protected unit, before any later instruction.
6. Starting with IF=0, `sti; hlt` with a pending vector: HLT return first, then delivery
   before the next instruction, saved IP after HLT. IF=0 plus a queued vector: HLT returns and
   the embedder parks without acknowledging/waking on that vector.
7. Budget ends exactly on OUT/IN/HLT: original exit wins and includes that unit. IN blocks
   readiness until completed; completion adds no step; positive rerun without its value
   is refused. N=0 with every pending completion/request/vector leaves all state unchanged.
8. Scalar and REP MMIO: repeated sub-transfer exits, supplied but unconsumed answers,
   recorded parts with empty answer slots, and same-address later iterations. No duplicate
   device effect, no premature step/IRQ, metadata survives, and budget limits replay too.
9. Ready on entry returns zero; queued eligible IRQ overrides another ready stop and
   reaches the handler before the next instruction. Budget/readiness ties, host-request
   precedence, non-Real16 refusal, and Real16 with a compiling backend follow the contract.
10. Verify unchanged ABI offsets and unstepped machine-code checks, then block-run
    dispatch/REP performance against the prerequisites-only baseline. These are required
    implementation witnesses, not results claimed for this documentation-only change.

## Open choices for the maintainer

- Accept the recommended IN handoff accounting despite its outstanding architectural
  value, or choose deferred accounting and specify the corresponding retirement/API change.
- Accept first-iteration shadow consumption for a REP successor and the no-rearm policy
  for protected consecutive SS loads; confirm the combined SDM reading with focused witnesses.
- Accept `run_steps` / `StepRun` / `StepStop` and a builder option, mode-based typed refusal,
  a level-triggered ready stop, and the stated exit/budget/host-request/IRQ precedence.
- Accept a local count plus the retirement relation, or require a separately scoped
  cumulative counter; retain normal suffix-cache resolution unless measurement warrants
  a validated continuation cursor. Neither choice may tax the unstepped hot loop.

Revisit when another execution mode needs exact stepping, a device requires a different
completion contract, or measured suffix-cache/code-size cost warrants another strategy.
