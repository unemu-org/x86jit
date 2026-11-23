# External x87 numeric-error routing

Design proposal for xj-srh.33, audited against engine revision `7d75712b`.
This note changes no implementation. The existing FPU axis is described in
[`spec.md` §§17.9–17.10](spec.md#179-real16-coprocessor-model-and-x87-limit-checks).
**Hardware** below means a statement from a primary source; **engine choice**
means a proposed software contract, including deliberate approximations.

## Hardware and board boundary

References use printed pages; PDF page numbers are one-based positions in the
supplied scans. The 80286 PRM includes a separately paginated 80287 supplement
(Part II). Source names below are stash-relative, never workstation paths.
The supplied 486 scan has the 1992 title page and SHA-256
`d86207aaebd9b10464f2058efc61559e6141d3b072db100e89bfe17c8187e143`;
its source pin must accompany any later implementation provenance update.

| Ref | Primary source and location | Hardware fact |
|---|---|---|
| H1 | `oracles/intel/i286-prm.pdf`, Part I, Appendix B, **#MF 16 Math Fault**, p. B-9 (PDF 217); **WAIT**, p. B-113 (PDF 321) | With the 80287 ERROR input connected to the 80286, error-checking ESC instructions and WAIT report exception 16. The #MF entry qualifies WAIT with MSW.EM=0. FNCLEX, FNINIT, FSETPM, FNSTCW, FNSTSW, FNSAVE and FNSTENV do not test ERROR. WAIT suspends execution until BUSY is inactive (high). The saved CPU instruction is the later reporting instruction; the offending numeric instruction is identified from the coprocessor environment. |
| H2 | Same PDF, Part II, Ch. 1, **Hardware Interface**, pp. 1-6–1-7 (PDF 356–357), and **The NPX Status Word**, Fig. 1-4, p. 1-10 (PDF 360); Ch. 2, **Proper Error Synchronization**, p. 2-52 (PDF 442) | Before most ESC instructions, the CPU waits for the 80287 to become ready. Status ES, bit 7, denotes an unmasked exception and asserts ERROR. B, bit 15, and BUSY describe execution or exception signalling, not merely a universally interchangeable copy of ES. The next error-checking ESC or WAIT invokes exception 16 with the direct connection. |
| H3 | `oracles/intel/i386-prm.pdf`, Ch. 11, §§11.1.1–11.1.4, pp. 11-1–11-2 (PDF 219–220), and §11.1.5.3, p. 11-4 (PDF 222) | The 80386 supports either 80287 or 80387 bus protocols, selected by CR0.ET. An unmasked numeric error asserts ERROR#; the CPU reports exception 16 at a subsequent WAIT or certain ESC instructions. WAIT first waits for BUSY# to become inactive, then tests ERROR#. With an empty socket, both inputs should be tied inactive. ESC tests EM/TS; WAIT tests MP with TS. |
| H4 | `oracles/intel/i486-prm.pdf`, §4.1.3, **NE**, p. 4-7 (PDF 129); §16.2.1.2, p. 16-20 (PDF 350); §§19.2.2, 19.2.4, pp. 19-2, 19-4 (PDF 388, 390) | CR0.NE is bit 5. NE=1 selects exception 16. With NE=0 and IGNNE# inactive, an unmasked error freezes execution immediately before the next **non-control floating-point instruction or WAIT**, awaiting an external interrupt. With NE=0 and IGNNE# active, the error is disregarded and execution continues. FERR# is activated for an unmasked error regardless of NE. External logic supplies the interrupt. This is the PRM's legacy compatibility error mode. |
| H5 | Same 486 PDF, §25.1 item 1, p. 25-1 (PDF 483), and item 3, p. 25-2 (PDF 484) | Hardware reset initializes NE=0. The 486 has no external BUSY# pin. A WAIT preceding an instruction that itself waits can sometimes act as a NOP, delaying reporting by one instruction. Undefined ESC opcodes do not check pending numeric errors. |
| H6 | `oracles/intel/sdm.pdf`, combined edition `325462-092US`, June 2026, Vol. 3A §2.5, **CR0.NE**, p. 2-16 (PDF 3221); Vol. 1 §§8.7.1–8.7.2, pp. 8-32–8-33 (PDF 240–241) | The SDM confirms NE=0 external reporting through FERR#/INTR/IGNNE# and NE=1 native #MF. A waiting x87 instruction or WAIT/FWAIT stops for the pending error; non-waiting control instructions permit handler access. IGNNE# inhibits error recognition in external mode. |
| H7 | Same SDM, Vol. 3B §25.18.6.7, p. 25-11 (PDF 4132) | On the 486, FERR# assertion can occur at the end of the error-producing instruction or immediately before the next floating-point instruction. Do not infer one universal cycle or instruction boundary from the existence of a pending error. |

ERROR#/BUSY# are written with `#` here to emphasize asserted-low signals; the
286 scan prints bars over their names. An API boolean `asserted=true` must mean
electrically active, not a high pin voltage. This naming convention is an
**engine choice**.

**Embedder responsibility / board requirement supplied for this design:** on
AT-class boards, the coprocessor error output reaches IRQ13 through board logic
rather than the CPU's ERROR input. A write to port F0h acknowledges the latched
board error and releases its BUSY# interlock. The embedder owns that wiring,
the latch, port decoding, interrupt-controller masks, vector mapping and EOI.
These are board properties; H1–H7 are not evidence for the F0h circuit. A board
schematic or technical reference must witness its exact clear/re-arm behavior
before implementation claims board fidelity. An extended-memory runtime can
depend on this external interrupt path even though the direct CPU connection
would deliver vector 16.

There are three distinct states: the x87 numeric exception, the CPU-visible
error/wait signals, and the board's interrupt latch. A board acknowledgement
must not be implemented by clearing the x87 status word. In particular, the
287's error BUSY state in H2 and the board's latched BUSY interlock need not
have identical lifetimes. The 486 freeze in H4 is not an external BUSY# pin
(H5).

## Current engine behavior

Locations below are `file:line` at the audited revision, not claims of complete
generation-specific coprocessor emulation. All paths use the shared numeric
engine; `FpuModel` currently has only `Present` and `None`
(`x86jit-core/src/state.rs:489`).

| CPU / intended coprocessor | Current behavior and implementation evidence |
|---|---|
| 8086/8088, 8087-era path | `Present` makes no faithful 8087 pairing claim. `interp/i8086.rs:369` handles `Coprocessor20`; after the CPU dummy memory read, a pending waiting operation returns **`Exit::UnknownInstruction`**, not vector 16 (`:390`). An empty socket leaves the numeric state untouched. This is also stated in `spec.md:2665`–`2673`. Do not silently replace this refusal with the 287 policy. |
| I80286 / 287-era path | For supported instructions, `interp/legacy.rs:3152` converts `mf_pending_before` to exception 16. `lift/legacy.rs:187` emits the CPU coprocessor gate first; `interp/legacy.rs:3366` applies EM/TS and MP/TS checks. Protected-mode WAIT and coprocessor output forms with `Present` still have scope refusals (`lift/legacy.rs:222`, `interp/legacy.rs:3116`); there is no promise that every protected-mode waiting form reaches the numeric check. The overrun path independently prioritizes a pending numeric exception (`lift/legacy.rs:435`, `interp/legacy.rs:4391`). |
| I80386 / 387-era path | The same `interp/legacy.rs:3152` check raises exception 16 in real and supported protected contexts. `lift/legacy.rs:1807` lowers WAIT and `:1813` onwards handles the FPU instruction families/environment formats. There is no external error output, board acknowledgement, or IGNNE input. Writing CR0 masks with `0x8000_001f` (`interp/legacy.rs:3595`), so bit 5 is not an implemented NE control. |
| 486 | No `CpuModel::I80486` exists (`model.rs:16`–`30`). Neither 486 NE routing nor FERR#/IGNNE# is implemented. Application Long64/Compat32 behavior must not be called a 486 model: its interpreter's shared x87 check produces vector 16 (`interp/control.rs:93`). |

Paths in that table are relative to `x86jit-core/src/` except the spec reference.
`x87.rs:1063` sets sticky exception flags and ES/B on an unmasked arithmetic
error. `x87.rs:1160` checks ES **and** whether the operation waits; it does not
consult board wiring. `x87.rs:1144` exempts the no-wait control kinds. The shared
lift folds some waiting forms into their no-wait kind (`lift/control.rs:670`);
`x87.rs:1140` explicitly records that approximation. A future routing change
must distinguish an actual WAIT prefix from its no-wait twin at the reporting
boundary, or retain and document the limitation rather than claiming fidelity.

After committed control/environment operations, Machine execution recomputes
ES from sticky flags and masks (`interp/legacy.rs:3186`–`3204`); the 386 path
also refreshes B, while the 286 preserves its existing B policy. Thus pending
error can appear on a control-word/environment change as well as arithmetic.
`x87.rs:1341` restores entry status after a failed numeric memory transfer.

“Raises exception 16” does not always mean the embedder sees an exit:
`ExceptionDelivery::{Outward, InGuest}` is a separate choice
(`legacy.rs:23`). Outward delivery uses `Exit::Exception { addr, vector, .. }`
(`exit.rs:321`); in-guest Machine delivery enters the guest handler. The
dispatcher routes an exception through `service_exception`
(`vm/legacy.rs:3343`). That option controls delivery of a CPU exception; it
does not select external board wiring.

## Proposed contract — engine choices

### Selection and notification

Add `NumericErrorRoute::{Internal, External}`, default `Internal`, as a
per-vCPU runtime setting. A VM setter may supply the construction default, as
`set_fpu_model` already does (`vm.rs:1245`, `:2807`). Initially accept External
only for supported I80286/I80386 Machine contexts with `Present`; reject an
unsupported profile/model without changing state. Keep `None`, early-CPU
refusals, Application behavior and unsupported protected-mode forms unchanged.
Select routing independently of `ExceptionDelivery`.

Recommend this typed handoff (names are proposed, not existing API):

```rust
enum NumericErrorEvent { Asserted, Waiting }

// New Exit variant:
ExternalNumericError { addr: u64, event: NumericErrorEvent }

// Vcpu methods, called between runs:
set_numeric_error_route(route: NumericErrorRoute) -> Result<(), NumericErrorConfigError>
numeric_error_route() -> NumericErrorRoute
numeric_error_pending() -> bool
ack_external_numeric_error() -> Result<(), NumericErrorConfigError>
set_ignne(asserted: bool) -> Result<(), NumericErrorConfigError> // future 486 only
```

The getter reports the raw numeric-error level, independent of IF, the PIC,
notification history or acknowledgement. With a consistent numeric image it
is ES; it must use the same committed summary as the delivery check, rather
than a second exception calculation. These methods do not inject an interrupt.

* **Internal:** retain current exception-16 generation and the selected
  `ExceptionDelivery` policy. No new external exits for existing embedders.
* **External assertion:** when a committed instruction first makes the error
  pending, return `Asserted` before executing another guest instruction. The
  raising instruction has completed its existing numeric exception semantics;
  `addr == cpu.rip` names the next instruction. Notify once per uninterrupted
  pending-error episode, including one installed by state restore/edit at run
  entry. `Asserted` does not mean that this next instruction is stalled.
* **External wait:** before a supported operation that tests numeric-error/
  busy readiness, test the external hold described below independently of ES.
  If held, return `Waiting` without constructing exception 16 or executing
  the waiting instruction. `addr == cpu.rip` names that instruction;
  its registers, memory and numeric pointers remain unchanged. Repeated runs
  can return the same zero-progress wait. This is an instruction-local stop,
  never a global gate preventing the interrupt handler from executing.

Notifying at the first committed pending boundary is an **engine timing
choice**, not a claim about signal cycles. In particular H7 permits deferred
486 FERR# assertion. This deterministic first version would choose early
assertion; exact 486 timing and the WAIT optimization in H5 remain outside its
promise. A later 486 model must use the PRM's non-control/WAIT recognition
classification (H4), not blindly apply a modern “every x87 instruction” rule.

| Surface | Fit with existing APIs | Tradeoff |
|---|---|---|
| Typed exit, recommended | Device requests already cross the core boundary as `Exit::PortIo`/MMIO (`exit.rs:261`, `:335`). The embedder updates its board and feeds its selected vector to `inject_irq` (`vm.rs:2374`). | A pending assertion interrupts a long run promptly; a wait cannot spin inside the engine. Requires handling a new Exit variant and precise post-instruction versus pre-instruction positions. |
| Poll only after every run | A read-only level fits `CpuState` inspection and `run_steps`; the embedder can schedule small quanta. | `run` can execute an unbounded integer loop without returning, so IRQ13 can be delayed indefinitely. A level that rises then falls within one run is missed. Polling does not itself explain a stopped waiting instruction; a separate stop reason would still be needed. |

The getter supplements the typed exit for level inspection and deassertion;
it is not the scheduling contract. Re-arm assertion notification when the
committed source becomes inactive, even if it becomes active again in the
same run. Observe changes only after the instruction's memory transfers
commit; a retry or rolled-back status change must not publish a phantom
assertion.

### Waiting, board acknowledgement and interrupt entry

The board converts `Asserted` into its error latch/IRQ13 request. It chooses
the interrupt vector from its controller state, not the literal number 13.
Existing `inject_irq` queues one vector and a later injection overwrites it;
`has_pending_irq`/`irq_ready` expose queue/acceptance state (`vm.rs:2374`–`2407`).
The board must preserve controller arbitration and avoid overwriting an
unrelated queued vector. The core must not fabricate a second queue, PIC EOI,
an NMI, or an `INT 13` instruction.

**Proposed 287/387 abstraction:** retain a small CPU-side external wait latch
in addition to the numeric pending level. An external assertion sets it;
`ack_external_numeric_error` releases it, representing the CPU-side effect of
the board's F0h acknowledgement. The embedder clears its own IRQ latch at the
same board action. A waiting operation remains stopped while this interlock
is latched, including after numeric clearing has deasserted ES.
Acknowledgement releases the CPU-side error wait for the current
episode even if ES remains set: ERROR is disconnected from the CPU in this
board configuration, so ES alone must not recreate exception 16 or immediately
re-latch the hold. In particular, a WAIT before the handler's clear operation
can then complete. No-wait control operations retain their access to the
error state (H1–H3). This is an **engine choice** implementing the supplied
board requirement, not a CPU-manual claim about F0h. The handler must still
service the numeric error; the call does not repair arithmetic or promise
faithful coprocessor execution of further arithmetic with uncleared ES.
Acknowledgement alone preserves CW/SW, registers,
TOP, tags and saved pointers, does not retire a guest instruction, and is
idempotent. It must not create another assertion while the same numeric
episode remains active. Numeric clearing re-arms notification but does not
acknowledge an already latched board interlock. The exact suppression of the
coprocessor's error-BUSY contribution needs a board source before this
abstraction can claim board fidelity; see the maintainer questions below.

A normal flow is: numeric instruction completes → `Asserted` → board raises
IRQ13 → embedder queues the controller-selected vector → engine accepts it
when ready → handler reads no-wait status, acknowledges F0h through
`Exit::PortIo` and clears/masks the numeric error → handler returns → waiting
instruction retries. If F0h occurs after numeric clearing, the order of those
last two actions is reversed without losing the board latch. EOI remains a
separate controller operation. The core never decodes F0h itself.

At a blocked waiting instruction, the next positive run must still perform
normal boundary arbitration before retrying that instruction. An acceptable
IRQ/NMI can enter its handler while the numeric level remains asserted. IF,
interrupt shadows and incomplete memory/port continuations keep their
existing acceptance rules (`vm.rs:2385`–`2407`, `:2452` onwards). If no event
can be accepted, return `Waiting` promptly; do not spin or retire a dummy
WAIT. A host exit request retains its existing boundary priority.

### Future 486: NE and IGNNE#

Do not give a 486 two contradictory routing authorities. **Engine choice:**
on a future 486, the route getter derives Internal/External from CR0.NE;
setting the route between runs sets NE to 1/0 respectively. Subsequent guest
CR0 writes are authoritative. The present 386 mask must remain unchanged;
adding 486 control bits belongs to the 486 model implementation.

With NE=1, recognize pending error as #MF regardless of IGNNE# (H4). With
NE=0, apply the PRM recognition points and freeze while pending unless IGNNE#
is asserted (H4). `set_ignne(true)` bypasses recognition without clearing the
numeric image; deasserting it with an error still pending makes the next
recognition point wait again. It is not a one-shot “resume instruction”
completion. A board F0h handler may drive this input if its circuit does so;
`ack_external_numeric_error` must not secretly assert IGNNE# or clear NE.
There is no synthetic external BUSY-pin latch in the 486 recognition test
(H5). The board's IRQ latch still needs its own acknowledgement.

H4 says FERR# can assert even with NE=1: native #MF delivery does not imply
FERR# is deasserted. Keep the raw level inspectable in both
routes; first-version exit notifications are only enabled in External. A
board that needs exact FERR# observation while NE=1 would need an explicit
signal-observation option or instruction stepping, not a false assertion that
NE disables FERR#.

### Steps, saved state, caching and delivery evidence

These are **engine choices** constrained by the existing execution contract:

* Preserve `run_steps` units and boundary priorities (`vm.rs:3049`–`3070`).
  `Asserted` after a completed instruction includes that instruction's one
  unit; as an instruction-produced exit it wins over simultaneous exhaustion.
  `Waiting` contributes zero. An assertion discovered at positive run entry
  contributes zero. IRQ/NMI delivery itself contributes zero. A zero budget
  neither consumes a notification nor changes a latch, signal or retry.
  `run` must observe the same handoffs despite its different budget unit.
* Append the route and a compact external-error state to `#[repr(C)]
  `CpuState` (`state.rs:296`, current tail `:473`): notification episode state,
  external wait latch and future IGNNE input. Reuse ES as the numeric level;
  do not store a competing pending-error flag. Append fields without moving
  existing JIT offsets. A cloned checkpoint preserves both whether the
  assertion has been reported and whether the board hold was acknowledged.
  Restore of an unreported pending episode reports it; restore of an already
  reported one does not manufacture a new IRQ. Direct state edits are
  reconciled at positive run entry without rewriting the caller's SW/CW.
* A whole-machine checkpoint must also save board/PIC state, queued injected
  events and the existing Vcpu state outside `CpuState`; a CPU clone alone
  has never been a full machine checkpoint (`vm.rs:2075` onwards).
  `reset_execution_state` normally preserves the register image
  (`vm.rs:2208`): it must not silently acknowledge the board hold. Its
  existing optional `ResetOptions::with_x87_init` clears the numeric image;
  reconcile that source change without treating it as a board acknowledgement.
  An actual reset installs the selected CPU and board reset images explicitly.
* Test routing at runtime in both Direct and Checked Machine execution,
  including the overrun precedence check at `interp/legacy.rs:4391`. Do not
  bake a per-vCPU route into shared cached IR. Setter changes occur only
  between runs with no instruction/device continuation in flight; reject
  other changes. Ensure retained execution shortcuts observe the new state
  without requiring a global cache flush. A future compiled consumer must
  also read the runtime state before it can support External.
* An external assertion or wait is not a delivery and adds no delivery-log
  entry. Only actual handler entry is recorded (`vm.rs:2496`–`2516`). A
  board-routed numeric interrupt records the controller-selected vector as
  `DeliveryKind::Irq` (`delivery_log.rs:32`), never a synthetic Fault/16.
  Internal + InGuest still records the real exception-16 delivery; Outward
  still returns it without a record. Board signal traces belong to the
  embedder. Restoring CPU state does not rewind the delivery log; retain the
  existing explicit take/clear discipline.

## Migration and rejected alternatives

Keep Internal as the default for all existing models and constructors. Preserve
their current exits, exception delivery, state initialization and early-CPU
scope refusals. External is opt-in; it does not introduce an 8087 pairing or
complete protected 287 coverage.

For a future 486, distinguish convenience construction from hardware reset:
an ordinary Internal construction preference can initialize NE=1 explicitly,
but an advertised hardware-reset image **must have NE=0** (H5) and therefore
select external recognition. No existing 486 behavior changes, because none
exists. Do not alter a hardware-reset NE value to satisfy a blanket default.

Rejected alternatives:

* Reuse `Exit::Exception { vector: 16 }` and ask the board to reinterpret it:
  InGuest can deliver it before the embedder sees it, and the event is not a
  processor exception in External.
* Auto-inject a fixed IRQ13/vector in the core: conflates an IRQ line with a
  programmable vector and puts board/controller policy inside the library.
* Poll only: loses transitions and gives no prompt handoff from a long run.
* Clear SW/ES with the board acknowledgement: loses the diagnostic context
  the handler needs and invents a numeric instruction side effect.
* Turn a wait into HLT, budget exhaustion, or automatic instruction skipping:
  obscures its cause or changes retirement/RIP. Busy waiting is resumable and
  must permit ordinary external-interrupt entry.
* Add a general pin bus, callbacks or cycle-accurate coprocessor execution:
  exceeds the required error-routing seam. This proposal adds no timing model.

## Witness plan

This is a plan, not a claim that new tests ran. Keep expected bytes, vectors,
RIP and counts independent of the implementation; cite H1–H7 beside hardware
assertions and mark board/engine-policy expectations explicitly.

1. **Pin today's matrix first.** Reuse
   `x86jit-tests/tests/legacy/legacy_i386_x87.rs:411`
   (`error_pin_reports_vector_16_on_wait_and_waiting_esc`): `D8 F1`
   divides a seeded nonzero ST0 by zero ST1 with CW=`037Bh`, then `9B` or
   `D9 E8` checks pending error. Assert ZE/ES after the divide, unchanged
   numeric destination where the current pre-operation rule requires it,
   one retirement, and later exception 16 with saved IP on WAIT/FLD1.
   That existing test covers 386 real/protected and 16/32-bit code;
   add the supported 286 real case and protected-form refusal cases. Cross
   Outward/InGuest to verify exit versus handler entry and the delivery log.
2. **Pin gaps as gaps.** Add the same pending WAIT to the early-CPU path and
   assert `UnknownInstruction`, no delivered vector 16, no retirement and
   stable IP; check `None` separately. Reuse
   `legacy_i286.rs:546` for protected coprocessor refusals. Confirm by the
   catalogue that no 486 case is runnable yet; do not fake a 486 with a 386
   and an edited reserved bit.
3. **Pending is not just arithmetic.** Extend the existing control/environment
   witnesses (`legacy_i386_x87.rs:463` onwards): masked ZE then FLDCW unmask,
   pending FLDENV/FRSTOR, FNSTENV masking, FNSAVE/FNINIT/FNCLEX clearing,
   and FNSTSW/FNSTCW inspection. Use explicit waiting-prefix byte streams
   beside no-wait forms to expose the current folding limitation. Verify a
   failed memory transfer produces neither committed status nor notification.
4. **External lifecycle.** Under both Direct and Checked execution, assert
   `Asserted` with next IP/one unit immediately after the producer, no
   duplicate assertion on resume, then `Waiting` at the later recognition
   point with zero units and no numeric/memory mutation. Prove F0h releases
   a 287/387 WAIT while preserving ES, followed by a no-wait status read and
   clear; numeric clearing alone must not acknowledge the board hold. Exercise
   both status-clear/F0h orders, repeated acknowledgements, IF=0, STI/SS shadows,
   a pending host request, accepted IRQ and NMI while waiting, a handler
   using no-wait forms, and a second independent error episode. Prove that
   an integer loop after the producer cannot hide the assertion inside run.
   Repeat with exact/zero step budgets, interleaved vCPUs sharing cached IR,
   and an error competing with the 286 overrun check.
5. **Board fixture and evidence.** A test embedder supplies an IRQ13 latch,
   remappable controller vector and F0h PortIo handler. Expect its chosen
   vector once, preserve error status through acknowledgement, and log only
   actual `Irq` handler entry. Test masked/controller-queued requests and
   an unrelated already-injected vector. Keep these expectations labelled
   board policy until its primary board reference is selected.
6. **Checkpoint/replay.** Save/restore before notification, after notification,
   while stalled, after F0h and after numeric clearing, with matching board
   state. Compare exits, counts, source/interlock state and the subsequent
   independent episode. Explicitly clear/take the non-rewound delivery log.
7. **When a 486 exists.** Add NE=0/1 × IGNNE asserted/deasserted × pending/
   masked cases, PRM control/non-control recognition, guest NE changes,
   acknowledgement without status clearing, raw FERR observation at NE=1,
   and a true reset image with NE=0. Witness the selected early assertion
   policy separately from hardware-permitted timing. A modern host's #MF
   or another emulator's IRQ behavior cannot establish board timing or the
   historical CPU pin contract.

## Maintainer decisions (2026-11-23)

1. **Surface:** the typed exit `Exit::ExternalNumericError` with the
   `Asserted` and `Waiting` events, plus the raw-level getter, as proposed.
   Deterministic early assertion is accepted.
2. **Board acknowledgement:** the CPU-side external wait latch released by
   `ack_external_numeric_error()`, standing for the board's port F0h write.
   It is an engine choice and an embedder requirement; x86jit cites no board
   source for it, and ES stays set until the guest clears it.
3. **Scope:** the first implementation covers the 286/387-era forms the
   Machine path already supports. Separating the waiting forms from their
   no-wait twins is a separate bead (xj-srh.34) and does not block it; the limitation is
   recorded in `spec.md`.
4. **486:** CR0.NE is the only routing authority (NE=1 #MF, NE=0 external
   through FERR# and IGNNE#); the hardware reset image has NE=0; the embedder
   drives IGNNE# explicitly with `set_ignne`.
