---
id: decision-22
title: Embedder-controlled legacy physical A20 gate
date: '2026-11-11'
status: accepted
---

## Context

A legacy embedder owns the board's A20 controller and needs to drive the CPU's
physical address gate between runs, including while handling a port-I/O exit.
Decision-18 reserved `LegacyCpuState::a20_enabled` and included it in the context
stamp. Decision-20 already retains distinct complete stamps per vCPU.

## Proposed contract

Expose `Vcpu::set_a20_enabled(bool)`, `Vcpu::a20_enabled()` and
`Vcpu::pending_a20_enabled() -> Option<bool>`. Keep the existing per-vCPU byte
as the effective line; the private legacy execution state holds a queued request.
Ordinary legacy entry enables the line. An embedder modeling one shared board signal fans changes out to every vCPU between
runs. This matches the existing exclusive vCPU execution/state API and translation
ownership, without adding shared atomics or a second copy of the signal to `Vm`.
Revisit shared ownership if simultaneous legacy vCPU execution needs asynchronous
board-line changes. Controller policy and I/O-port decoding remain with the embedder.

At the last physical bus boundary, a disabled line clears only address bit 20.
`legacy::physical_span` supplies both the gated address and the contiguous transfer
length up to the next 1 MiB boundary. Legacy instruction fetch, scalar/string/stack
transfers, descriptor reads/accessed-bit writes, and table/IVT accesses all use this
hook. Segmentation and whole-operand checks precede it. Paging is still refused;
future page walks must call this same hook for every physical transfer after
translation, and split at physical discontinuities rather than masking a linear
address before the walk. This change does not establish paging execution evidence.

Split writes probe both physical destinations before committing RAM. MMIO exits
name each physical subtransfer and its own width/value. Retry answers use physical
addresses.

The architectural interruption point is the boundary at which an A20 request
applies. An instruction or string element in flight, including an MMIO
continuation, a partially completed split transfer, a pending acknowledgment or
a decoded retry, keeps its effective line and decoded IR. Consumed reads,
completed writes and acknowledgments survive retries, including when a transfer
overwrites the instruction's own bytes.

A completed instruction, a completed REP iteration and an instruction abandoned
by an architectural fault each reach this boundary. Clear the completed or
abandoned retry and apply the queued value before the next instruction or REP
element, before accepting an IRQ at that boundary, or before returning or
delivering the architectural fault. An IRQ handler entered between REP iterations
can change the line; the resumed REP uses that effective line. The rule is the
same whether the request was queued during the preceding element or made after
the iteration paused. A handler-fetch failure cannot keep the abandoned
instruction's change pending. An accepted interrupt-delivery continuation itself
finishes under its existing line before a change queued during delivery applies.

With nothing in flight, changes apply immediately. Calls requesting the effective
value are no-ops, including while a change is pending; repeating a queued request
leaves it unchanged. `a20_enabled()` reports the effective value;
`pending_a20_enabled()` reports the queued value or `None`. A zero-budget run
does not apply a queued change. Port-input completion, pending IRQs and CPU
registers are preserved.

This boundary timing is an engine contract. An embedder modeling an A20 controller
driven by OUT changes the line after that OUT completes, at the architectural
interruption point before the next instruction. The REP iteration rule follows
the architectural interruption window documented below; it does not hold the
starting line across the whole REP or specify controller ports or board hardware.

Fault exits continue to name the gated physical subtransfer in `Exit` addresses.
Their metadata reports the ungated address as `AddressSpace::Linear` through
`linear_address()` and the gated address through the appended
`physical_address()` field/accessor. Fetch policy and mapping checks use physical
addresses. Application metadata paths leave the new physical evidence unknown.

A20 is already in the complete `LegacyContextStamp`. Toggling chooses the matching
retained context; returning to a retained state reuses its IR after byte, mapping
and fetch-policy revalidation. No global flush is required. Application Real16,
Compat32 and Long64 never consult the hook. Their execution paths and existing ABI
field offsets remain unchanged. The setter has no memory-retry effects in Application.

`reset_execution_state` preserves the A20 request because it is a board signal:
if a value is queued, explicitly abandoning the instruction through reset applies
that value. Otherwise the effective line is unchanged. Replacing the whole CPU
image explicitly also replaces its stored line; embedders installing a reset image
should preserve/reapply their board's current value.

## Primary sources and scope

* Intel SDM **325462-092US**, **Vol 2B §4.3**,
  **REP/REPE/REPZ/REPNE/REPNZ**, Description and Operation,
  **pp. 4-564–568**: a repeating operation may be suspended by an exception or
  interrupt. REP's description on **p. 4-564** states: “The source and destination
  registers point to the next string elements to be operated on, the EIP register
  points to the string instruction, and the ECX register has the value it held
  following the last successful iteration of the instruction.” The operation
  services pending interrupts before executing the next element.
* SDM **Vol 2A §3.3**, **DIV**, **pp. 3-264–266**, and **Vol 3A §7.15**,
  **Event 0**, **p. 7-25**: a zero divisor raises #DE, a fault with the saved
  instruction pointer at the DIV and no accompanying program-state change.
  Applying a queued board-line change when that attempt is abandoned is the
  engine boundary contract above.
* Intel SDM **325462-092US**, **Vol 3A §11.7.13.4**, External Signal
  Compatibility: the external A20M# pin forces physical bit 20 to zero for all
  external bus memory accesses. The pin can be shared by logical processors.
* SDM **Vol 3B §23.1.1**, Address Translation in Real-Address Mode, and
  **§25.33.1**, Segment Wraparound: ordinary real translation can exceed 1 MiB;
  an external A20M# signal produces the compatibility wrap. The pinned edition
  places 8086 emulation in **Vol 3B**, and dates the CPU pin to Intel486.
* Intel **80386 PRM 230985-001** (page image checked), **§14.7 item 18, p. 14-7**: the 386 does
  not itself truncate real addresses at 1 MiB; contrasts the 8086's twenty-bit
  wrap. This does not establish an A20M# pin on a 386.
* Intel **80286/80287 PRM 210498-005** (page image checked), **Appendix C, software item 16 and hardware item 2, p. C-3**:
  software needing real-mode address wrap requires external hardware to force
  upper address lines low. Hardware item 2 specifically requires forcing A20 low
  externally for 1 MiB wrapping. Both describe a board remedy, not an A20M# CPU pin.

Applying a bit-20 gate on both I286 and I386, in real and protected operation,
is an explicit engine bus contract. It is not a claim that either CPU has a
486 pin, nor an implementation of the 286 appendix's broader upper-four-line
mask. Addresses above 2 MiB preserve all bits other than bit 20.

## Witnesses and validation

`x86jit-tests/tests/legacy_a20.rs` pins fetch/data/string/stack wrap and crossing
transfers for both generations, retained-state changes, table operands, GDT
accessed-bit writes, relocated IVT, protected high addresses, precise split-store
faults and distinct linear/physical fault evidence, MMIO read/write completion,
deferred line changes preserving device effects and acknowledgments, MOVSW consumed
reads, immediate changes, port-exit control, per-vCPU independence,
reset preservation and Application Real16 isolation. Private cache witnesses
assert two retained contexts/two IR blocks and no relifts after warmup. The
self-modifying far CALL witness in `legacy_protected.rs` toggles A20 before
resuming its second frame write and verifies that the saved CALL still completes.

The ignored core test `vm::legacy::tests::a20_toggle_throughput` measures 200,000
alternating line changes, each followed by a one-instruction run at FFFF:0010,
in three sequential release samples. It asserts two cached blocks, zero relifts,
correct step counts and zero backend compilation. This measures setter, context
lookup, byte revalidation and interpreter dispatch together. Application-path
cost is checked separately with the existing `real16_loops` benchmark.

| Scope | Authority / contract | Witness in `legacy_a20.rs` unless qualified |
|---|---|---|
| Physical bit 20, both generations, real and protected operation | SDM Vol 3A §11.7.13.4; 80386 PRM §14.7 item 18; I286 PRM Appendix C software item 16 / hardware item 2; proposed bus contract above | `fetch_ffff_0010_and_toggle_between_runs`, `data_reads_and_writes_at_and_across_one_mib`, `only_bit_twenty_is_masked_in_protected_mode` |
| Every part of a crossing fetch/data transfer | Same physical-bus gate | `fetch_instruction_straddling_ffff_000f`, `dword_split_handles_every_boundary_position`, `crossing_two_mib_preserves_the_next_higher_address_bit` |
| Strings and stack | Same physical-bus gate; ordinary instruction transfers remain unchanged | `strings_read_write_and_rep_crossing`, `stack_push_and_pop_at_and_across_wrap` |
| Descriptor tables, accessed-bit writes and IVT/interrupt stack | Same physical-bus gate | `table_operand_and_descriptor_reads_and_accessed_write_are_gated`, `relocated_ivt_and_interrupt_stack_use_the_gate` |
| Mapping/policy checks and MMIO retries at the physical alias | Engine transfer/retry contract above | `fetch_policy_and_fault_address_follow_the_physical_alias`, `split_store_faults_before_committing_ram_and_names_physical_address`, `split_mmio_read_and_write_complete_with_physical_parts`, `changing_line_defers_until_mmio_read_retires_and_same_value_is_a_noop`, `changing_line_keeps_completed_device_write_and_pending_acknowledgment`, `changing_line_keeps_movsw_consumed_device_read`, `gated_fetch_and_data_faults_keep_both_address_spaces` |
| Reuse of correct cached IR across toggles | Decision-20 complete stamps | Core `a20_toggles_reuse_only_the_matching_retained_blocks` plus fetch/toggle witness |
| Per-vCPU control, port-exit handling and reset preservation | Engine API contract above | `port_exit_can_toggle_line_and_reset_preserves_it` |
| Application profiles unchanged | Decision-18 separation | `application_real16_ignores_the_legacy_line_for_fetch_data_string_and_stack`, `application_long64_and_compat32_ignore_the_line`; `real16_loops` comparison |
| Future page-walk cycles | Hook contract above | Paging remains refused; no page-walk execution claim |

## Measured cost and verification

The measurements below describe the original implementation before the deferred
line-change and fault-address corrections; they are not new measurements of the
revised setter. The fix-round checks are recorded separately below.

Measured on AArch64 Darwin with rustc 1.98.1, release defaults, a Flat RAM map,
and the interpreter. The three 200,000-toggle samples measured **2,806,220**,
**2,732,436**, and **2,711,995 toggles/s**: median **2,732,436 toggles/s**, or
**366.0 ns per toggle plus one interpreted instruction**. Every sample retained
two instruction blocks, performed zero further lifts, and made no backend
compilations. These are sequential microbenchmark samples, not an isolated
whole-program performance guarantee or a measurement of the setter alone.

The Application `real16_loops` comparison used the original `af757dc` core
sources and the changed sources on the same host, with argument `10` and the
benchmark's existing warmup. Host instructions per step in the long loops ranged
from **−0.019% to +0.162%** relative to baseline. Elapsed times were lower in the
changed run, but a single sequential before/after pair does not establish a
speedup. No A20 check was added to any Application execution path. Sample values:

| Application loop | Before ns/step | After ns/step | Before host instructions/step | After host instructions/step |
|---|---:|---:|---:|---:|
| `alu` | 17.921 | 16.941 | 398.495 | 398.783 |
| `load` | 24.262 | 20.321 | 523.065 | 523.470 |
| `store` | 21.683 | 19.559 | 486.181 | 486.970 |
| `push-pop` | 25.616 | 22.112 | 551.379 | 551.823 |
| `call-ret` | 51.208 | 46.092 | 1027.955 | 1029.549 |
| `rep-movsw` | 7.787 | 7.047 | 190.185 | 190.149 |

[Raw samples](decision-22-a20-measurements.csv) include all benchmark rows and
all three toggle samples; build time is excluded.

All test binaries ran sequentially, with at most two test threads (one for the
throughput measurement), two Cargo build jobs, and a 180-second timeout per
command. Validation used offline Cargo and explicit host SDK/compiler paths;
no repository configuration was changed for the host toolchain.

| Binary / check | Result |
|---|---|
| Core library, all features | 378 passed, 2 ignored |
| New `legacy_a20` witnesses | 17 passed |
| `legacy_contract` | 6 passed |
| `legacy_real` | 34 passed, 1 pre-existing snapshot failure |
| `legacy_protected` | 18 passed |
| `legacy_i286` | 15 passed |
| `real16_wrap` | 20 passed |
| `real16_jit_guard` | 6 passed |
| `real16_x87_model` | 21 passed |
| `run_steps` | 47 passed |
| `irq_shadow` | 38 passed |
| `vcpu_reset` | 10 passed |
| `boundary` | 1 passed |
| Release `a20_toggle_throughput`, explicitly run | 1 passed |
| Workspace Clippy, all targets/all features, `-D warnings` | Passed |
| `cargo fmt --all --check`, including pinned rustfmt 1.98.1 | Passed |

The original sources fail all **15 legacy A20 witnesses** and pass both
Application isolation witnesses. For this before-change run only, a test-local
adapter writes/reads the already reserved A20 byte because the public methods
were absent. The implementation and original test file were restored afterward.

`application_real16_ir_matches_frozen_parent_snapshots` fails identically on
original `af757dc` sources: the current `RepStringReal` debug image includes
`addr_bits: 16`, while the frozen snapshot omits it. No lifter/IR/snapshot files
were changed here. The regression total is **611 passed, 1 pre-existing failure,
2 ignored**, plus the separately successful release toggle benchmark. The full
workspace suite, hardware oracles and real-program ladder were not run; these
measurements do not claim paging or protected interrupt-gate support.

Reproduce the throughput measurement with:

```sh
timeout 180 cargo test --offline --release -p x86jit-core --lib \
  vm::legacy::tests::a20_toggle_throughput -- \
  --ignored --exact --nocapture --test-threads=1
```


## First fix-round verification

The revised contract retains memory continuations and decoded retries across line
changes. The A20 witnesses cover both acknowledgment orders, consumed MOVSW reads,
changes taking effect before the next instruction in the same run, zero-budget
preservation, immediate changes and queued changes at explicit reset. The protected CALL witness runs with and without a queued A20
change after its first frame write overwrites its instruction bytes. Fetch and
data fault witnesses check linear 0x100000 separately from physical zero, including
crossing transfers. A core witness checks recovered host evidence against the
physical address, including an exact later-byte fault.

Final checks used offline Cargo, rustc 1.98.1 on AArch64 Darwin, two Cargo build
jobs, two nextest test threads and a 180-second timeout per command. Each final
nextest command selected one binary. An explicit SDK/compiler wrapper resolved a
local SDK/linker mismatch, including the core test's nested rustc link; repository
toolchain configuration was unchanged.

| Check | Result |
|---|---|
| `cargo check -p x86jit-core --tests` | Passed |
| Core library, all features | 379 passed, 2 ignored measurements |
| Core integration binaries, all features, selected separately | 36 passed |
| `legacy_a20` | 22 passed |
| `legacy_protected` | 18 passed |
| `legacy_real` | 35 passed |
| `legacy_i286` | 15 passed |
| `legacy_contract` | 6 passed |
| `cargo clippy --workspace --all-targets -- -D warnings` | Passed |
| `cargo fmt --all --check` | Passed |

The final nextest total is **511 passed, 2 ignored**. This fix round does not
remeasure throughput or claim the full workspace, Unicorn, hardware-oracle or
real-program-ladder gates.


## Second fix-round boundary witnesses

The architectural interruption rule supersedes the first fix round's retention
of a starting line across REP iterations. `line_change_at_rep_iteration_boundary_uses_new_line_for_next_element`
starts with CX=2 and an MMIO source: a request queued during the first element
applies after that element, within one run or across step slices. A request made
after the pause applies immediately. The second element reads the new RAM alias
in each case. `irq_handler_changes_line_between_rep_iterations` accepts an IRQ
at CX=1, executes a controller OUT in the handler and returns with IRET. The
handler toggles the effective line in both cases: it disables A20 without a
prior request and re-enables A20 after a queued disable. The resumed REP reads
RAM or MMIO according to that handler-selected line.

`mmio_divide_fault_applies_queued_line_and_discards_retry` completes a memory DIV
with a zero MMIO divisor, checks #DE and unchanged architectural destinations,
and observes the queued line already applied at the outward fault. It then
retries through the new RAM alias and checks fresh instruction fetch.
`divide_fault_handler_fetch_failure_does_not_leave_line_pending` explicitly
routes the outward fault vector through the supported real interrupt-delivery
seam (legacy exceptions remain outward under decision-18 §4). The handler fetch
is denied at the new physical alias on two successive runs; the line remains
effective and no request remains pending. Both witnesses run on I286 and I386.

All four boundary witnesses fail against the first fix-round engine at `66ee3e4`
and pass with this revision. Final validation used offline Cargo and rustc 1.98.1
on AArch64 Darwin, two Cargo build jobs, two nextest test threads, one binary per
nextest command and a 180-second timeout per command. The existing SDK/compiler
wrapper also covered the core test's nested compiler; repository toolchain
configuration was unchanged.

| Check | Result |
|---|---|
| `cargo check -p x86jit-core --tests` | Passed |
| Core library, all features | 379 passed, 2 ignored measurements |
| Core integration binaries, all features, selected separately | 36 passed |
| `legacy_a20` | 25 passed |
| `legacy_protected` | 18 passed |
| `legacy_real` | 35 passed |
| `legacy_i286` | 15 passed |
| `legacy_contract` | 6 passed |
| `cargo clippy --workspace --all-targets -- -D warnings` | Passed |
| `cargo fmt --all --check` | Passed |

The second fix-round nextest total is **514 passed, 2 ignored**. Throughput,
Unicorn, hardware-oracle and real-program-ladder gates were not rerun.
