---
title: Exit-owned exception metadata with explicit knowledge and partial-progress limits
date: '2026-11-06'
status: accepted
---

# Exit-owned exception metadata with explicit knowledge and partial-progress limits

## Context

Accepted by the maintainer on 2026-11-06 for **xj-ir6.16**, with the four resolutions recorded under *Resolved questions*. This record does not claim the implementation. Evidence is from revision `fbad3bcafa792ad08d1b0b55eda77e8bc2f6a9c3`. **READ** means source inspection, with repository-relative `file:line` references; **MEASURED** means a scratch-copy probe on macOS AArch64, Rust 1.98.1. The inspected source files matched the HEAD archive used for probes.

An embedder needs the exception that ended execution and the state it can safely deliver or retry. Today `UnmappedMemory` reports address/access, `Exception` reports saved RIP/vector, and neither carries an error code. Saved RIP is already deliberately different for faults and traps (READ `x86jit-core/src/exit.rs:118`, `x86jit-core/src/exit.rs:171`). Spec §5.2 describes that boundary and §16 requires commit-last lowering (READ `backlog/docs/design/spec.md:373`, `backlog/docs/design/spec.md:1199`).

Three qualifications constrain an honest API. `Prot` is advisory; region kind controls routing, ROM writes are discarded on interpreter paths, and Trap fetches are deliberately refused (READ `x86jit-core/src/memory.rs:294`, `x86jit-core/src/memory.rs:334`). Guest paging, descriptor enforcement and privilege-dependent access checks do not exist as a complete model. Permission enforcement and protected-mode segmentation remain deferred (READ `backlog/docs/deferred.md:67`, `backlog/docs/deferred.md:241`). Decision-13 now preflights crossing compiled stores before any destination bytes commit. Faults at probes prove no committed RAM bytes; faults in actual transfers remain qualified because concurrent protection revocation after preflight can still allow partial effects.

### Primary-source sections

The requested citations are **Intel SDM Vol 3A §4.7, Page-Fault Exceptions; §6.13, Error Code; §6.15, Exception and Interrupt Reference**. The supplied `oracles/intel/sdm.pdf` is order **325462-092US, June 2026**. Reading it with `pdftotext -layout` establishes this crosswalk:

| Requested section | Section in the pinned PDF | Use here |
|---|---|---|
| Vol 3A §4.7 | **§5.7**, Figure 5-12 | P/W/R/U/S/RSVD/I/D/PK/SS/HLAT/SGX, including I/D control prerequisites |
| Vol 3A §6.13 | **§7.13**, Figure 7-7 | Selector index, EXT/IDT/TI; no pushed code for software INT n |
| Vol 3A §6.15 | **§7.15**, Events 0/1/3/6/10–14/16/17 | Error-code presence, fault/trap saved IP, exception-specific state |

Subsequent SDM references use the requested numbering with the pinned numbering in parentheses. They refer to these named sections, not to unrelated sections with the old numbers in the new edition. REP restart state additionally follows Vol 2B, REP/REPE/REPNE, Description/Operation; the implementation cites it at `x86jit-core/src/interp/mod.rs:7036` (READ).

## Options and trade-offs

| Option | Benefits | Costs / reason to choose or reject |
|---|---|---|
| Replace exits with a mandatory architectural exception structure | One uniform object | Breaks consumers and incorrectly implies every host or map failure is a fully known guest exception. Reject. |
| `Vcpu::last_fault_metadata()` side record | Keeps `Exit` small; can avoid fault-time allocation | Requires a precise lifecycle for every run/step return, panic, reset and retry; invalidation adds work even to non-faulting calls. A query cannot reconstruct lost host-transfer information. Viable if allocation is forbidden, but not preferred. |
| **Optional boxed metadata on existing fault exits, queried through `Exit`** | Owns the event, works with run/step/standalone execution, no stale-vCPU state or per-run clearing; source compatible | One allocation on a fault, larger clone cost for that exit; keep allocations outside signal context. Recommended. |
| Decode current memory or pass the host error code through | Little new static metadata | Code may have changed or become unreadable; host paging/privilege/ISA describes a different machine. Reject as an architectural source. |

**The bead's “extend Exit = breaking” premise is outdated.** The enum and its struct variants are already non-exhaustive; external patterns require `..`, and stable const constructors promise defaults for future fields (READ `x86jit-core/src/exit.rs:26`, `x86jit-core/src/exit.rs:111`, `x86jit-core/src/exit.rs:231`, `x86jit-core/src/exit.rs:282`). Adding an optional field and accessor is additive for supported Rust source consumers. Rust enum layout is not a stable binary/FFI contract.

## Recommended decision

### 1. Shape, ownership and field meanings

Add `metadata: Option<Box<FaultMetadata>>` to `UnmappedMemory` and `Exception`, plus `Exit::fault_metadata(&self) -> Option<&FaultMetadata>`. Keep variant names, existing field meanings, constructors and `run()` signatures. Existing constructors use `None`; add explicit enriched constructors/builders. Engine-produced fault exits populate metadata even if some fields are unknown. An embedder-created legacy exit has no asserted metadata. New metadata structures are non-exhaustive with private fields/accessors and constructors.

Use three distinct states: `Knowledge<T> = Known(T) | Unknown | NotApplicable`. `Unknown` never means zero; `NotApplicable` means the field has no meaning for this event. Error codes additionally distinguish **NoCode** from **Unknown** and **Bits { value: u32, known_mask: u32 }**. Require `value & !known_mask == 0`. A full architectural-code accessor returns `Some(value)` only when all 32 bits are justified and the event is an identified guest exception; a bit query returns `None` for unknown bits. Known zero is different from absent or unknown.

| Metadata field | Contract |
|---|---|
| `origin` / `class` | Guest exception with cause; guest-map miss; refused Trap fetch; independent host-access failure. Distinguish fault, trap and unknown class. |
| `vector`, `error_code` | Identified guest exception only; do not infer a guest vector from a host signal number. Preserve existing `Exception.vector`. A canonical flat-map miss may offer a **labelled #PF projection**, separate from a confirmed architectural vector/code. |
| `saved_rip`, `instruction_rip`, `instruction_len` | Saved RIP equals vCPU RIP at this exit. Instruction RIP identifies the reporting instruction, including before #BP/#DB advancement. Length is the decoded instruction's length, never the block length; incomplete/overlength/ambiguous invalid decoding is Unknown. |
| `linear_address` | Address actually reported for the failed memory access/fetch; preserve whether it is an exact byte or only a transfer address. It is CR2-like only for an identified #PF, not for #GP/#SS. Real16 additionally identifies address space and CS:IP rather than relabelling a physical fetch address as linear RIP. |
| `operand_base`, `operand_size` | Optional decoded memory-operand context, separate from failing address; includes inactive vector elements and correct broadcast/narrowing memory width. Never derive base as `si_addr - size`. |
| `access`, `transfer_access`, `transfer_size` | Guest operand intent (including ReadWrite), underlying failed transfer direction, and width requested by that transfer. Keep legacy `AccessKind` unchanged. A rejected fetch byte can report size 1 with a fetch-byte unit; complete instruction size stays Unknown. |
| `progress` | Register/flag state category, memory/device effects and exactness, plus REP restart state where applicable. A convenience “had partial progress” returns Known(true/false) only when proved; otherwise Unknown. |

“Exact” describes a fact within its recorded origin, not a promise that the engine implements every condition of an x86 exception. A **flat #PF projection** contains only map/access facts and is never accepted by the full architectural-code accessor. This lets an embedder apply an explicit guest policy without mistaking host VM state for guest paging state.

Metadata is immutable and valid for the owned `Exit`'s lifetime, including after another run, CPU mutation or memory unmap. Build it before retry cleanup can discard relevant records. No vCPU “last exit” invalidation protocol is required. Non-exception exits and `UnknownInstruction` remain distinguishable and do not acquire an invented exception. Real16 exceptions delivered internally through the IVT are not last-exit metadata; an escaping memory failure during delivery describes that failed delivery access (READ `x86jit-core/src/vm.rs:1141`, `x86jit-core/src/vm.rs:1495`).

### 2. What error-code bits can be known

SDM Vol 3A §4.7 (§5.7) describes **guest paging accesses**, not host memory accessibility. The current map permits useful projection facts but cannot produce a full architectural #PF code. In particular, neither `CpuMode` nor CS selector storage supplies a complete privilege/access model (READ `x86jit-core/src/lift/mod.rs:52`, `x86jit-core/src/state.rs:356`).

| #PF bit | Current knowledge / rule |
|---|---|
| **P, bit 0** | A verified absent byte in the stable guest map supports P=0 **in the flat-map projection**. A span miss alone or a transfer failing region-containment is not sufficient evidence that a particular page is absent; confirm the failed byte. A mapped host-protected address does not prove a guest protection violation: P remains Unknown. Future enforced guest permissions could justify P=1. |
| **W/R, bit 1** | Known for an identified ordinary read or write. Preserve guest RMW intent through lowering; do not blindly translate the first failed host load to guest read. Until an RMW producer has instruction-level provenance and a primary-source witness, leave this bit Unknown for it. |
| **U/S, bit 2** | Unknown today. Describes the access's guest privilege, not `Prot`, the host process's CPL, or 64/32-bit decode mode. Requires an explicit guest access context; adding a CPL/paging model is separate work. |
| **RSVD, bit 3** | Unknown: no guest paging-structure walk or reserved-entry-bit validation. Never fill with zero just because the host did not report it. |
| **I/D, bit 4** | Data accesses justify 0. Fetch intent is exact, but architectural I/D=1 has control prerequisites in §4.7 (§5.7): SMEP, or PAE with NXE. Those controls are not modelled; leave the architectural bit Unknown for fetch. A labelled projection may expose `is_fetch=true` separately. |
| **PK / SS / SGX, bits 5 / 6 / 15** | Unknown: no guest protection-key, shadow-stack or enclave fault model. SS here means **shadow stack**, not the SS segment used to choose #SS. |
| **HLAT, bit 7; other/reserved bits** | Unknown in the partial projection. HLAT is present in the pinned edition. Reserved bits may be known zero only for an identified architectural code with a specified architectural format; do not claim them for a host fault. |

Thus no default “user read = 4 / user write = 6 / fetch = 20” synthesis. An unmapped ordinary data read projection can know P=0, W/R=0, I/D=0 (`known_mask=0x13`, `value=0`), while the complete code remains unavailable. An independent host-access failure may know only direction and address, with no justified P or vector. Keep its error code Unknown; do not use MAPERR/ACCERR as guest P. Trap data accesses are MMIO and have no #PF code; refusing Trap fetch is an engine routing policy, not proof of NX (READ `x86jit-core/src/memory.rs:337`, `x86jit-core/src/vm.rs:1718`).

For other exceptions, SDM Vol 3A §6.13 (§7.13) and §6.15 (§7.15) require cause-specific handling:

| Class | Decision |
|---|---|
| #DE, #UD, #BP, currently emitted #DB, #MF | **NoCode**, not Known(0). Fault/trap class comes from the cause; not every possible #DB is a trap. #MF identifies the waiting/reporting instruction, not necessarily the earlier FP producer (READ `x86jit-core/src/interp/control.rs:17`). |
| #GP / #SS | Emit a full known zero for proven non-canonical/alignment/length-limit causes where defined. Do not default all vector 12/13 events to zero. |
| IRET-related #GP | NT/non-canonical cases can justify zero; selector rejection must preserve its cause and attempted selector or remain Unknown. Current code merges selector mismatch and non-canonical IP into vector 13 without a descriptor model (READ `x86jit-core/src/interp/mod.rs:3289`, `x86jit-cranelift/src/codegen/control.rs:397`). |
| #NP / #TS | No descriptor-driven producers found today. Future producers must supply selector index and EXT/IDT/TI from the actual failing operation; generic/synthetic vector-only exits stay Unknown. |
| #AC | No general guest #AC producer found. A host alignment fault is replayed, not translated to #AC. A future ordinary alignment-check cause can supply zero when justified; do not hardcode zero by vector: the pinned §7.15 Event 17 also describes bus-lock causes with nonzero codes. |
| #PF / other or synthetic vectors | Preserve verified bits/cause if supplied; otherwise Unknown. A vector alone never proves an error-code value or even its delivery origin. Software INT n does not push a code merely because its vector names an exception. |

### 3. Tiers and required bookkeeping

**E** = exact information exists today, though not all is public; **B** = new cold/static bookkeeping needed; **U** = stays unknown without a new model or execution work. This is a capability table, not a claim that the proposed API already exists.

| Execution path | RIP / length / vector | Address / access / width | Code | Progress |
|---|---|---|---|---|
| Interpreter | E saved and current instruction RIP; B valid length/class/cause retention | E scalar size, vector `VecFault.size`, REP `StrFault.elem`; B preserve trap origin and operand context before constructing Exit; address precision varies by miss | E implied zero/no-code for proven causes; B explicit encoding; U paging/privilege and unsupported causes | E architectural restart state; B cold effect summary; U REP entry-count delta, uninstrumented split effects |
| JIT block, explicit cold exit | E saved RIP/vector and compile-time owner; B length/cause descriptor | E cold failed address/direction; B requested transfer width separately from unbacked tail, operand intent/base | Same cause rules; B cold ABI cause/bits | E recovered entry registers/flags for covered instructions; B summary; U untracked store pieces |
| JIT region | Same as block **per sub-block/instruction**, not region entry/end | Same as block; B descriptors survive folding, merging and internal edges | Same as block | Same as block; region fuel/accounting is not a commit count |
| Compiled host fault via decision-13 | E trap-site full-width guest RIP; B decoded length/cause; U guest vector for independent host access | E rebased host fault byte (Access), host transfer width and Read/Write/ReadWrite in table; B retain site identity and guest operand intent; U transfer base unless recoverable | U guest P/U/S/control bits; no host-code forwarding | E recovered registers/flags; opt-in probes prove no RAM bytes; U default split-store effects and actual-transfer faults after concurrent revocation |
| Routed Rust helper / interpreter fixup | E reporting RIP from caller; B helper-specific cause/length | E fixup byte and caller request; B preserve size/origin through helpers that return address only | Same uncertainty as independent host-access failure | E preflight guarantee only under stable protections; B helper effect summary; U concurrent-revocation effects |
| Decode/fetch failure, any tier | E attempted instruction start and first unavailable fetch byte; U complete length; U architectural vector for policy/host failures | E Execute; B reason/precision and explicit fetch-byte unit | B flat projection only for verified map absence; U fetch I/D controls and U/S | E no instruction effects, earlier completed prefix retained |

READ anchors: interpreter's size is dropped at `x86jit-core/src/interp/mod.rs:6475`, `x86jit-core/src/interp/mod.rs:7150`; vector context enriches only MMIO at `x86jit-core/src/interp/mod.rs:6151`; x87/FX helpers lose details at `x86jit-core/src/interp/control.rs:29`. Compiled bounds exits store **remaining unbacked tail**, not necessarily original transfer size (`x86jit-cranelift/src/codegen/mod.rs:3248`); `MemCtx::unmapped_exit` consumes helper operand context and returns only the old Exit (`x86jit-core/src/jit_abi.rs:409`). Do not expose `fault_size` under one misleading universal meaning.

The exact guest-trap-site table already contains owner RIP, direction and **host-transfer width**, including split operands (READ `x86jit-core/src/codemap.rs:27`). Preparation/collection survives optimizer source-location changes and publishes sites for both blocks and regions (READ `x86jit-cranelift/src/lib.rs:57`, `x86jit-cranelift/src/lib.rs:137`). The classifier writes RIP, fault address, width and collapsed direction before landing (READ `x86jit-core/src/jit_abi/fault.rs:306`). Retain an immutable descriptor/site token on this cold path so Rust can recover ReadWrite, instruction identity and static fields without decoding changed guest code. The landing stub itself lives in core, not the backend; it only restores FP control and returns `RET_HOST_FAULT` (READ `x86jit-core/src/jit_abi/landing.rs:8`, `x86jit-core/src/jit_abi/landing.rs:20`). Do not publish a fault address for a host Alignment recovery before replay recomputes it: FreeBSD arm64 deliberately leaves the address out-field untouched (READ `x86jit-core/src/jit_abi/fault.rs:221`).

Build immutable instruction/cause descriptors at lift/compile time; for valid ordinary lifted blocks, successive `InsnStart` addresses and `guest_len` can establish length within each block. Validate inference; overlength/invalid/synthetic IR may remain Unknown. Do not add mandatory fields to public `IrBlock`/`InsnStart` literals just to carry metadata (READ `x86jit-core/src/ir.rs:125`, `x86jit-core/src/ir.rs:3496`). Use private side data and additive internal/backend entry points with default compatibility behavior for external backends. An external backend that supplies only today's ABI gets Unknown fields, not fabricated exactness. Required acceptance tests must cover the built-in interpreter and JIT, including regions.

Keep full details through decode-window and helper errors: `FetchWindow::refill` currently discards failure origin, and `LiftError::DecodeFault` retains only address (READ `x86jit-core/src/lift/mod.rs:120`, `x86jit-core/src/lift/mod.rs:149`). Add an internal detailed result alongside the compatibility API. Preserve the original compiled descriptor when decision-13 replays an unmapped instruction; replay cannot discover already committed compiled-store effects. Mapped host failures return directly; Alignment always replays and may complete with no fault exit (READ `x86jit-core/src/vm.rs:1703`, `x86jit-core/src/vm.rs:1718`, `x86jit-core/src/vm.rs:1741`).

### 4. Partial progress and what committed

Report effects of the **reporting instruction**, not all earlier instructions in a run. Define `progress` with independently qualified register/flag state and memory/device effects. Its memory summary can contain an operand-relative byte/element mask of definitely committed pieces, a mask of uncertain pieces, and scope/completeness. A zero known mask is not a proof that nothing committed. Dynamic-size operations such as REP use restart-state form, not an arbitrarily large bitmap. Report issued/acknowledged device effects separately from RAM writes; MMIO reads may have irreversible device effects too.

For a proved commit-last fault, report entry registers/flags and no committed RAM. Preserve documented exceptions instead of applying this by vector: #MF can report a pending earlier FP event, and #BP/#DB traps have completed-instruction saved RIP. Current memory-fault tests cover entry state and commit-last behavior (READ `x86jit-tests/tests/fault_atomicity.rs:1`, `x86jit-tests/tests/region_fault_state.rs:1`).

REP commits one completed iteration at a time. Report current RSI/RDI/RCX at the decoded address width, DF, element width, operation kind, relevant flags and reporting RIP; these are the exact restart state, including 16/32-bit wrapping semantics. Do not claim that the remaining count equals completed iterations. The loop updates pointers after the successful transfer and decrements RCX afterward (READ `x86jit-core/src/interp/mod.rs:7004`, `x86jit-core/src/interp/mod.rs:7050`, `x86jit-core/src/interp/mod.rs:7118`). The current loop does not retain an instruction-entry count. **Completed-iteration delta and whether any iteration completed in this attempt remain Unknown under the strict zero-added-hot-work decision.** Exact delta would require retaining the initial count in a local/record and validating register pressure/cost; it is not necessary to build a restart frame. A REP `Interrupted` exit remains resumable; this fault-metadata API does not reinterpret cooperative interruption as an exception.

Masked/narrow stores can retain earlier active elements in default mode; routed mode preflights selected destinations. At a failing loop index, an ordered mask and the current operand/mask establish prior completed RAM slots without a per-element journal, provided no unknown fault occurred within one slot. Preflight failure means no new RAM value committed in that attempt, under stable protection; same-value write probes can still affect host dirty/COW tracking. Skip inactive slots and retain their indices. Do not count an MMIO exit as performed before the embedder acknowledges it, and do not lose acknowledgements from earlier retries (READ `x86jit-core/src/interp/mod.rs:7252`, `x86jit-core/src/interp/mod.rs:7291`, `x86jit-core/src/state.rs:626`). Default versus routed device-effect ordering and probe effects are documented separately (READ `backlog/decisions/decision-13 - Signal-context-landing-pads-for-compiled-guest-faults.md:383`). Capture the existing retry ledger before `finish_mmio_retry` can clear it (READ `x86jit-core/src/vm.rs:1211`).

Compiled split-store precision is **opt-in**, selected with
`Vm::set_precise_split_stores(true)`, and **off by default** (decision-13).
Disabled translations retain their existing partial-store behavior and
`PossiblyPartial` metadata. Enabled translations preflight crossing destinations.
Whole-operand map-check failures and one-byte compiled Write probe
faults on ordinary single-destination stores report `MemoryEffects::None`, also
when the dispatcher replays an unmapped probe. Guest instruction identity and
operand width remain distinct from the one-byte Write transfer. The immutable
instruction descriptor restricts this inference to supported single-destination
store forms and the policy used to compile them; it does not apply to arbitrary
multi-destination instructions. Ordinary one-byte stores are not probe sites.
Actual scalar x86-64 transfers without probes retain conservative RAM-effect
metadata. A policy change flushes translations, links and vCPU predictions and
rejects stale background results; forks inherit the policy.

A fault in an actual wider transfer still reports `PossiblyPartial`: another
thread can revoke protection after preflight, including between stores reusing
a dominating preflight within a guest block. Helpers invalidate that reuse except
for dirty-page notification, which cannot change protections. Static site order and memory
rereads cannot reconstruct the committed set under that race. Helper effects
remain conservatively qualified unless the fault evidence proves preflight
failure; REP still describes completed iterations and restart state. No new
exact byte bitmap or device-history guarantee is claimed.

### 5. Cost on the non-faulting path

Default-off split-store precision preserves main's emitted code byte for byte
in the sampled store forms on both ISAs (decision-13). The separate opt-in
preflight has a measured cost: aligned store loops add approximately **1–2.4%**
retired instructions, sliding-address loops **5–9%**, and memcpy/SIMD/MMX remain
within **0.15%**. These are historical measurements of the enabled lowering,
not fresh timing measurements of policy selection. Metadata adds no extra hot
bookkeeping in either mode; enabling execution precision explicitly accepts
the crossing-test work.

**Require zero additional per-access, per-instruction, REP-iteration or compiled-chain instructions/stores for this metadata feature.** Do not add instruction-entry snapshots, progress counters, inline permission lookups, dynamic descriptor stores or a per-call recovery trampoline. Existing checks, recovery routing and FP guards remain existing costs. Static tables grow at compilation; cold branches record cause/descriptor/transfer details only after failure. Preserve every existing ABI offset and its append-only extension rule (`x86jit-core/src/jit_abi.rs:73`, READ). Keep `MemTrap` small and use detailed cold error channels rather than widening every success `Result`.

New cold ABI storage must not introduce mandatory zeroing in `MemCtx::for_memory` on every dispatch. Prefer a documented detailed-failure tag that repurposes existing chain-only out-words (`next_entry` / `link_slot`) for a descriptor token/cold details; those words have no chain meaning on that return. Legacy failure tags decode without reading these details and yield Unknown fields. The dispatcher and standalone `run_compiled` decoder recognize both. A detailed landing return must also guarantee that the classifier initialized its payload. This validity protocol avoids stale data without successful-call stores. If more storage proves necessary, introduce an additive versioned context API with `MaybeUninit` cold out-storage and tagged initialization; merely appending public `MemCtx` fields would still break external struct literals. Include initialization/tag and external-backend compatibility tests.

The signal classifier remains bounded and allocation/lock/TLS free. It copies only cold scalar facts/site identity; allocate the optional box and form masks in ordinary Rust after landing/fixup. There is no per-run metadata reset and no allocation on a successful exit. Faults pay lookup, metadata construction and allocation; cloning an enriched Exit clones the box. Default synthetic exits pay no allocation.

MEASURED: current `Exit` is **32 bytes**, `MemTrap` **1 byte**; a payload-equivalent enum with the proposed optional boxes is also **32 bytes** on this host/compiler. This is a layout feasibility probe, not an ABI or latency guarantee. Existing decision-13 measures zero added compiled-call-edge instructions; its benchmark numbers are historical READ evidence, not new measurements here. Implementation acceptance requires generated-code comparison and representative timing: code layout/compiler changes can still affect performance even when no hot instruction is intentionally added.

Strict zero added work and an exact journal of every partial write / REP entry delta cannot both be promised with current execution semantics. This proposal chooses exact restart state and explicit Unknown where history is absent. If “exact metadata” means a complete #PF code and exact committed bytes even under concurrent revocation, implementation must wait for the missing guest model and stronger synchronization; this record does not weaken that completion criterion silently.

## Consequences and migration for embedders

Existing matches using `..`, constructors, fault handlers, MMIO completion and retry remain valid. Embedders opt into inspection through the returned Exit, preserve the owned event as needed, and stop reading metadata before mutating/retrying only if they also need the live CPU state. Metadata itself survives those operations. A full signal frame still needs a CPU snapshot taken at the exit; metadata is not a replacement register image.

Consumers must handle NoCode, known zero, partial bits and Unknown separately. Use the full-code accessor only for an identified architectural code; an unknown bit cannot be cleared by `unwrap_or(0)`. Embedders needing a complete guest #PF may combine labelled map facts with their **explicit** privilege/paging policy outside core, or retain an unavailable diagnostic. A host-protection callback/barrier must be resolved according to embedder policy before deciding to deliver a guest exception. No new host signal decoding or guest permission enforcement is implied.

Custom backends and standalone callers may use legacy constructors and receive absent/Unknown details, or adopt enriched cold result construction. Keep `CpuState` layout unchanged and public raw-ABI construction compatible; use tagged out-word aliases or an additive versioned context as above. Do not alter public `GuestTrapSite` struct literals merely to attach descriptors: maintain parallel immutable metadata keyed by site identity. Preserve Exit's existing Send/Sync properties and const constructors. Update spec §5.2/§8 and public examples when accepted; retain the permission/segmentation and SIMD-exception deferrals (READ `backlog/docs/deferred.md:175`).

## Validation and test plan

MEASURED in the scratch HEAD archive: `address_exceptions` **49**, `decode_faults` **23**, `fault_atomicity` **41**, `fault_landing_pad` **20**, `rust_memory_faults` **15** tests passed (**148 total**). The filtered backend `guest_trap_sites` suite passed **16** tests; two core noncanonical-map tests passed. These witness existing behavior, not the unimplemented metadata API. Signal adapters for other OSes may have pure-data tests here; this does not establish runtime validation on those hosts.

Two additional scratch-only probe tests passed (`wt/x86jit-tests/tests/metadata_design_probe.rs`; outputs in `probe-results.txt`). Across interpreter, JIT block and a verified compiled region: storing `0x12345678` into an RX region returned Hlt and changed bytes; out-of-span `48 83 03 01` (ADD qword [RBX],1) returned access Read; `f0 48 83 03 01` (LOCK ADD) returned Write. RIP stayed at the reporting instruction. This proves legacy access direction is not enough to reconstruct guest intent. The second test measured enum sizes. Trap-site output is in `trap-site-results.txt`.

Reproduce with `RUSTC_WRAPPER='' CARGO_TARGET_DIR=../target cargo test --offline` in `wt`, adding `-p x86jit-tests --test address_exceptions --test decode_faults --test fault_atomicity --test fault_landing_pad --test rust_memory_faults`; use `-p x86jit-tests --test metadata_design_probe -- --nocapture` for the scratch probes, or `-p x86jit-cranelift --lib guest_trap_sites` for site witnesses. The empty wrapper bypasses an unavailable sandboxed sccache. The compiler emitted four existing runtime-symbol signature warnings in `libcalls.rs`; tests completed successfully.

Required implementation tests, parameterized over interpreter/block/verified-region and both Rust recovery modes where applicable:

1. **Exception classes:** #DE zero/overflow, #UD invalid encoding, overlength #GP(0), noncanonical #GP(0)/#SS(0), required-alignment #GP(0), #BP/#DB saved-versus-reporting RIP, pending #MF reporting instruction. Test NoCode versus Known(0), and Unknown for unsupported/synthetic #NP/#TS/#AC/selector-dependent #GP; do not invent producers to claim architectural coverage. Add cause-specific selector-code witnesses when descriptor support exists.
2. **Memory:** known map holes, out-of-span and wrapping references, scalar/vector/helper and broadcast/narrow widths, instruction fetch split at region/page/Trap boundaries, mapped host PROT_NONE/read-only faults. Assert reported byte precision, requested transfer width versus operand width versus unbacked tail. Preserve all Unknown code bits; test the P projection independently from host reason and `Prot`. Test RMW guest intent, failed compare-exchange and source/destination sides of MOVS; obtain primary-source/native x86 witnesses before marking ambiguous W/R exact.
3. **Progress:** read-before-write RMW fault; PUSH/POP/call state; first/later REP faults, source/destination fault, DF and address widths; masked holes, all-zero masks, RAM/MMIO mixtures and acknowledged retries. Split 16/32/64-byte and page-crossing scalar stores must report no RAM effects at a proven preflight fault; actual-transfer faults retain uncertainty, including across replay. Exercise preflight before device writes and document stable-protection assumptions.
4. **Recovery:** exact sites versus adjacent PCs, optimized/folded accesses, RIP above 4 GiB, multiple sub-blocks, absent recipe, nested frame and unsupported signal rejection. Ensure classifier allocations stay zero and all host recovery state remains preserved. Run actual landing/fixup tests on each supported host/OS; cross-compilation alone does not validate delivery.
5. **API and cost:** compile a downstream consumer using legacy patterns/constructors, including consts and custom backends; owned metadata remains valid across retry/unmap and no successful Exit inherits a prior fault. Check layout on supported targets; compare hot machine code and interpreter success loops, check no new allocation on success, then interleave representative dispatch/load/store/REP timings. Do not infer zero cost from timing noise alone.

## Resolved questions and trigger to revisit

The maintainer accepted the recommended resolution of each open question (2026-11-06):

- **Ownership:** metadata is owned by the exit, as optional boxed data on the existing variants, read through `Exit::fault_metadata()`. It is allocated only after fault recovery, never in signal context. No vCPU query with a lifetime rule.
- **#PF bits:** partial facts now. Architectural bits the engine cannot know are reported as Unknown, and the flat-map projection is labelled separately from a confirmed architectural code. A guest privilege/paging/permission model is not a prerequisite.
- **Partial progress:** exact restart state is enough. Per-attempt deltas and an inventory of committed bytes are out of scope; crossing compiled stores now use decision-13's preflight guarantee, with concurrent-revocation exceptions retained.
- **IRET selector failures:** an Unknown-code cause until descriptor support exists.

Revisit this decision when paging/privilege/descriptor enforcement ships, store synchronization changes, a consumer requires exact REP entry deltas, or fault-path allocation becomes measurably material. Acceptance of this design is followed by implementation and per-class tests; it is not acceptance of fabricated completeness.


## Amendment 2026-11-06: raw host fault record

Identity-mapped embedders can map their entire guest span as RWX RAM and keep
protections only in the host page tables. The flat-map projection cannot observe
those protections; the synchronous host signal carries the relevant evidence.

`FaultMetadata::raw_host_fault()` optionally exposes a fixed-size `RawHostFault`,
labelled with its host OS and architecture. Its fields are independently optional:
trap number, host error code, fault address (CR2/FAR), ESR, signal number, signal
code and signal address. Linux/FreeBSD x86-64 adapters copy the saved trap/error/
CR2 fields. AArch64 adapters copy ESR and FAR where exposed; FreeBSD's current
mcontext exposes neither and retains signal address/code only. The signal address
is retained separately because it can name a PC for some host fault kinds.
These are opaque, host-sourced values, **never a confirmed guest #PF code**.
Neither the architectural-code accessor nor the flat-map projection interprets them.

The additive `classify_with_raw_fault` entry point copies this record into the
existing per-thread recovery snapshot on a compiled landing, with no allocation.
The dispatcher attaches it to the exit-owned metadata after returning from the
handler, including when replay produces the exit. Legacy `classify` leaves it
absent; transfer fixups do not themselves produce compiled landing metadata.
There is no successful-access instrumentation or per-run metadata reset.
The guest linear address and, where reconstructible from the immutable lowered
address operations, guest operand start remain separate from the host address;
page-crossing splits can therefore report both addresses. Unavailable guest
operand reconstruction remains explicitly Unknown. Signal adapters own ABI
extraction; core retains its dependency boundary and does not install handlers.
