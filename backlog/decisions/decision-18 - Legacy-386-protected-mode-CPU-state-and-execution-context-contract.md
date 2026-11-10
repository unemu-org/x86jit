---
id: decision-18
title: 'Legacy 386 protected mode: CPU state and execution-context contract'
date: '2026-11-10'
status: accepted
---

# Legacy 386 protected mode: CPU state and execution-context contract

## Context

Epic **xj-srh** contains thirteen requests from a Real16 embedder (unemupc98).
Its **protected-mode-gaps.md (unemupc98)**, commit `7ac366d`, sections “Probe
results” and “x86jit requests”, reports an entry stopping at LGDT, 28 unsupported
forms among 33 isolated probes, refusal of Compat32 `run_steps`, and manual IRQ
injection into Compat32 constructing a six-byte IVT frame. These are the
embedder's measurements at `96a49a9`, not protected-mode execution evidence.
This record fixes the contract required by **xj-srh.1**. The maintainer accepted it
on 2026-11-10. No code, build, or new execution measurements accompany it.

READ at `bf4c8f6`: `lift::CpuMode` has exactly `Long64`, `Compat32`, `Real16`.
It selects decoder bitness, IP/SP wrapping, segment lowering and backend policy.
`Vm`/`Vcpu` keep it fixed; `IrBlock.mode` and `BlockKey { guest_addr, mode,
mxcsr_mode }` carry it through lifting, interpretation, caching and compilation.
Real16 fetch separates physical address from decode IP but the key contains only
the former. Its scalar lowering tests `Code16` to select selector-shift addressing;
its stack and control helpers also assume 16-bit real mode. Compat32 uses flat
addresses apart from configured FS/GS bases. Neither path is descriptor-backed.

READ: `CpuState` already contains all six visible selectors and
`rflags_system: u64`, including storage for IOPL and NT; it has no descriptor
caches, table registers or CR0/CR2/CR3. `vm.rs::run_steps` implements
[decision-16](decision-16%20-%20Instruction-exact-step-budget-and-interrupt-acceptance-for-the-Real16-interpreter.md)
only for Real16. `irq_ready` excludes other modes, but `run_inner` can consume a
manually queued vector through `interp::deliver_interrupt` without that mode
restriction. That helper constructs a real-mode frame. Comments restricting caller
usage are insufficient protection against the measured injection error.

The relevant design seams are spec.md §§3, 6–7, 8.2.1–8.2.2, 9 and 17. This
proposal activates the requested protected-segmentation seam in deferred.md; it
does not activate a legacy JIT tier or introduce execution-mode traits.

### Primary-source sections

Checked in the pinned Intel SDM, **325462-092US, June 2026**, using local text
extraction. This edition puts paging in chapter 5, protection in chapter 6,
interrupt delivery in chapter 7 and initialization in chapter 12. Section numbers
below refer to this edition, not older editions' chapter numbering.

| Authority | Hardware fact used by this proposal |
|---|---|
| SDM Vol 3A §2.5; §§12.9.1–12.9.2 | CR0.PE selects real/protected operation, CR0.PG enables paging; PG cannot be enabled with PE clear. PE changes and subsequent segment reloads are distinct operations. |
| SDM Vol 3A §§3.4.2–3.4.3, §3.4.5 | Segment selectors and hidden base/limit/access information are distinct. Cached descriptors survive edits to their table entries until a reload. CS.D selects default operand/address sizes; SS.B selects implicit stack-pointer size independently. |
| SDM Vol 1 §6.2.3; Vol 3A §3.4.5 | Explicit operand addressing, implicit stack addressing and transfer width are separate attributes; prefixes do not replace SS.B. |
| SDM Vol 3A §§2.4, 3.5.1, 10.2.4 | GDTR/IDTR hold table base/limit; LDTR/TR also retain selectors and cached descriptor state. |
| SDM Vol 3A §6.5; Vol 1 §§3.4.3.3, 21.1.2 | CPL is distinct from DPL and IOPL. EFLAGS includes IOPL and NT; earlier CPU generations have different identification-flag behavior. |
| SDM Vol 3A §§5.3, 5.6–5.8, 5.10.4.1 | Two-level 32-bit paging, access permissions, CR2/page-fault error information, A/D effects and translation invalidation. Later paging extensions in these sections are not automatically a 386 feature set. |
| SDM Vol 3A §§7.6, 7.8.1, 7.8.3, 7.12–7.13; §7.15, Event 8 | Restart boundaries, IF and inhibition, protected interrupt/trap gates, error codes and double-fault/shutdown escalation. |
| SDM Vol 3A §§6.8.3–6.8.6, 7.12.1, 10.2.1, 10.2.4; Vol 1 §20.5 | Call/interrupt privilege stacks, TSS state and I/O-permission checks. |
| SDM Vol 3A §§12.1.4, 12.9.1–12.9.2 | Reset fetch uses hidden CS state; PE changes do not justify synthesizing all segment caches from visible selectors. |
| SDM Vol 3B §23.1.4 | Real-address interrupt-table entries and delivery frames. |

These sources license hardware semantics; the representations, cache policy and API
refusals below are engine decisions. Where the SDM is silent or undefined, consult
the pinned **AMD64 APM** as a second witness and cite its volume/section before
implementing a resolution. The CPU APM is the stash's `intel/amd64-apm.pdf`;
`amd/` contains GPU manuals. AMD64 APM Vol 2 §4.1 is the second witness for
retention of hidden limits/attributes during real-mode segment reloads;
§14.6, Table 14-4, witnesses legacy default sizes selected by cached CS.D.
These clarify the retained-cache representation, not complete 386 model fidelity.
The **80386 Programmer's Reference Manual** is only a possible second witness for
386-specific behavior: it must first be pinned and audited in `oracles/`.
Other emulators' implementations do not supply semantics.

## Decision

Adopt the following contract.

### 1. Separate application contexts from legacy CPU execution

Keep `CpuMode` and its existing variants/discriminants as the fixed application
decode contexts. Add an explicit, immutable VM execution-profile choice:
`ExecutionProfile::Application(CpuMode)` or `ExecutionProfile::LegacySystem`.
Existing constructors and `set_cpu_mode` select Application with their present
defaults and behavior. A new `Vm::set_legacy_cpu_model(...)`, before vCPU creation,
selects LegacySystem and a CPU-model profile. A running legacy CPU never changes
the VM's application mode or becomes `CpuMode::Compat32` upon setting PE.

The model profile supplies `CpuGeneration` (`#[repr(u8)]`: `I8086`, `I286`,
`I386`) and an immutable `model_semantics_id: u64`. Generation governs instruction
availability and model policies; it is neither decode width nor a CPUID display
family. **xj-50w owns the CPU-model catalogue**: use one shared model-profile
authority and a generation projection, not a second preset registry in xj-srh.
Existing `GuestCpuFeatures`, CPUID tables and overrides remain separate feature/
enumeration facilities. This decision needs no completion of xj-50w's catalogue;
it establishes the shared profile boundary. Context derivation covers all three
generation tags; this epic delivers the I386 system path and preserves the existing
Real16 personality, without claiming complete 8086 or 286 system emulation.

For LegacySystem derive a value `LegacyDecodeContext` at dispatch from the model
and CPU state. Fields are `generation: CpuGeneration`, `regime: LegacyRegime`
(`Real`/`Protected`), `default_operand_bits: u8`, `default_address_bits: u8`,
`stack_address_bits: u8`, `cpl: u8`, and `paging_enabled: bool`.
PE/PG come solely from CR0; CS.D and SS.B come solely from hidden attributes.
Do not store writable duplicate booleans or infer regime from decoder bitness.

| Legacy state | Derived context |
|---|---|
| Ordinary real entry, PE=0, normal CS/SS caches | Real, defaults 16/16, stack 16, CPL 0, paging off. |
| PE=1, CS.D=0 | Protected, defaults 16/16; stack 16 or 32 from SS.B; CPL from current CPU privilege state. |
| PE=1, CS.D=1 | Protected, defaults 32/32; stack independently 16 or 32; CPL from current CPU privilege state. |
| PE/CS transitions with retained caches | Use the retained caches, including their size attributes; do not reset them merely because PE changed. |

The size/cache rules follow SDM Vol 3A §§3.4.3, 3.4.5, 12.9.1–12.9.2,
with AMD64 APM Vol 2 §§4.1, 14.6 as the second witness for retained size attributes;
entry CPL follows §12.9.1. Validate generation-specific combinations against the
selected model before execution. I8086 cannot be configured as a protected CPU;
I286 does not acquire I386 forms merely through a 32-bit decoder setting
(SDM Vol 1 §§2.1.1–2.1.3). Reject unsupported configurations explicitly.

Iced receives the derived default decode width; instruction prefixes determine the
effective operand/address widths. Shared lowering must receive context explicitly:
`Code16` alone must never imply real-mode addressing. Legacy `rip: u64` holds the
architectural offset, not a physical fetch address. Preserve the distinction between
instruction/target operand width and default decode width when computing EIP; no
blanket 16-bit mask for every CS.D=0 transfer. Central stack helpers select SP/ESP
from SS.B and transfer bytes from the instruction's operand size (SDM Vol 1
§6.2.3; Vol 3A §3.4.5). Segment lowering retains the segment identity and full
effective offset until checks finish, then forms a linear address; fetch, scalar,
string and implicit stack accesses use the same legacy access boundary.

Append `legacy_context: Option<LegacyDecodeContext>` to `IrBlock`: `None` preserves
the application path; `Some` is authoritative for legacy interpretation. Retain the
public `mode` field as application/decode compatibility metadata, never as a legacy
regime or stack-size input. Select the legacy interpreter walk outside its instruction
loop. Materialization and region formation must reject `Some` before backend entry.

Application Long64/Compat32 keep their current flat/TLS lowering and backend paths.
Application Real16 keeps its current interpreter path. They are not reconstructed
from synthetic CR0/descriptor values. IA-32e entry, EFER, CR4 and a general 64-bit
system model are outside this decision; accepting LegacySystem is explicit opt-in.

### 2. Append state, with one owner for every architectural value

Append **`legacy: LegacyCpuState` after `x87_pointer_model`**, the present last
`CpuState` field. Use `#[repr(C)]` on the new aggregate and its nested records;
all new discriminants/boolean images use fixed-width `u8` storage. Reserve the
following storage in xj-srh.1; later children activate it, rather than moving fields.

| Record/field | Concrete representation and ownership |
|---|---|
| `TableRegister` | `base: u32, limit: u16, reserved: u16`; engine layout, not a packed guest operand image. |
| `SegmentCache` | `base: u32, limit: u32, attributes: u16, unusable: u8, reserved: u8`. Limit is expanded to an inclusive byte limit. Attributes store the descriptor access byte in bits 7:0 and descriptor flags AVL/L/D-or-B/G in bits 12:15; bits 11:8 are zero. `unusable` is an engine validity marker. |
| `SystemSegment` | `selector: u16, reserved: u16, cache: SegmentCache`; LDTR/TR need both visible and cached state. |
| `LegacyCpuState` prefix | `cr0: u32, cr2: u32, cr3: u32, gdtr: TableRegister, idtr: TableRegister, ldtr: SystemSegment, tr: SystemSegment`. |
| `LegacyCpuState` segment array | `segments: [SegmentCache; 6]`, indexed by a named `SegmentId` in CS, DS, ES, SS, FS, GS order. Existing `cpu.cs/ds/es/ss/fs/gs` remain the sole visible selectors. |
| `LegacyCpuState` tail | `cpl: u8, a20_enabled: u8, reset_fetch: u8, reserved: u8`. CPL validated in 0..=3; other bytes are 0/1. |

The descriptor representation follows SDM Vol 3A §§3.4.3–3.4.5; table/system
register contents follow §§2.4, 3.5.1, 10.2.4. `cpl` is explicit execution state,
updated with CS/SS transfers and checked against their selector privilege bits;
it is not the descriptor DPL, including for conforming code (§6.5).
Generation/model identity belongs to immutable VM/vCPU configuration and is included
in checkpoints; it is not another writable architectural register in `CpuState`.

Reuse `rflags_system` for IOPL (bits 13:12) and NT (bit 14), and existing
`flags.if_` for IF; add no second EFLAGS image and do not enlarge/reorder `Flags`
(SDM Vol 1 §3.4.3.3). Model-aware FLAGS/EFLAGS helpers own masks and privilege
rules. Stored VM/TF bits do not enable V86/debug execution: legacy entry or a return
requiring those unsupported facilities must fail explicitly. NT storage/round-trip
does not license a task return; encountering one yields a typed unsupported-system
exit, not an ordinary IRET (SDM Vol 3A §10.3; Vol 2A §3.3, IRET/IRETD/IRETQ).

In LegacySystem the six caches own segment bases, including FS/GS. Existing
`fs_base`/`gs_base` keep their application/TLS meaning and are not competing inputs
to legacy addressing. A cache is not refreshed when guest GDT/LDT memory changes
(SDM Vol 3A §3.4.3). Direct selector writes between legacy runs do not silently
rebuild hidden state: an embedder must use a validated state setter for a coherent
selector/cache pair. Restore preserves the actual hidden cache, not a descriptor
re-read. PE changes preserve caches until the specified reload (§§12.9.1–12.9.2).

Default storage is inert for Application. A legacy ordinary-entry constructor
installs a coherent real context, paging off, normal selector-derived caches,
IDTR base 0/limit `0x03ff`, `a20_enabled=1`, `reset_fetch=0`; this is an engine
entry preset, not hardware reset (real table format: SDM Vol 3B §23.1.4).
xj-srh.11 gives A20 its physical-access meaning. xj-srh.12 supplies explicit reset
initialization; hidden CS base plus RIP represent its high fetch address
(SDM Vol 3A §12.1.4). `reset_fetch=1` records that origin until a successful CS
reload; it never overrides the cached base. No device-controller policy or reset
side effects are implemented merely by reserving these bytes.

All existing offsets and `jit_abi` field/return-code meanings are frozen. Do not
insert fields, reuse padding, change an existing type/alignment, or place an
over-aligned new member that shifts the prefix. New `CpuOffsets`, `MemCtx` fields
or return discriminants, if eventually needed, append only. The legacy interpreter
needs no new compiled offsets now. This preserves offsets, not `size_of::<CpuState>()`
or a portable raw-memory serialization ABI.

### 3. Translation identity, guards and mode-switch boundaries

Keep Application `BlockKey`, constructors, hash stream and compiled fast paths
unchanged. Use a separate interpreter-only legacy cache namespace with
`LegacyBlockKey { owner_id: u64, context_epoch: u64, model_semantics_id: u64,
decode_bits: u8, stack_bits: u8, regime: LegacyRegime, decode_ip: u32,
linear_fetch: u32, mxcsr_mode: u32 }`. `owner_id` is a unique vCPU identity;
no cross-vCPU legacy block sharing is required. Model identity includes generation
and every model policy that changes lifting. Both decode IP and linear fetch belong
in identity: two CS mappings can fetch the same bytes yet resolve relative branches
and restart locations differently. A physical address alone is insufficient.

Maintain a vCPU-local `LegacyContextStamp` and monotone `context_epoch`. Compare
the stamp at run/step API entry to catch public between-run state edits; update it
through centralized in-run setters. The stamp includes CR0's supported controls,
CR3, CPL, all selectors/caches, GDTR/IDTR/LDTR/TR, IOPL/NT, A20/reset state and model
identity. Any change increments the epoch and discards that vCPU's legacy blocks
and derived access caches; in-run changes end the current block. CR2 is an
output/register value read at runtime, not a
translation assumption. IF and STI/SS shadows are live boundary state; they are
not key fields. MXCSR keeps its existing RC/FTZ/DAZ specialization policy.
Epoch rollover discards the namespace before reuse. A checkpoint restores neither
owner identity nor derived epochs. This deliberately conservative guard covers
runtime permissions and cached descriptor attributes without putting every field
directly in a large key; later optimization needs a separate witness-backed change.

Every successful MOV CR0/LMSW affecting execution, CR3 write, segment reload,
far transfer, IRET and interrupt/exception delivery terminates the current legacy
block and returns to dispatch before another instruction. Commit state only after
required validation/accesses; a fault does not install the proposed new context.
A PE write does not reload CS or select 32-bit decoding; the following far JMP
loads CS independently (SDM Vol 3A §§12.9.1–12.9.2). Successful SS loads change stack
context independently of CS (§3.4.5). Recompute context after delivery/return, carrying
decision-16's completion and shadow state across the boundary. A same-value CR3
write still invalidates derived translations (§5.10.4.1).

xj-srh.1 establishes a CPU-aware legacy access interface: segmented offset → checked
linear address → CPU translation → physical bus address → existing RAM/MMIO map.
Initially translation is identity; xj-srh.7 supplies paging and physical fetch-page
dependencies. Never teach shared `Memory` one vCPU's CR3. xj-srh.11 applies A20 at
the physical boundary, including page walks, as an explicit engine gate contract.
Its hardware/model details require source-backed witnesses before activation.

Initially use no software TLB. With paging, revalidate fetch mappings/permissions
and physical code-page versions before executing each cached instruction that
depends on them, including both pages of a crossing instruction. Discard stale
IR on changed bindings; do not turn a fault beyond the completed prefix into an
early exit. Track SMC spans by physical backing so aliases invalidate every affected
legacy block; RAM/MMIO splits and page walks use the same physical map. Ordinary
PTE writes do not refresh hidden segment caches. No legacy chaining, regions,
background tier-up or compiled predictor entry may bypass these checks.

### 4. Execution, stepping and event delivery

| Profile/regime | Run/step contract | Queued manual maskable IRQ |
|---|---|---|
| Application Real16 | Existing interpreter and decision-16 API. | Existing real delivery and readiness. |
| Application Compat32/Long64 | Existing execution; `run_steps` remains `UnsupportedMode`, even at zero budget. | Never invoke the IVT helper. A positive run with a queued vector returns a typed unsupported-system exit before guest effects, retaining the vector. |
| Legacy real, protected USE16 or USE32 | Interpreter only with every backend. xj-srh.5 extends `run_steps`/`StepOptions` uniformly across transitions. | Queue with the existing `inject_irq(u8)` API; choose delivery from current architectural regime at acceptance, not at enqueue. |

Keep `inject_irq`'s existing single-vector overwrite behavior and signature; its
queue operation itself neither acknowledges an external controller nor constructs
a frame. `irq_ready` remains false for Application Compat32/Long64 and for legacy
contexts whose delivery seam has not landed. Where supported it uses decision-16's
IF/shadow/completion predicate. Invalid gate contents can fault on delivery; readiness
does not pre-walk the IDT. IF/inhibition rules are SDM Vol 3A §§7.8.1, 7.8.3;
completion blocking is an engine contract. Guard the existing application misuse
in xj-srh.1, before enabling any new protected execution.

xj-srh.5 preserves decision-16's zero-budget non-effects, completed-instruction/
REP-iteration units, retirement relation, pending-IN refusal, MMIO retry ownership,
exit precedence and optional ready stop. Extend restart/index/count handling to
effective 16/32-bit widths; switching PE, CS.D or SS.B does not reset counters,
pending vectors or shadows. Delivery itself costs zero steps; a completed software
INT costs its instruction unit, a fault does not (SDM Vol 3A §7.6). An unfinished
device operation must finish before injection. No fallback from legacy system
execution into the application JIT is permitted.

Use one internal `LegacyEvent { vector: u8, source: EventSource, saved_ip: u32,
error_code: Option<u32> }` path for INT, INTR and CPU exceptions. Preserve fault/
trap origin, exact restart location and error information through existing outward
fault metadata. Real delivery uses the real-address table/frame rules
(SDM Vol 3B §23.1.4), including `IDTR.base + 4 * vector` and table-limit checking
on LegacySystem once .3 activates IDTR. Application Real16 retains its existing
delivery contract. Protected delivery uses IDTR interrupt/trap gates, with gate
width independent of interrupted CS.D (SDM Vol 3A §§7.12–7.13).

Before xj-srh.6, xj-srh.4 generates precise protected faults and returns them to
the embedder; unavailable IRQ/gate delivery returns `Exit::UnsupportedSystemContext`
with a typed operation reason, never a real IVT frame or a fabricated #UD. Before
the corresponding consumers land, legal PE/PG-enabling writes also return this
unsupported-engine exit without committing; architecturally illegal PG-without-PE
still generates #GP (SDM Vol 3A §2.5). Unsupported engine scope is distinct from
a hardware fault. xj-srh.6 routes same-CPL events internally and implements
architectural escalation through #DF to typed `Exit::Shutdown`, preserving failure
evidence (SDM Vol 3A §7.15, Event 8). xj-srh.9 adds privilege-changing delivery,
call gates and outer returns; until then a required privilege transition is a typed
unsupported-system exit. Host mapping/MMIO exits remain embedder handoffs, not
automatically guest #PF; only CPU paging generates that exception (§5.7).

## Implementation phases (bead briefs)

Storage reservation is xj-srh.1; “owns” below identifies the child activating a
field's behavior. Each seam ships whole with its own witnesses and the full gate.
This is a dependency contract, not another task board.

| Child | Dependencies and owned field/behavior |
|---|---|
| **xj-srh.2** | After .1: I386 real forms, FS/GS, 16:32 transfers, PUSHFD/POPFD/PUSHAD/POPAD, full effective-offset checking and model-aware identification flags; preserve the existing Real16 personality (SDM Vol 1 §21.1.2). |
| **xj-srh.3** | After .1–.2: activate GDTR/IDTR and CR0/CR2/CR3 primitives, SMSW/LMSW/CLTS, operand images and legality; legacy real delivery consumes IDTR base/limit (SDM Vol 3B §23.1.4). PE/PG effects remain refused until .4/.7, not successful no-ops. |
| **xj-srh.4** | After .1–.3: activate all six caches, PE/regime and CS.D/SS.B/CPL context changes, same-CPL segment loads and far transfers/IRET; presence/type/limit/accessed checks and precise outward faults (SDM Vol 3A §§3.4.3–3.4.5, 6.8, 12.9). |
| **xj-srh.5** | After .4: exact stepped execution in both legacy regimes, mixed sizes, counters, REP, completion and persistent shadows; system execution stays interpreted. |
| **xj-srh.6** | After .4–.5: activate protected IDTR delivery, INT/INTR/IRET, 16/32-bit gates, error codes, #DF/shutdown; no protected IVT fallback (SDM Vol 3A §§7.12–7.15). |
| **xj-srh.7** | After .1, .3–.6: activate PG/CR3 translation, CR2/#PF, 4-KiB two-level walks, P/U/S/R/W and A/D, physical aliases/SMC/MMIO and fetch validation (SDM Vol 3A §§5.3, 5.6–5.8). No TLB optimization. |
| **xj-srh.8** | After .4, .6: activate LDTR selector/cache, LLDT/SLDT and descriptor queries/ARPL (SDM Vol 3A §§2.4.2, 3.5.1, 6.10). |
| **xj-srh.9** | After .6, .8: activate TR selector/cache, LTR/STR/busy state, TSS privilege stacks, call/interrupt transitions and outer IRET (SDM Vol 3A §§6.8, 7.12.1, 10.2). |
| **xj-srh.10** | After .5, .9 and scalar/string-I/O support from **xj-42k**: enforce CPL/IOPL and TSS I/O bitmap before exits/effects; complete privilege rules for CLI/STI/HLT/POPF/IRET (SDM Vol 1 §20.5; Vol 3A §6.9; Vol 2A §3.3, CLI/HLT/IRET; Vol 2B §4.3, POPF/STI). |
| **xj-srh.11** | After .1, .7: activate `a20_enabled`, physical gating and split/alias/cache coherence; external control policy stays in the embedder. |
| **xj-srh.12** | After .1: activate `reset_fetch`, reset hidden CS and sparse high physical backing; bounded high fetch and first-CS-reload witnesses (SDM Vol 3A §12.1.4). |
| **xj-srh.13** | After .5–.11, and .12 for complete reset continuation: export/restore selectors/caches, system registers, model identity, A20/reset state and vCPU vectors/shadows/counters. Restore flushes derived state; no persistent machine-file format. |

.2 and .12 can ship independently after .1. After .6, .7 and .8 are independent
branches; .9–.10 do not require paging, while .11 does. .3 can ship its primitive
subset before PE execution; .4/.5 can ship before internal IDT routing with the
explicit outward-fault/refusal contract. Until .10, do not expose unrestricted
nonzero-CPL execution: .9 may witness transitions internally but public execution
must refuse before an unenforced privileged/I/O operation. Architectural legality
checks required by a child are never postponed as permissive temporary behavior.

## Alternatives considered

| Choice | Assessment |
|---|---|
| Add `Protected16`/`Protected32` to `CpuMode` and switch that enum | **Rejected.** Still conflates generation, regime and two independent size attributes; misses retained-cache states and encourages sharing application translations. |
| Turn Compat32 into the system protected mode | **Rejected.** Changes flat application contracts, JIT code and stepping/IRQ behavior; cannot express USE16 or an independent SS.B. |
| Derive all application modes from one architectural CR0/EFER/descriptor model | **Rejected.** Expands this epic into IA-32e system emulation and risks every compiled fast path. Keep the explicit profile split. |
| Recompute hidden state from selectors/tables on every use or restore | **Rejected.** Loses cached descriptor and reset state (SDM Vol 3A §§3.4.3, 12.1.4). |
| Key only by fetch PA plus decoder bitness; flush on PE alone | **Rejected.** Misses decode IP, stack width, model, privilege, mapping and descriptor changes. |
| Key every descriptor and control bit; share legacy IR across vCPUs | **Rejected for this epic.** Larger keys and more reuse proof than needed; a private namespace/epoch guard is sufficient and concrete. |
| Put system state behind a pointer or in a side table | **Rejected.** Direct fixed-width appended state makes snapshots and ownership inspectable; no allocation or pointer lifetime is needed for these registers. |
| Add legacy JIT support or a generic mode/addressing framework now | **Rejected.** Unrequested scope and codegen risk; central explicit context/access helpers suffice. |

## Consequences and risks

Legacy system execution becomes CPU-local and state-derived without changing fixed
application contexts. Conservative cache flushing and interpreted paging cost
performance and can retain less reusable IR; no legacy throughput target is implied.
Appending the aggregate increases every `CpuState` size, including application
states, so allocation/cache effects still need benchmarks even with identical JIT
bytes. Public mutable state creates inconsistent-state risks: validate at API
entry and provide coherent setters; never silently repair guest caches. Faulting
multi-access instructions and event delivery remain the highest correctness risks.

Modern manuals contain extensions absent from a 386. The model must select exact
CR0/FLAGS masks, paging permission behavior and reserved-bit rules rather than
inherit a modern default. Any unresolved model-specific fact blocks that behavior's
implementation until a cited primary/second witness and a value test exist.
Full hardware task switches/task gates/NT task returns, **V86 mode**, **DR/TF
debugging**, and **386/387 control/error integration** remain out of scope. Storing
their related bits is not execution support. A Real16 JIT and IA-32e system entry
also remain deferred.

## Witness tests required before implementation acceptance

1. **Context isolation (.1).** Table-test generation × PE × CS.D × SS.B × CPL × PG,
   valid/invalid configurations, retained caches and both mixed code/stack widths.
   Same bytes in real, protected USE16 and USE32 produce the selected semantics;
   profile selection never changes existing Application defaults or enables JIT.
2. **ABI offset witness (.1 and every child).** Retain
   `mxcsr_is_appended_and_existing_cpu_offsets_are_unchanged`,
   `if_flag_uses_pad_byte_and_keeps_xmm_at_160`, `memctx_offsets_match_layout` and
   `retstack_offsets_match_layout` in `jit_abi.rs`. Before implementation, freeze a
   literal offset/size/alignment manifest for **every existing CpuState member and
   nested Flags member**, every `CpuOffsets` entry, MemCtx field/constant and RET
   value on each supported host ABI. Compare `offset_of!`/measured offsets to that
   pre-change manifest, not two values recomputed from the changed layout. Existing
   fixed examples include RIP=128, FS/GS bases=136/144, Flags=152, IF=159, XMM=160,
   YMM-high=672, ZMM-high=1184 and kmask=2208. Preserve the entire old prefix through
   `x87_pointer_model`; new storage starts after it. Overall CpuState size may grow.
3. **Cache/memory witnesses (.4/.7/.11/.12).** Reuse the same bytes/address while
   changing each key/guard dimension; alias equal PA with different CS-relative IP;
   test a same-selector descriptor reload, unchanged cache after GDT edit, CR3 writes
   of equal/different values, remapping, permissions, physical-alias SMC, MMIO and
   page/instruction crossings. Exercise reset high fetch without a 4-GiB allocation.
4. **Execution/event witnesses (.2–.10).** Reproduce each supplied probe as an
   isolated guest-agnostic fixture, including illegal real uses, not just success
   paths. Test .4's outward faults before .6; then both gate widths and failure
   escalation. A manual Application Compat32 IRQ must leave stack/memory untouched.
   Slice across PE/CS/SS changes, REP widths, pending device completions and IRQ
   shadows; assert exact counts, restart prefixes and no duplicated effects.
   Test ring transitions and denied I/O before any device handoff. Paging's marker
   must distinguish translated from untranslated memory and recover through #PF.
5. **Checkpoint witnesses (.13).** Resume before/after PE/PG, CS/SS reloads, reset
   fetch and shadowed events; compare continuation, memory, counters and delivery.
   Permit snapshots only at documented completion boundaries: initially refuse
   outstanding or answered-but-unfinished port/MMIO work rather than serialize an
   embedder transaction. Round-trip hidden caches even when tables have changed;
   discard translation/access/predictor caches on restore.
6. **Existing-mode codegen and cost evidence (every child).** Capture baseline and
   candidate with identical toolchain, backend version/options, host ISA/features,
   guest snippets and memory configuration. Preserve emitted block/region bytes,
   relocation records and disassembly for Application Long64/Compat32, including
   stack, branches/chains, memory/TLS, REP and MXCSR specializations. Require exact
   bytes for deterministic fixtures with fixed relocation targets; otherwise compare
   every byte outside recorded relocation payloads and symbolically verify those
   payloads/targets. This is relocation normalization, not permission for changed
   instructions. Real16 has no generated guest code: prove no materialization and
   compare its lifted IR and optimized unstepped interpreter disassembly. Interleave
   before/after dispatch, REP, mixed-workload and Real16 benchmarks using the same
   host/configuration and the repository perf gate; retain raw results and investigate
   regressions. Land decision-17's compiler upgrade separately, then rebaseline;
   never hide a context change behind compiler-version differences.
7. **Full gate (every child).** Run the current `scripts/ci-gate.sh` lanes for
   format, guest-agnostic guard, clippy, deny, tests with required 80286 corpus and
   Unicorn, release, lane32, doctests and cross-target checks; preserve native-oracle
   coverage on x86-64 and ARM64 where supported. Run the full real-program ladder
   after lifter/codegen changes and the performance gate. Cite command logs and
   unavailable coverage. These are implementation acceptance requirements, not
   results claimed by this documentation-only proposal.

## Maintainer decisions (2026-11-10)

Accepted as written: the Application/LegacySystem opt-in, the model-profile boundary
shared with xj-50w, the per-vCPU cache namespace and epoch policy, the appended inline
state, and the staged outward faults and typed unsupported exits. The 80386
Programmer's Reference Manual is to be pinned in `oracles/` as a second witness before
any 386-specific fact listed below is implemented. Public API names are confirmed in
xj-srh.1's review.

### Choices as originally posed

- Accept the explicit Application/LegacySystem opt-in and the proposed model-profile
  boundary with xj-50w; confirm public naming before .1 changes API.
- Approve the conservative per-vCPU cache namespace/epoch policy and appended inline
  state cost; cross-vCPU IR reuse and legacy compilation require later evidence.
- Pin the 80386 Programmer's Reference Manual as a second witness before resolving
  model-specific gaps in the modern SDM/APM, particularly CR0/FLAGS masks, paging
  permissions/reserved bits and reset/A20 details. No unpinned semantics are accepted.
- Confirm staged outward faults/typed unsupported exits and keeping public nonzero-CPL
  execution constrained until .10's checks land; accept .12 as a prerequisite of
  .13's claim to complete reset checkpoint continuation.
