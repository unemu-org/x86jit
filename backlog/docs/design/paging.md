# 386/486 two-level paging on Machine models

Design proposal for xj-srh.7; engine audit at `274d48a8`. This note changes no
implementation. It extends the Machine address boundary in
[`spec.md` §17.10](spec.md#1710-cpu-model-catalogue-and-machine-profile), the central address-lowering rule
in §16, and the physical bus gate delivered by xj-srh.11. **Hardware facts** below
come from primary manuals. **Proposed policies** describe the engine contract;
they do not claim cycle, prefetch, cache-capacity or board-bus fidelity.

The target is interpreted Machine execution on `I80386`, `I80486Dx` and
`I80486DxCpuid`: 32-bit linear and physical addresses, 4 KiB pages, CR0.PG,
CR3, PDE/PTE P/U/S/R/W/A/D, CR2, restartable #PF, and the 486's WP and INVLPG.
Instruction fetch, explicit operands and implicit system accesses all participate.
Application profiles retain their existing memory contract and execution paths.
PSE/PAE/PGE, debug registers and extensions to hardware task switching beyond
the existing switch engine's paged CR3 handoff are outside this proposal.
V86 was outside the original paging proposal and is now supported through the
same walker and continuations ([v86.md](v86.md) §§2.1, 4.4, 11). V86 fetch,
data and stack accesses are user accesses at every IOPL; implicit IDT/GDT/TSS
accesses remain supervisor accesses. Authority: SDM Vol 3B §§23.2.3–23.2.4
pp. 23-7–8; 386 PRM §15.2 pp. 15-3–4; Intel486 PRM §23.2 pp. 23-4–5.
Run-boundary CpuState/RAM restores call `invalidate_paging` even with unchanged
CR0/CR3; retained proofs and same-CR3 remapping restores have V86 witnesses.

## 1. Sources and hardware contract

References give **printed pages** and, where useful, one-based PDF positions.
The combined SDM is `325462-092US`, June 2026: its paging chapter is **Vol. 3A
Chapter 5**, rather than Chapter 4 in older editions. The other sources are the
80386 PRM `230985-001` (1986) and Intel486 Family PRM `240486` (1992), named
`oracles/intel/i386-prm.pdf` and `oracles/intel/i486-prm.pdf`. The SDM path is
`oracles/intel/sdm.pdf`. The scans' protection tables, #PF error-code diagrams,
486 WP description, protection-fault A-bit rule and INVLPG entry were checked
against page images as well as extracted text. Existing pins are documented in
[`PROVENANCE.md`](../../../PROVENANCE.md); no manual is a build dependency.

### 1.1 Address formation and permissions

| Ref | Source, section and page | Hardware fact |
|---|---|---|
| H1 | SDM Vol. 3A §§5.1.1–5.1.2, pp. 5-2–5-4; §5.3, pp. 5-9–5-13. 386 PRM §5.2, pp. 5-7–5-11 (PDF 119–123), Figs. 5-8–5-11 | With PG clear the paging translation stage is bypassed. With PG set, CR3[31:12] identifies the physical page directory; linear bits 31:22 select a 32-bit PDE, bits 21:12 select a 32-bit PTE, and bits 11:0 select the byte in its 4 KiB frame. Both tables have 1024 entries. Paging cannot be enabled with PE clear. |
| H2 | SDM Vol. 3A §5.3, pp. 5-9–5-13; 386 PRM §§5.2.4.1–5.2.4.2, p. 5-9 (PDF 121); 486 PRM §§5.3.1–5.3.4, pp. 5-18–5-21 (PDF 154–157) | The table addresses are physical, not recursively paged linear addresses. P=0 at either level prevents translation and produces #PF; the remaining bits of a not-present entry are available to software. There is no present bit in CR3. |
| H3 | SDM Vol. 3A §§5.6.1, 5.10.2.2, pp. 5-33–5-35, 5-43; 386 PRM §§6.4.1–6.4.2, pp. 6-20–6-22, Table 6-5 (PDF 148–150); 486 PRM §§6.8.1–6.8.2, pp. 6-23–6-25, Table 6-4 (PDF 187–189) | Effective U/S is the AND of PDE.U/S and PTE.U/S. Effective R/W is their R/W AND. CPL 0–2 accesses are supervisor, CPL 3 accesses are user. A user access requires effective U/S=1; a user write also requires effective R/W=1. On the 386, supervisor reads and writes ignore R/W, including read-only user pages. |
| H4 | SDM Vol. 3A §5.1.3, p. 5-5; §5.6.1, pp. 5-33–5-35. 486 PRM §4.1.3, WP, p. 4-7 (PDF 129); §6.8.1.2, p. 6-24 (PDF 188) | 486 CR0.WP is bit 16. WP=0 preserves supervisor writes to read-only pages. WP=1 makes supervisor writes sensitive to R/W. The 486 PRM describes the user-page copy-on-write case; the SDM explicitly applies WP to both supervisor and user mappings. Use that full SDM rule, with the historical wording qualification recorded in §6 Q3. There is no WP control on the 386. |
| H5 | SDM Vol. 3A §5.6.1, p. 5-33; 386 PRM §§6.4.3–6.5, pp. 6-21–6-22; 486 PRM §§6.8.3–6.9, pp. 6-24–6-25 | Segment protection precedes page protection. Implicit GDT/LDT/IDT/TSS accesses use supervisor paging privilege regardless of CPL. Inner-stack accesses on a privilege transition use supervisor privilege. Ordinary explicit references to bytes that happen to contain a table still use the instruction's normal privilege. |

For these models the translation arithmetic is:

```text
pde_pa = (CR3 & 0xfffff000) + 4 * (linear >> 22)
pte_pa = (PDE & 0xfffff000) + 4 * ((linear >> 12) & 0x3ff)
pa     = (PTE & 0xfffff000) | (linear & 0xfff)
user   = access_privilege == User
wp     = model_is_486 && CR0.WP
allow  = (!user || (PDE.U/S && PTE.U/S))
      && (!write || (!user && !wp) || (PDE.R/W && PTE.R/W))
```

P must be checked at each level before using its frame or permissions. Present
checks complete before a protection #PF is selected. A read/modify/write operand
requests write permission even when its eventual stored value is unchanged.
Fetch uses read permission for paging; these models have no execute-disable bit.
An embedder fetch policy is a separate physical-memory policy.

Do not import modern reserved-bit faults or page-size semantics. SDM §5.3,
p. 5-10, says there are no reserved-bit checks in 32-bit paging with PSE disabled;
the older PRMs mark fields reserved for future use. Proposed policy: interpret
only these models' documented fields, preserve other stored bits on A/D updates,
and do not synthesize modern RSVD faults from them. CR3 masks remain
`0xfffff000` on the 386 and `0xfffff018` on the 486 (PCD/PWT, 486 PRM §4.1.3,
p. 4-6). PSE, PAE, global pages, PAT and NX are absent.

### 1.2 #PF, restart and page-table side effects

| Ref | Source, section and page | Hardware fact |
|---|---|---|
| H6 | SDM Vol. 3A §5.7, pp. 5-37–5-38; §7.15, Event 14, pp. 7-47–7-50 (PDF 3396–3399). 386 PRM §9.8.14, pp. 9-17–9-19, Fig. 9-8 (PDF 193–195); 486 PRM §9.9.14, pp. 9-21–9-23, Fig. 9-7 (PDF 241–243) | #PF is fault vector 14. CR2 receives the faulting **linear** address. Error bits 0, 1, 2 describe protection versus not-present, write versus read, and user versus supervisor access. The original instruction is restartable; the saved IP normally identifies that instruction. A later #PF can overwrite CR2, including during delivery. |
| H7 | 386 PRM §9.2.1, pp. 9-2–9-3; §9.8.14.1, p. 9-18; §14.6, pp. 14-4–14-5. SDM Vol. 3A §7.15, Event 14, p. 7-50 | Fault restart restores instruction state; REP may retain completed iterations. Task-switch faults have a distinct outgoing/incoming-task boundary and can name the new task's first instruction. The 386's real-mode exception rules do not make paging legal with PE clear. |
| H8 | SDM Vol. 3A §5.8, p. 5-39 (PDF 3298); 386 PRM §5.2.4.3, p. 5-10 (PDF 122); 486 PRM §5.3.4.3, pp. 5-21–5-22 (PDF 157–158) | Hardware sets A (bit 5) in both levels used by a translation, and PTE.D (bit 6) before a write to the mapped page. Hardware does not clear them. The historical PRMs leave PDE.D undefined. Clearing A/D in memory while a translation is cached need not make a later access set them again without invalidation. |
| H9 | 486 PRM §9.9.14, p. 9-21 (PDF 241); SDM Vol. 3A §§5.10.2.3, 5.10.3.1, pp. 5-43–5-46 | On a 486 protection #PF, **both PDE.A and PTE.A are set**. The SDM permits upper-level accessed entries/cache entries even when no complete translation exists below them. A/D effects are therefore not universally rolled back with a faulting operand. The cited 386 PRM does not specify every failed-walk A-bit intermediate state. |
| H10 | SDM Vol. 3A §11.1.2.1, p. 11-3 (PDF 3464); 386 PRM §11.2.2, p. 11-5 (PDF 223); 486 PRM §13.1.2, p. 13-3 (PDF 295) | A/D updates use locked memory cycles. Each update must preserve competing edits to the entry. This does not make the two-level walk, all its updates and the operand one indivisible transaction. |

For this feature the architected error-code calculation is
`protection | (write << 1) | (user << 2)`. Instruction fetch has W/R=0; retain
the separate execute-access metadata but do not invent an I/D bit. Both historical
error-code diagrams label bits 31:3 **undefined**. Proposed deterministic policy:
deliver zero in those bits, and mark only `0x7` as known architectural bits in
`ErrorCode`. Modern RSVD/I/D/PK meanings do not apply. The older prose describes
U/S in terms of execution mode; H5 and the SDM's access-based definition require
implicit supervisor accesses to report U/S=0 even when they originated at CPL 3.

The manuals require the before-operand ordering and individual locked updates;
they do not prescribe one universal, bus-visible ordering of every PDE/PTE update
on every failed or speculative access. The deterministic walk order proposed in
§3.3 is an engine policy except where H8–H10 explicitly constrain it. In
particular, do not claim that all x87 memory stores are architecturally all-or-none:
SDM Vol. 3A §11.1.1, pp. 11-2–11-3, permits partial writes by some multi-access
x87 instructions. The stronger restart policy below follows this repository's
existing fault-atomicity contract.

### 1.3 Walk reads, caching, MMIO and missing physical memory

386 PRM §5.2.5, p. 5-11 (PDF 123), and 486 PRM §5.3.5, p. 5-22 (PDF 158),
describe caching translations and avoiding table bus reads on cache hits. Reloading
CR3 invalidates translations; the historical descriptions also specify task
switches that change CR3. SDM Vol. 3A §§5.10.2–5.10.4, pp. 5-42–5-52, permits
TLBs and upper-level paging-structure caches, but software cannot require a given
entry to remain cached. A functional implementation may omit both caches.

Translation caching and caching the *memory reads of the tables* are different.
Walks read physical memory (H1–H2); 486 PRM §5.3.4.5, p. 5-22, and
§12.3.1, pp. 12-4–12-5, specify PCD/PWT controls. CR3 supplies those controls
for the directory and the PDE supplies them for the next table; compare SDM
Vol. 3A §5.9.1, p. 5-40. The 486 has an internal write-through data cache
(PRM §12.1, p. 12-2). This proposal preserves cache-control register bits but
models neither that data cache nor external cache timing.

No cited rule turns a physical address into #PF merely because a board decodes
it as a device or fails to supply RAM. A PDE/PTE in MMIO is a physical read from
that device, subject to the board's response and the CPU's caching; A/D may
require a locked update there. There is no architectural guarantee that arbitrary
device registers behave like page tables. Likewise, an unpopulated physical
directory address has no CR3.P bit that could produce a not-present fault: the
board must supply a response, or the bus does not complete. If a supplied value
has P=0, *that value* causes #PF. These conclusions follow from H2 and the bus
boundary, rather than from an imagined architectural "unmapped RAM" exception.

**Proposed initial engine policy:** require page-directory/table storage in
ordinary RAM; return a typed facility refusal for device/ROM table storage before
performing table transfers. Do not emulate a locked table update with separate
MMIO read/write exits. A physically absent table transfer returns the existing
physical-memory exit, with walk-level and original-linear-address evidence; it
does not change CR2 or deliver #PF. Operand MMIO after a successful walk remains
supported through the normal physical bus handoff. This explicit limitation and
its possible later atomic-device extension are Q1, not a hardware restriction.

INVLPG is a 486 addition: 486 PRM §26, **INVLPG**, p. 26-172 (PDF 670), and
§5.3.5, p. 5-22. Its memory addressing form identifies the translation to
invalidate; it does not read the named operand. It requires CPL 0 in protected
mode, rejects a register form with #UD, and preserves flags. It is unsupported
on the 386. SDM Vol. 3A §5.10.4.1, pp. 5-48–5-49, defines its operand as a
linear address and allows conservative invalidation. A full functional-TLB flush
is an acceptable initial engine choice.

## 2. Current engine audit

All `file:line` locations below are relative to `x86jit-core/src/` at the audited
revision unless explicitly prefixed otherwise. They describe present behavior,
including facilities that are still refused.

| Area | Evidence and current behavior |
|---|---|
| Model/state | `model.rs:16` contains all three target entries; both 486 entries project to the I386 generation. Select WP/INVLPG by model semantics, not just generation. `legacy.rs:209` already contains `cr0`, `cr2`, `cr3`, tables, hidden segment caches and CPL. `state.rs:455` embeds it in the `repr(C)` CpuState. No new CR2 field is needed. |
| Logical → linear | `legacy.rs:1337` (`access`) checks cached segment type, usability, inclusive limits and expand-down bounds, then returns wrapping 32-bit base+offset. `interp/legacy.rs:352` (`segment_access`) delegates to it for checked contexts. Address IR lowering and stack helpers feed that boundary. |
| Linear → bus today | `interp/legacy.rs:516` (`read`), `:799` (`write`), `:890` (`probe`) apply `legacy::physical_span` directly to the linear address. `legacy.rs:1478` chooses the model bus mask; `:1497`/`:1501` split at address-width/A20 discontinuities and gate physical bit 20. There is no page walk between those stages. |
| Physical calls | `interp/legacy.rs:531`/`:847` perform `mem.raw().read/write`; `:905` probes physical writes. `LegacyMemory`/`LegacyAccess` at `:449`/`:399` carry ordering, retry visits and write observations. `memory.rs:5641`, `:6017` define `Access` over `InterpreterMemory`; `:3281`, `:3418`, `:3511`, `:3636` provide Memory read/write/RMW/CAS. There is no separate public `Bus` trait: these physical Memory operations and typed MMIO exits are the bus boundary. |
| System calls into memory | Descriptor reads are `interp/legacy.rs:1050`/`:1061`; descriptor busy/accessed CAS is `:1116` and currently gates a linear address directly to a physical byte. IDT/frame delivery is `:1649` (`deliver_protected`); privilege-stack TSS reads are `interp/legacy/privilege.rs:89`; task TSS reads/writes are `interp/legacy/tasks.rs:572`/`:758`. All must use supervisor translation. Crucially, `interp/legacy/privilege.rs:823` currently returns `LegacyIoPermissionBitmap`: there is no working bitmap read to retrofit. |
| x87 | `interp/legacy.rs:1847` (`PhysicalFpMemory`) implements `FpMem` by A20-splitting direct Memory reads/writes. `:1887` probes every destination before stores. It bypasses scalar `read`/`write`, so translating only scalar helpers would miss x87 and its environment/save images. |
| Split locked operands | `interp/integer.rs:1014` documents `locked_split_update`: staged RAM pieces and device completions provide the existing single-processor result, with no multi-vCPU atomicity guarantee for split RAM pieces. Machine models run one vCPU (`spec.md` §8.2.3); paging retains that boundary. |
| Fetch | `vm/legacy.rs:1290` (`fetch_watched`) selects the generation limit/mask; `:1325` (`fetch_bytes`) checks CS and reads one required byte at a time through `Memory::fetch_code` (`memory.rs:3327`). It decodes incrementally and stops as soon as the instruction is known. The fetch policy is called with a physical page number. |
| Checked decode cache | `vm/legacy.rs:18` (`LegacyBlockKey`) keys private instruction IR by owner/context/model identity, decode/stack widths, regime, decode IP, **linear_fetch**, and MXCSR mode. `:651` builds that linear address. `:682` re-fetches required bytes and checks equality even on a hit. `LegacyContextStamp` (`:32`, `:95`) includes CR0/CR3 and segment/table state but deliberately normalizes CR2 away (`:52`). |
| Direct/shared cache | `vm/legacy.rs:1034` (`resolve_direct`) uses the post-A20 **physical** fetch address with `BlockKey::machine_direct` (`cache.rs:40`). Publication at `vm/legacy.rs:1252` registers physical source spans. Direct execution and the shared Application cache must never be reused as the paged path merely because the starting physical byte matches. |
| SMC | `memory.rs:1636` tags code by backing-memory page; Machine calls supply physical addresses. `:1967` observes physical writes and invalidates covered source bytes/pages. `cache.rs:913` and `:1595` protect source publication and invalidate overlaps. `Memory::declare_code_alias` (`memory.rs:1659`) handles distinct physical ranges sharing host storage; two linear aliases of one physical frame already converge before this layer. |
| Backing | `memory.rs:1100` (`Memory::sparse`) reserves a host span and commits mapped physical storage on demand; `memory/sparse.rs:48` owns the reservation. `HostRam` (`memory.rs:154`) supplies embedder host backing and protection hooks; `:1474` validates it. `MemoryModel::SoftMmu` is a physical backing choice here, not guest CR3 paging. A guest PTE does not allocate host RAM or change Memory region permissions. |
| PG refusal/control writes | `interp/legacy.rs:187` rejects PG with PE clear as #GP and otherwise refuses PG=1 before changing CR0. Positive Machine entry independently refuses a paged context at `vm/legacy.rs:1615`. `interp/legacy.rs:3715` already masks 386/486 control writes differently. `:3727` stores CR3 and reports both context change and translation invalidation even for an equal-value write. `vm/legacy.rs:576` clears retained translation state. |
| INVLPG today | `lift/legacy.rs:1102` lowers INVD/WBINVD/INVLPG to the same operand-less `I486Invalidate` (`legacy.rs:761`). `interp/legacy.rs:3433` completes that as a no-op with PG off and refuses it with PG on. A real INVLPG implementation must retain its effective linear address and distinguish it from the other two instructions. |
| Task CR3 today | `interp/legacy/tasks.rs:474` refuses all paged task switches before saves/busy updates. `:572` reads the incoming image, but does **not** read TSS offset `0x1c` or install CR3; `:609` explains that CR3 is read only with PG=1. `:633` explicitly excludes CR3 from outgoing saves. `:776`/`:792` already distinguish committed incoming-task state from the old task. |
| Exception metadata/delivery | `interp/legacy.rs:168` creates architectural fault metadata, including error codes for vector 14, but there is no translation-generated #PF producer. `fault.rs:260` defines `PageFaultProjection` as partial map/access evidence, explicitly separate from #PF. `fault.rs:612` preserves linear and physical bus addresses. `vm/legacy.rs:2257` routes architectural faults to InGuest delivery or returns them outward; `:1846` carries `LegacyEvent.error_code` through delivery. |
| Checkpoints/retry | `lift/legacy.rs:335` inserts checkpoints for checked instructions requiring rollback. `interp/legacy.rs:2174` snapshots GPRs, flags, **all legacy state** and selectors; `:2240` restores them on an uncommitted failure. This would overwrite a freshly set CR2. MMIO completions live in the CpuState ledger (`interp/legacy.rs:531`, `:847`); task/event continuations also have Vcpu-private state. |
| Census | `x86jit-tests/tests/legacy/legacy_census_386.rs:571` and `:577` are ignored runtime censuses for the 386 and both 486 entries. `:583` runs real/P16/P32 probes, classifies exits and emits TSV reports; it does not constitute a paging witness. Extend the setup matrix deliberately rather than relabeling all PG refusals as success. |

## 3. Proposed implementation contract

### 3.1 One translation boundary, selected outside the default path

Keep logical address lowering centralized. The ordered boundary is:

```text
effective offset
  → segment/access checks and wrapping 32-bit linear address
  → translate_linear(linear, access, access_privilege, paging_context)
  → model physical width and A20 gate
  → physical Memory/Access operation or MMIO handoff
```

`translate_linear` is one proposed Machine-only implementation, shared by fetch,
data, string, x87 and system adapters. Its input distinguishes Fetch/Read/Write
(RMW requests Write) and User/Supervisor; implicit system privilege must be passed
explicitly, not derived blindly from live CPL. Its result is a translated span,
an architectural page fault, a physical-bus failure, or a typed facility refusal.
A page fault carries the original linear byte, access kind and low-three-bit
error code. A successful span ends at the next linear page boundary. The caller
then splits it again at physical region/A20 boundaries; physically adjacent
frames do not justify skipping the second page's permission check.

Use two concrete Machine memory adapters, selected at positive run entry and
after a control-state transition: existing PG=0 access and paged access. Use
static dispatch/monomorphization in the instruction walk where needed. Do not
add a PG test or a TLB pointer lookup to every existing `Memory::read/write`,
Application operation or unpaged Machine transfer. CR0 changes end the current
instruction/block and select the next adapter before the next fetch. PG=1 always
takes the checked interpreted route; it does not enter Direct or a JIT tier.

Page-walk reads and A/D CAS use a *physical-only* operation beneath translation.
They use the same bus gate and ordering as operand traffic but cannot recurse
through `translate_linear`; guest R/W permissions on a linear mapping of the
tables do not prevent these physical A/D updates. Distinguish the original
operand linear address, pre-gate
physical address, final bus address, walk level and transfer identity. Update
fault metadata naming if necessary: `RealModePhysical` must not be used to label
a protected-mode physical bus address as though it were a linear address.

For paged x87, prepare translated load/store spans before invoking the numeric
engine; retain a typed translation failure beside the `FpMem` adapter, because
its current boolean/address-only return cannot express #PF by itself. On failure
return the retained architectural fault and preserve x87 destinations, TOP/tag
state and pending exception state. The scalar helper is not an alternative
translation implementation. Shared string fast paths must similarly use the
paged adapter or fall back to the checked per-element loop.

### 3.2 Functional TLB and invalidation

A small bounded per-Vcpu TLB is sufficient, outside CpuState in the Machine
execution object. Start with a simple direct-mapped array, for example 64 entries;
capacity/replacement is an engine choice. Each successful entry contains linear
page, pre-A20 frame, effective U/S and R/W, PTE physical location, and whether
the translation has completed D-setting for writes. Its identity includes the
current translation generation; CR3/model/PG/WP are owned by that generation.
Do not cache negative results or host pointers. No separate PDE cache is needed.
Permissions are checked on every hit with the current access privilege.

A clean read-filled entry is not a fully completed write translation. On its
first write, evict it and walk again to set D safely before caching a writable
result. This may observe a changed mapping, which is allowed by the manuals'
cache-eviction rules. It avoids blindly setting D in a stale PTE and continuing
with a different frame. A/D clearing requires guest invalidation for predictable
re-setting; ordinary table stores do not flush the TLB automatically.

| Trigger | Required engine action |
|---|---|
| MOV CR3, including equal value | Clear functional TLB, advance translation generation, discard paged fetch proofs and prevent old-generation block execution. |
| CR0.PG change | Same action; reselect paged/unpaged execution before the next instruction. Keep PG=1/PE=0 #GP before commit. |
| 486 CR0.WP change | Same conservative flush. The 386 mask continues to ignore this nonexistent control. |
| 486 INVLPG | Evaluate the addressing form centrally to a linear address without reading/paging the named operand; invalidate its page or conservatively flush the whole functional TLB. End the current block and advance paged block generation. Keep CPL/form checks. INVD/WBINVD remain distinct cache operations. |
| Incoming 386 TSS CR3 load with PG=1 | Install through the same CR3-load routine and flush, including equal CR3 as a conservative policy. Do not read/load it for PG=0 or a 286 TSS. |
| Effective A20 change | Flush paged translations/proofs and advance generation at the existing architectural boundary. Although frames are pre-gate, the tables themselves may now be read through different physical addresses. |
| Embedder control/segment/model edits, memory-map replacement, restore/reset | Reconcile at positive entry; flush paged state when translation assumptions or the physical map generation change. Supply an explicit paging-invalidation operation for page-table edits/restore with unchanged CR0/CR3. |

CR2 edits do not invalidate translations. The existing normalized context stamp
can detect changed CR0/CR3, but cannot detect an equal-value CR3 *reload* or an
embedder restoring identical control bytes with different page-table RAM. An
explicit invalidation API is necessary; guest reloads already have an outcome
flag. A zero-budget call must neither walk nor mutate CR2/A/D/TLB state. Generation
wrap discards the paged cache before an old identifier can be reused.

### 3.3 Walk order and A/D updates

Use this deterministic order on a TLB miss:

1. Read the PDE physically. On a physical-bus failure return that failure; on
   PDE.P=0 produce not-present #PF without touching its other bits or reading a PTE.
2. Atomically OR PDE.A, preserving other bits, then read the PTE physically.
   On PTE.P=0 produce not-present #PF. PDE.A may remain set.
3. Atomically OR PTE.A. Check combined permissions only after both present entries
   have been obtained. A protection fault therefore leaves both A bits set, as
   explicitly required by the 486 manual. Use the same deterministic choice for
   the 386; its failed-walk intermediate states are not fully specified here.
4. Finish translation/preflight of the access's remaining fragments and other
   potentially faulting operand requirements. For an allowed write, atomically OR
   PTE.D for every writable fragment before the first operand write. A permission
   failure never sets D merely because the access requested a write.
5. Transfer the operand and commit instruction outputs. Do not clear page-table
   bits if the operand later encounters a physical-bus handoff or fault.

The order is not a claim that real processors never update/speculate differently.
Do not set a bit in a not-present entry. Leave PDE.D untouched as the deterministic
choice for its historical undefined state. A later-page fault can leave A on
earlier translations while leaving operand RAM unchanged. A successful D update
can survive a later physical failure; document it as a page-walk side effect.

Implement each RAM bit update with the existing physical atomic primitive/CAS
loop, at the gated address, merging only the required bits. If CAS observes a
change to frame/P/permissions, restart that translation from the changed data
rather than publishing the old frame under new rights. On a failed CAS preserve
all competing software bits. Do not hold a Rust reference that assumes table RAM
is immutable, and do not treat an unlocked read followed by write as sufficient.
Walk/CAS ordering is separate from a guest LOCKed operand's atomicity.

All successful A/D writes go through physical write observations and SMC handling.
This conservatively exposes changes to executable page-table storage; SDM §5.8,
p. 5-39, explicitly permits variation in hardware SMC visibility for these writes.
Avoid publishing a fetched source that predates an A/D write to the same bytes.

### 3.4 Fault production and commit boundaries

Create a dedicated page-fault producer instead of using map-fault projections or
the generic selector-error helper unchanged. It creates `Exit::Exception` vector
14, fault class, instruction/saved IP, linear-address evidence and the `0x7`
known error-code mask. The failing byte in the second page, rather than the
operand base or the PDE/PTE physical address, becomes CR2. RMW failures report
write access; system failures report supervisor access. A segment fault wins
before translation and leaves CR2 alone.

Carry the new CR2 value in the typed fault until ordinary instruction rollback
has finished, then publish it **before either outward return or guest exception
delivery**. Apply the same rule to failed exception delivery. The current broad
legacy-state checkpoint must not undo CR2. Do not roll back A/D memory updates
with architectural destinations. Audit generic fault-progress reporting so that
"operand RAM unchanged" does not incorrectly claim "no RAM effects" when A/D
writes occurred. A physical map/host failure does not become #PF just because
PG=1, and must not update CR2.

| Access family | Commit/retry rule |
|---|---|
| Scalar or page-crossing operand | Translate and permission-check every fragment, probe physical destinations, stage load values/results, then commit. A second-page #PF leaves all operand bytes, GPRs and flags unchanged. Respect 32-bit linear wrap before splitting, followed by physical A20 splitting. |
| Multi-access instructions/stack frames | Collect and validate all potentially faulting operand reads and destinations before writing RAM/register outputs. Do not rely on a register checkpoint to undo an earlier RAM store. Keep the existing task-switch commit boundary as the explicit exception. |
| REP/string and port strings | Atomicity is per iteration, not across the REP. Keep completed iterations and count/index progress; preflight all accesses for the next iteration before its irreversible device/port operation. A zero count performs no walk. On #PF discard that iteration's continuation/answers; after handler repair, execute it anew. Preserve existing compare-string fault flag rules. |
| LOCKed RMW / implicit XCHG | Resolve and check write permission for all pages before reading/updating the operand. One physical span uses existing Memory atomic operations. Feed noncontiguous frames to a scatter form of the existing split-operand plan: retain one operand snapshot, stage every piece, suppress guest interruption through device handoffs and commit once. This preserves the current single-vCPU Machine contract; independent per-piece RMWs do not. Keep the supported one-device split protocol and typed refusal for shapes the bus API cannot complete. Multi-vCPU scatter atomicity is not introduced here. |
| x87 memory/environment/save images | Preflight the whole requested image and stage loads/results before committing the numeric state or stores; pass all fragments through the shared translator. Keep existing #NM/#MF and segment-check priority. This is the repository's strong restart policy, with the H7–H10/§11.1.1 qualification above. |
| GDT/LDT/IDT/TSS/bitmap | Translate their linear addresses with implicit supervisor privilege, including descriptor accessed/busy **writes**, privilege-stack reads/writes and bitmap reads. A descriptor-byte CAS checks paging write permission first, including WP. LGDT/LIDT's explicit memory operand uses ordinary privilege before its contents become a linear table base. |
| Exception delivery | A #PF reading an IDT/descriptor or writing a handler stack updates CR2 to that new fault's linear byte; preserve error-code delivery through 16/32-bit gates and existing #DF/shutdown classification. Do not report a physical missing region as an architectural second #PF. |

MMIO cannot be rolled back. Before issuing any operand MMIO or port I/O,
preflight paging for the entire current atomicity unit so a later #PF cannot
duplicate an earlier device side effect. For physical-bus resumptions keep the
decoded instruction and a stable translated-span plan: distinguish operand slot,
transfer visit, bus address, direction and translation generation in completion
identities. The existing physical-address-only write key is insufficient for
two distinct stores that alias one bus address. Do not reuse a completed answer
for another walk or another REP iteration. Changed inputs invalidate a retry
plan only at a documented abandonment boundary; do not silently remap an
already-partially-completed device instruction on retry.

Page tables that alias operand storage require special care: resolve and retain
the complete transfer plan before committing operand writes, so the first piece
cannot rewrite the PTE used to locate the next piece. Concurrent embedder/DMA
page-table edits require the documented synchronization contract; this proposal
does not promise an atomic snapshot of a changing page hierarchy. Do not expand
the single-vCPU Machine contract as part of paging.

### 3.5 Fetch, blocks and physical aliases

Retain incremental instruction fetch. Check/translate each required byte (or an
equivalent bounded span), including a second page only when decode needs it.
A one-byte instruction at page offset `0xfff` must not fault because a speculative
15-byte buffer reaches an absent next page. A real cross-page instruction with
an absent second page reports #PF for that first required missing byte and saves
the instruction's starting IP. Fetch itself can set A; it never sets D.

Start with one-instruction paged blocks using the Checked cache. Each lookup is
guarded by model/decode/segment identity, linear start/decode IP, translation
generation, CR3 identity, PG/WP assumptions and the ordered physical source spans
actually fetched. Revalidate page permission/fetch policy and bytes even on a
hit. A block lifted at linear L under directory A cannot execute under directory
B; CR3 alone is also insufficient after equal-value reload, INVLPG or remapping.
If a TLB eviction causes a new mapping inside one generation, fresh fetch and
physical-span identity must reject the old proof even when the bytes are equal.
Avoid unbounded generation-keyed cache growth: flush/drop paged entries when
advancing the generation, with bounded retention inside a generation.

Register source coverage using **post-A20 physical bus spans**, including both
frames of a cross-page instruction. Store barriers, descriptor updates, A/D CAS,
MMIO write acknowledgements and embedder/DMA `note_write` calls all invalidate
those same physical spans. Thus linear pages L1 and L2 mapping frame P require
no guest linear-alias declaration: a store through either invalidates code
fetched through both. Distinct physical ranges sharing host storage still use
the existing `declare_code_alias` contract. Guest table edits and operand-code
SMC are separate mechanisms: an ordinary PTE edit need not flush the TLB, while
a write to currently mapped code must trigger physical SMC handling.

Keep `LiftSource`/publication-generation protection for fetch/write races. A/D
writes can invalidate a source while fetching it; acquire/register the final
source after those updates or retry validation. Never register only the first
frame or infer one contiguous physical range from a linear instruction length.
Initially there is no paged multi-instruction block, chaining or compiled-block
reuse to prove safe; any later optimization must preserve these guards.

### 3.6 A20, backing and MMIO

A20 is applied **after translation**, to both PDE/PTE traffic (including CAS) and
final operand/fetch traffic. Never clear linear bit 20 before selecting the PDE
or PTE. A gated walk can read a different directory/table; a gated final frame
can alias code/data even when the linear addresses differ. Store pre-gate frames
in the TLB and gate each final physical transfer through the existing
`physical_span_with_gate`. Freeze the line over an instruction/REP iteration and
device retry using the existing xj-srh.11 boundary contract.

Physical `Memory` stays authoritative for RAM/ROM/MMIO region kind, sparse
commit state, HostRam recovery, fetch policy and bus exits. Guest page R/W is
not `Prot`; a page-allowed write to physical ROM still follows ROM policy. Guest
P=1 cannot make a physical hole readable. Guest page U/S does not turn an
embedder memory trap into an architectural protection #PF. The paged adapter
must not bypass `Access` to index `host_base + linear` or make a separate 4 GiB
linear host mapping per CR3.

Operand MMIO exits carry physical bus address/actual fragment width/value for
the embedder, plus original linear evidence. Paging failure precedes operand
MMIO classification. Table MMIO uses the explicit initial refusal in §1.3;
supporting it later requires atomic table-update and retry semantics rather
than treating a device completion as an ordinary operand answer.

### 3.7 TSS handoff, state layout and checkpoints

For an already-supported switch to a 386 TSS with PG=1, read its CR3 image at
offset `0x1c` through the outgoing address space along with the incoming image.
The 386 PRM §7.1, Fig. 7-1, pp. 7-2–7-3, and §7.5, p. 7-8, define that
static field and incoming PDBR load; SDM Vol. 3A §10.3, pp. 10-10–10-11,
gives the paging handoff sequence. Save the outgoing dynamic state using the
old CR3; CR3 is not written back as a dynamic TSS field. At the existing incoming
commit phase install the masked new CR3 and flush before translating the new
LDT/segment descriptors, privilege-stack references or instruction fetch.
Postcommit faults retain incoming-task ownership/IP; precommit faults retain
the old task. Do not move CR3 loading after descriptor validation or pretend
the whole switch has ordinary instruction rollback. Retain existing switch
forms and limits; adding another hardware-task feature is outside scope.

The new feature also needs an actual I/O permission-bitmap reader: on the
currently refused wide-TSS path, read its base/permission bytes as implicit
supervisor accesses through this same translator, check all relevant permission
bits before port I/O, and preserve the existing 286/IOPL rules. Its bit/limit
behavior must be witnessed from 386 PRM §§8.3, 17.2.2.5, pp. 8-4–8-6,
17-15; no host I/O-device policy belongs in this reader. If another change
provides the reader first, integrate that implementation instead of duplicating it.

CR2 already has storage; keep CpuState and its embedded legacy layout unchanged
unless an additional architecturally visible field is demonstrably necessary.
Any necessary CpuState extension is append-only at the **end of CpuState**;
growing/reordering the embedded legacy struct would shift later ABI fields.
TLB entries, generations, block proofs and translation plans belong in Vcpu's
Machine execution state. They are derived engine state, not guest registers.

An instruction checkpoint restores architectural destinations but preserves the
new fault's CR2 and completed page-walk writes. A full Machine snapshot includes
CpuState (therefore CR2/CR3), physical RAM/page tables, board/device state,
pending events and existing in-flight continuations. Restore at a quiescent
boundary, discard TLB/block proofs and reconstruct derived translation state.
Saving CpuState alone is not a complete snapshot of an in-flight MMIO operation.
If a mid-handoff snapshot is supported, preserve its stable span plan and device
completion identities too; otherwise require completion/explicit abandonment
before taking it. `reset_execution_state` must clear derived paging state while
preserving its existing architectural-register contract.

### 3.8 Default-path performance requirement

Acceptance requires **at most +0.5% host instructions per guest step with PG=0**,
against the implementation's base revision. Selection outside the existing inner
memory loop, no enlarged CpuState, and Machine-only paged storage are design
constraints, not proof of the result. Application Long64/Compat32 and the frozen
Real16 alias keep their paths; inspect generated code if shared generics change.

Use `scripts/perf-campaign.sh BASE` in the pinned dev shell for the existing
release/perf profiles, all unpaged Machine/alias/Application paths and counted
store probe described in [`commands.md`](../commands.md). Add equivalent
protected P16/P32 and both 486 PG=0 instruction-count cells where the campaign
does not cover them. Compare each cell and process spread; do not hide a regression
in a workload average or substitute elapsed time for retired host instructions.
Repeat a near-threshold cell with the campaign's established replication protocol.
A persistent >0.5% regression blocks landing and requires changing the dispatch
seam. PG=1 performance and TLB capacity have no optimization acceptance target.

## 4. Witness plan

Tests assert manual-derived results directly, with source section/page beside
each witness. Unicorn or hardware comparisons may falsify a claim but do not
define it. No test needs a manual, network, host privileged execution or a
particular emulator at runtime. Use small public, guest-agnostic Machine programs
and byte-defined tables; inspect both the outward fault and an InGuest handler.

| Witness group | Required distinguishing cases |
|---|---|
| Walk and precedence | Nonidentity CR3 mapping for data/fetch; distinct PDE indices and PTE indices; P=0 independently at each level with other bits filled; segment limit/type failure before page walk; PG/PE invalid combination leaves control state unchanged. Physical table hole is an outward bus failure, not #PF. |
| Permissions/model | All 16 PDE/PTE U/S/R/W combinations × CPL 0/1/2/3 × read/write/fetch. Separate 386 and both 486 models, WP off/on where present. Pin that 386 supervisor writes ignore R/W; 486 WP write faults have P=1/W=1/U=0. Include read-only supervisor mappings to witness Q3. |
| Error and CR2 | Low-bit errors 0/2/4/6 for not-present accesses and 1/3/5/7 for the corresponding protection classes where reachable; no fake protection case for an always-allowed supervisor read. Check metadata known-mask `0x7` and deterministic upper zeros. Cross-page CR2 names the first failing required byte; implicit access at CPL3 reports U=0; non-#PF faults leave CR2 unchanged. |
| A/D | Fetch/read sets PDE.A/PTE.A without D; allowed write sets PTE.D before operand storage; preserve PDE.D and software bits; 486 protection #PF sets both A bits; PTE-not-present may leave PDE.A, never modifies the not-present PTE. CAS-race injection preserves competing frame/permission/software edits. Clear A/D with/without invalidation; first write after a read-only TLB fill sets D. |
| Restart | Boundary scalar load/store and aliased register-source RMW; full stack frame and system multi-read; REP with a completed iteration followed by #PF; zero-count REP; locked noncontiguous-frame operand; x87 10-byte operand and save/environment image. Assert unchanged operand destinations on fault and correct one-time completion after repair. Allow the specified A/D effects. |
| Fetch and blocks | One-byte instruction at `L+0xfff` with next page absent; instruction genuinely split across two mapped nonadjacent frames; needed second-page #PF; same L under two CR3 mappings; equal CR3 reload after a PTE edit; INVLPG, PG/WP changes and embedder invalidation. Same bytes in a different physical frame must not reuse old source coverage. |
| System/delivery | GDT/LDT/IDT/TSS straddling pages; descriptor accessed/busy write to a read-only page under WP; supervisor bitmap permission read on a supervisor page at CPL3, including bitmap #PF before any port effect; 16/32-bit #PF error frames; fault while delivering #PF updates CR2 and reaches the existing #DF/shutdown rules. |
| Task CR3 | Supported switch between two page directories, with GDT/LDT bytes differing at identical linear addresses; outgoing saves use old mapping, incoming descriptor/fetch uses new mapping; equal incoming CR3 invalidates conservatively; PG=0 leaves incoming offset `0x1c` unread. Assert pre/postcommit fault ownership. |
| Physical aliases/SMC | Two linear pages map one code frame: execute through both, write through either, observe fresh code through both. Cross-page fetch tags both frames; a write during source publication cancels it. Include A/D table/code alias and distinct physical HostRam aliases under the existing declaration contract. |
| MMIO/backing/A20 | Paged operand to RAM, ROM, MMIO, physical hole and HostRam-recovery fault; second-page #PF occurs before any first-page device callback; successful split-device retry produces each completion once, including two aliased stores. Table MMIO/ROM gives the documented typed refusal. Gate PDE read, PTE read, A/D CAS and final fetch/data separately; toggle A20 between runs and during a held retry. Repeat core cases with sparse backing without committing a dense 4 GiB host allocation. |
| Compatibility/performance | Retain existing PG=0 model/register/INVLPG results, Application tests and fast-tier gates; no new typed refusal in previously supported unpaged instructions. Record the +0.5% campaign verdict separately from functional success. |

Extend the census with opt-in paged P16/P32 setups using nonidentity RAM mappings
and explicit fault/refusal expectations. Keep the unpaged runs and artifact
columns so the before/after refusal counts remain comparable. Run the existing
ignored `legacy_census_386::runtime_census` and `runtime_census_i486` tests in
the `legacy` test binary with `--ignored --nocapture`; run the new paged matrix
separately. Opcode coverage cannot substitute for page-crossing, atomicity,
handler-frame or alias witnesses. Record actual measured results during
implementation; none are claimed by this design note.

## 5. Reviewable implementation sequence

Each step keeps the existing PG=0 adapter and Application paths intact and must
pass the default-path performance gate. Intermediate support is explicitly
internal/test-only: retain public PG entry/write refusal until the complete
required contract is ready. A PG=1 partial path must never silently execute an
untranslated access. After public enablement, any still unsupported board/device
operation uses an explicit typed refusal.

1. **Translation and data faults.** Add the central physical walk and paged
   adapter, typed #PF producer, CR2-after-rollback handling and span preflight for
   scalar data. Include minimal correct A/D behavior from the start; it cannot be
   postponed while claiming architectural paging. Exercise through private
   paged tests while public PG=1 remains refused. Validate permissions first with
   pre-set A/D entries as well as basic update witnesses.
   The private entry refuses instructions with multiple operand addresses until
   instruction-wide preflight is available. Split operands involving a device
   receive a typed refusal before transfers or D updates until step 5 retains
   their translated plan and operand snapshot across bus handoffs.
2. **Fetch and translation identity.** Add incremental paged fetch, private
   one-instruction block key/proofs and physical source registration; wire CR3,
   PG and embedder flushes. Cover cross-page fetch, two directories at one L,
   equal reload and source-publication races. No paged block reaches Direct/JIT.
3. **A/D completion, 486 controls and invalidation.** Complete CAS race/retry,
   fault-side-effect and clean-TLB-to-write witnesses; 486 WP and addressed
   INVLPG with a separate IR form; PG/WP/A20 generation changes. Preserve the
   386 model's masks and #UD for INVLPG. This completes, rather than first adds,
   the A/D contract established in step 1.
4. **All operand/system consumers.** Route string iterations, stack frames,
   locked split operands, x87 buffers, descriptor CAS, IDT/TSS/privilege accesses
   and the minimum I/O bitmap reader through the adapter. Integrate incoming
   TSS CR3 at the existing task commit seam. Validate outward/InGuest #PF,
   nested delivery and checkpoint behavior. Keep task-switch expansion out.
5. **Aliases, bus handoffs and public enablement.** Finish physical alias/SMC,
   operand-MMIO continuation plans, A20, sparse/HostRam and restore witnesses;
   pin the table-device refusal. Enable PG=1 for all three models only after the
   previous steps' witnesses and paged census pass. Run scoped fast tests,
   required lint/format checks and the +0.5% campaign; the supervisor runs the
   project's full landing gates. Remove the old blanket PG refusals only here.

An uncached walker is also a correct initial implementation if that materially
reduces review size. It must retain the same invalidation-generation and block
identity rules even when there are no TLB entries to clear. Neither that choice
nor an early functional TLB justifies adding an optimized software MMU to the
Application path.

## 6. Maintainer questions and recommended answers

| ID | Open question | Recommended answer |
|---|---|---|
| Q1 | Must PDE/PTE storage in MMIO/ROM work in the first release? | Require ordinary RAM tables initially. Return a named typed paging-table facility refusal for non-RAM storage and preserve physical missing-memory exits. Support operand MMIO normally. General table-device support needs an atomic-device bus contract and its own witnesses. |
| Q2 | Which A/D intermediate states should be promised on failed walks? | Adopt §3.3's deterministic order; require the documented 486 A-on-protection-fault behavior, preserve PDE.D and do not undo A/D on operand faults. Mark the less-specified 386 choices as engine policy and test them directly. |
| Q3 | Does 486 WP protect supervisor mappings too, despite the older PRM's user-page wording? | Yes: use the SDM's explicit R/W-AND rule for both mapping classes. Retain the wording qualification and pin both classes with independent witnesses; record any measured historical-chip disagreement before changing that rule. |
| Q4 | How much TLB/block machinery is necessary? | Use a small per-Vcpu functional TLB, no upper-level walk cache, full flush for INVLPG initially, and one-instruction paged Checked blocks with fresh fetch validation. Accept an uncached first version if simpler. Gate all changes on the PG=0 +0.5% instruction-count budget. |
| Q5 | How can an embedder invalidate after equal-byte state edits or checkpoint restore? | Add an explicit paging-invalidation operation, document it for page-table edits/restore, and automatically reconcile changed control/map state at positive entry. Keep TLB/proofs outside CpuState; a CPU-state copy alone cannot restore a whole Machine. |
| Q6 | Should paged task switching and the currently missing bitmap reader be deferred? | Include the CR3 handoff for existing 386-TSS switch forms and the minimum supervisor bitmap-read path required here. Integrate a sibling implementation if available; otherwise keep PG publicly refused until these system accesses are covered. Add no other task-switch forms. |
| Q7 | Can partial PG=1 support be exposed between implementation steps? | Keep it internal/test-only until fetch, every operand/system consumer, exception delivery and bus/alias behavior agree. This makes each step landable while preventing successful but untranslated guest accesses. |

## 7. Maintainer decisions (2026-11-24)

The maintainer accepted every recommended answer in §6 (Q1-Q7): page tables in
ordinary RAM only for the first release, with a typed refusal otherwise; the
deterministic A/D order of §3.3, with the 486 setting A on a protection fault and
the 386 choices labelled engine policy; 486 CR0.WP protecting supervisor and user
mappings alike; a small per-vCPU functional TLB, a full flush on INVLPG and
one-instruction paged blocks; an explicit embedder paging-invalidation call; the
CR3 handoff on 386-TSS switches and the I/O bitmap read through the adapter; and
PG=1 publicly refused until step 5 of §5 enables it.
