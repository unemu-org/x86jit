---
id: decision-32
title: 'Paging and privilege in compiled Machine blocks: paged Flat32 at CPL 0 and CPL 3 (J3+J4)'
date: '2026-11-27'
status: accepted
---

Decider: maintainer (2026-11-27), accepting every recommended answer (Q1-Q11).
Drafted 2026-11-27 for bead xj-to9.3, the merged stage J3+J4 that decision-27's amendment of
2026-11-26 accepted (Q2). It builds on decision-29 (J1) and decision-30 (J2, in review at
b409168b). It changes four accepted texts, and each change is also a question below: the miss
path of decision-27 amendment §3.2 for paged code (Q1), the relation of decision-30 §4's
compiled view to the TLB (Q2), decision-27 §6's J3+J4 done-when (Q10), and the order in
decision-30 Q8 (Q11). Everything else in decisions 20, 23, 27, 28, 29 and 30 stands. No code
changes with this record. Nothing here was measured: the numbers are from performance.md
("Interpreter ceiling study 2") and decision-30's implementation notes.

# Paging and privilege in compiled Machine blocks (J3+J4)

## Context

J2 compiles one class: Flat32, PG = 0, CPL 0, entered from `BlockWalk::Direct`
(`vm/legacy/compiled.rs:397-401`). A late-1990s desktop operating system spends its hot time
outside that class. It runs flat 32-bit code with paging on: its kernel and drivers at CPL 0,
and its applications at CPL 3. Its 16-bit components and V86 boxes are a smaller share. They
stay interpreted until J5 (decision-27 amendment §6).

### What the interpreter does today (b409168b)

| Piece | Where | Behaviour compiled code must reproduce |
|---|---|---|
| `Tlb`: 64 entries, folded index, per-vCPU | `interp/legacy/paging.rs:382-420` | Each entry holds page, frame, PTE, rights = PDE & PTE, and `fetched` |
| Invalidation | `:422-444` | `reconcile` empties the TLB on a change of CR3, CR0 & E0010001h (PG, CD, NW, WP, PE), A20 or the model's flags mask, and issues a new token |
| Hit rules | `translate_linear`, `:829-960` | A hit without rights faults and removes the entry (386), or re-walks to set both A bits first (486). A write to a clean entry removes it and re-walks |
| A/D | `table_or` CAS, `complete_dirty` `:971-1005` | A on both levels before the TLB caches the translation; D in the PTE before the first write |
| Rights | `permitted`, `:518-524` | User needs U. A user write needs W. A supervisor write needs W only with WP set, and only on the 486 (the 386 has no WP) |
| #PF | `PageFault::exit` `:636-666`, `publish_cr2` `:668` | Error code P/W/U, CR2 = linear, published at delivery |
| Fast lanes | `fast_window` `:2763-2800`, `tlb_lane` `:3023` | Use only existing translations. A miss "must not walk, update A/D, consume a device answer or change fault precedence" (`:2745-2746`); it takes the two-phase transaction path |
| Paged proofs | `vm/legacy/paging.rs:14-23`, `:179-185` | Keyed by the complete context stamp (decision-20 amendment), CR3, CR0 and the TLB token. **A new token clears every paged proof** |
| Paged lift | `vm/legacy/paging.rs:395-484` | At most 32 instructions, ends at the page end. A block that starts in the last 15 bytes may hold one page-crossing instruction |

### Measured baseline (performance.md, ceiling study 2, I80386, release)

Host instructions per guest step: `machine-paged32` 593 / 881 / 1481 (register / memory /
callret), `-cpl3` 596 / 884 / 1484. `machine-paged16` CPL 3 `segmix` 1458, `v86gp` 1434. The
unpaged interpreter costs 422 / 564 / 1042. J2 compiled, background T1 = 50: 124.5 / 138.5 /
370.3 (decision-30 implementation notes), with a linked compiled turn of about 530. On
pg32 `memory`, the paged lane choice takes 22 % of samples. Compiled code removes it.

## Decision

### 1. Principle

J3+J4 adds no new paging semantics. Compiled code completes an access inline only when the
interpreter's fast lane would complete it from an existing translation: a hit with the rights
for this CPL, D already set for a write, and a RAM (or, for reads, ROM) frame. Every other
access DEFERs at its instruction's start, as in J1 and J2. The interpreter then re-executes the
instruction and runs the rest of the block. Its walk sets A and D in its own order, raises #PF
with its error code, and its delivery publishes CR2. The DEFER is exact for the reason J1 gives
(SDM Vol 3A §7.5 p. 7-4; 80386 PRM §9.1 p. 9-2), and here it is also cheap to prove: compiled
code never writes a paging-structure entry, so it cannot set an A or D bit out of order.

### 2. Admitted class and entry contract

A turn may enter a compiled unit when decision-29's E1 and E3-E8 hold and E2 reads:

- the class is Flat32 (`legacy::flat32_cache` on CS/DS/ES/SS), the regime is protected (never
  V86), CPL is 0 or 3, and PG is 0 or 1;
- the walk is `BlockWalk::Direct` (PG = 0) or `BlockWalk::Paged` with an ordinary block
  (`vm/legacy/paging.rs:704-713`);
- the block's bytes lie in one 4 KiB linear page and one bus frame. A block whose lift
  fetched across a page stays interpreted (decision-23 §8, decision-27 amendment §1).

Two conditions are added:

- **E9.** The paged state has no outstanding operand transaction or port completion
  (`paging::State::transaction`, `port_input`, `event_transaction` are `None`), the paged
  analogue of E5.
- **E10.** The unit was reached through a paged proof hit or a link that the paged path
  validated this turn. Both carry the TLB token (`Key::generation`; the link's paging identity,
  `links.rs:95`). A token is reissued whenever a fetch-used entry is removed or replaced and on
  every `reconcile` change. So a valid proof or link means that the fetch translation of this
  page is the one the unit was compiled from: **the token is the entry check** (decision-27
  amendment §1, Q1). The compiled key does not carry it.

CPL 1 and 2 stay interpreted (Q5): no target workload runs there, and each CPL needs its own
witnesses. Unpaged CPL 3 Flat32 is admitted, because the identity fill grants user rights
(no page protection exists with PG = 0).

### 3. The compiled TLB probe

**One view, a subset of the TLB.** The vCPU keeps J2's `MachineTlb` (`jit_abi/machine.rs:66`),
256 slots, direct-mapped on `(linear >> 12) & 255`, the same probe for PG = 0 and PG = 1. With
PG = 1 it obeys one invariant: **every valid view entry equals a current entry of the vCPU's
`Tlb`**: same page, same frame, rights and D bit as cached there. The view is never a second
TLB (decision-23 §8). It is a cache of the TLB's answers, and so a compiled hit returns what
the interpreter's fast lane returns from the same TLB state. Two rules maintain the invariant:

1. **Only the miss helper fills, and only from a TLB hit.** Compiled code never walks.
2. **Every `Tlb` mutation clears the view slots it touches.** `insert` clears the slot of the
   page it evicts and of the page it adds. `remove` and `dirty` clear their page's slot.
   `invalidate` and `undo_lane` flush the view. Clearing is always safe: the invariant is a
   subset, not equality.

The paged `Tlb` gets a handle to the vCPU's view (it lives in `vm/legacy/paging.rs` `State`, the
view in `compiled::State`). The view's `reconcile` adds PG to its A20-and-mask test, so a PG
flip flushes it in either direction.

**Entry layout** (`FastEntry`, `repr(C)`, grows from 32 to 48 bytes, padded to 64,
`ENTRY_SHIFT` = 6, 16 KiB per vCPU):

```text
FastEntry {
    sup_read_tag:   u32,  // page + 1 if a supervisor read may go inline
    sup_write_tag:  u32,  // page + 1 if a supervisor write may go inline
    user_read_tag:  u32,  // the same at CPL 3
    user_write_tag: u32,
    addend:   u64,        // host address of the frame minus the linear page base
    tag_word: u64,        // the frame's live code-tag word (J2)
    frame:    u32,        // bus frame number, for the note helper
    tag_bit:  u32,        // the frame's bit in tag_word
}
```

CPL is in the key, so the lowering picks the tag pair at compile time, and the U/S check costs
nothing at run time (Q4). The fill computes the four tags with the interpreter's own
`permitted` (`paging.rs:518`) for the CPU state at fill time:

| Tag | Set when |
|---|---|
| read (S) | the TLB entry exists, and the frame is RAM or ROM in the backing |
| write (S) | `permitted(Write, Supervisor)`, PTE.D set, and the frame is RAM with a code-tag word. With WP = 0, or on the 386, a read-only page is supervisor-writable (80386 PRM §§6.4.1-6.4.2 pp. 6-20-22, Table 6-5; SDM Vol 3A §5.6.1 pp. 5-33-35) |
| read (U) | `permitted(Read, User)`: U set in both levels |
| write (U) | `permitted(Write, User)`: U and W set in both levels, and D and RAM as above |

**Dirty-known.** It needs no separate bit: no write tag exists until the cached PTE has D set.
The first write to a clean page misses into the helper and DEFERs. The interpreter sets D
(80386 PRM §5.2.4.3 p. 5-10; SDM Vol 3A §5.8 p. 5-39), `Tlb::dirty` clears the slot, and the
next access refills it with a write tag.

**No generation in the entry.** View slots are cleared on the vCPU's own thread, in the same
call that mutates its `Tlb`. Another vCPU never touches them (§3, "Cross-vCPU"). A flush costs
a 16 KiB clear per CR3 load. If the census shows that matters, a 256-bit fill mask clears only
the filled slots.

**Inline load at CPL c (size s, linear a):** `e = view[(a >> 12) & 255]`; if
`(a & 0xfff) > 0x1000 − s` or `e.<c>_read_tag ≠ (a >> 12) + 1` → helper; else
`load(e.addend + a)` with the tier's ordering. This is J2's probe with one offset changed.
**Inline store:** the same probe on `<c>_write_tag`, then J2's store barrier unchanged (the
live code-tag bit, the watch gate, `machine_note`).

**The miss helper, PG = 1** (`machine_jit::access`, with the paged arm):

| Case | Helper |
|---|---|
| The TLB holds a translation that passes `fast_window` (`paging.rs:2763`: hit, rights for this CPL, D for a write, RAM) | Completes the access and fills the view slot. This is the case where only the view missed (a slot conflict or a cleared slot) |
| The same, the frame is ROM and the access is a read | Completes and fills (a read tag only) |
| TLB hit to a Trap frame with an in-process handler for this run, rights pass, D set for a write | decision-28 in-block contract (`vm/device.rs:1164`), with `linear_address = Some(a)`, which is what the paged exit path reports (`PortPlan::pending_mmio_linear_address`) |
| TLB miss; rights fail; a clean write; ROM write; hole; crosses a page; Trap otherwise | **DEFER** at the instruction start. Nothing was written: no walk ran, no A/D bit changed |

The helper never calls `translate_linear` (Q1). The re-execution performs the walk, sets A on
both levels before it caches the translation (SDM Vol 3A §5.10.2.3 p. 5-43), and sets D before
the write. For an instruction with two accesses (CALL [m], PUSH [m], RET), the two-phase
transaction (`paging.rs:1677`, `execute_with_retry`) keeps the walks in instruction order and
gives the right fault precedence. A 486 protection fault re-walks to set both A bits
(Intel486 PRM §9.9.14 p. 9-21, as `paging.rs:895-900` cites it). A 386 protection fault leaves
them unset, which `paging.rs:898-902` records as ASSUMED. All of that is the interpreter's, unchanged.

**#PF and CR2 from compiled code.** None are raised there. A DEFER returns
`RET_MACHINE_DEFER` with ordinal k. The dispatcher walks instruction k onward on the paged
interpreter (`execute_flat32` from k's op index). That walk raises #PF with error code
P = protection, W/R = write, U/S = CPL 3 (80386 PRM §9.8.14, Fig. 9-8 pp. 9-17-18; SDM Vol 3A
§5.7 pp. 5-37-38; no I/D or RSVD bit on these models). It removes the faulting translation
(SDM Vol 3A §5.10.4.1 p. 5-49), charges k (decision-29 §4), and its delivery writes CR2 through
`publish_cr2`. Implicit supervisor accesses at CPL 3 (descriptor tables, TSS, IDT, the stack
of a privilege change) occur only in deferred or singleton instructions. Every access a
compiled block makes at CPL 3 is therefore a user access, so baking the user tag pair is exact.

**Invalidation.** Compiled code changes no paging state. Every invalidating instruction is a
singleton or a DEFER, and it reaches the view through rule 2:

| Event | Architectural effect | Path |
|---|---|---|
| MOV to CR3; task switch with a new CR3 | flushes the TLB (80386 PRM §5.2.5 p. 5-11; SDM Vol 3A §5.10.4.1 pp. 5-48-49) | interpreted control op → `reconcile` → `invalidate` → view flush |
| INVLPG (486) | removes that page's entries (SDM Vol 3A §5.10.4.1 p. 5-48) | interpreted → `Tlb::remove` → slot cleared |
| MOV to CR0 changing PG, WP, PE, CD or NW | PG 1→0 flushes. The others may flush: "the processor is always free to invalidate additional entries" (§5.10.4.1 p. 5-49) | `reconcile` already flushes (`paging.rs:429-445`). That makes WP a fill input (decision-27 amendment §1) |
| A20 change, model flags | the bus changes | `reconcile` and the view's own A20/mask test |
| #PF | removes the faulting page's entries (§5.10.4.1 p. 5-49) | the interpreter's `remove` |
| A guest store to a PTE with no invalidation | none: the TLB may keep the old entry (SDM Vol 3A §5.10.2.3 p. 5-43). A D bit cleared by software may stay unset on later writes (§5.8 p. 5-39) | none in either tier. Both tiers share one `Tlb`, so they see the same stale entry |

**Global pages and CR4.** No catalogued model has CR4 (`paging.rs:434`; `CpuModel` ends at
`I80486DxCpuid`). Global pages need CR4.PGE (SDM Vol 3A §5.10.2.4 p. 5-44), and 4 MiB pages
need CR4.PSE. Neither exists here. A later model with them gets its own record. Its global
entries would survive CR3 loads in the `Tlb`, and the view would follow without change,
because it is a subset of the `Tlb`.

**Cross-vCPU.** There is no architectural TLB shootdown. Each logical processor's TLB is
private, and the operating system propagates a paging change by having every processor
invalidate (SDM Vol 3A §5.10.5 pp. 5-52-53). Each vCPU's `Tlb` and view are private, so
x86jit sends nothing between vCPUs for paging. What is shared is physical: A/D updates by CAS
(`table_or`), and code-tag freshness. A stale write translation on one vCPU that reaches a
frame another vCPU has since tagged as code still tests that frame's live bit and calls the
note helper. Units are shared and physical, and their invalidation advances the epoch that
every vCPU's slot checks (decision-30 §2).

### 4. Code: one page, the fetch, SMC

- **Key.** `MachineBlockKey` is unchanged (decision-30 §2). `physical_pc` is the paged lift's
  first bus address, which the paged resolve returns with the block (today it computes the
  unpaged form, `compiled.rs:453-461`). The flags already carry PG, CPL, CR0.AM and A20
  (`compiled.rs:285-293`). **The key must name every lift input.** The paged lift reads the CPU
  only through the segment class (`specialize_segments`, `vm/legacy/paging.rs:450`) and the
  alignment mask (`alignment_mask(&cursor)`: CPL, CR0.AM). The complete stamp that paged
  proofs keep (decision-20 amendment) is a conservative private identity, not a lift input.
  So a kernel page shared by two address spaces, at the same linear address and the same
  frame, has one unit under every CR3.
- **Unit reuse after a CR3 load.** A new token clears every paged proof
  (`vm/legacy/paging.rs:179-185`). Each block is then relifted and gets a new
  `Arc<IrBlock>` and a fresh memo slot. With J2's memo, the slot would count to T1 again before
  it looked in the Machine map. In J3+J4, **a new memo slot looks the key up once** on its
  first eligible turn: one cold read-locked lookup per lift, against a lift that costs
  thousands of host instructions. If the unit exists, the slot enters it at once. Otherwise it
  counts to T1 as before (Q7).
- **One page.** The paged lift already ends at the page end (`paging.rs:404-480`). Admission
  rejects any block whose sources span two frames. Cross-page transfers return to the
  dispatcher, whose link carries the next page's token.
- **Page-crossing data** DEFERs in paged code. Two translations would need the transaction's
  walk order.
- **SMC** is decided by J2's live code-tag bit on the frame. Tags are physical, so a store
  through a second linear mapping of a code frame finds the same bit as a store through the
  code's own mapping. The note helper then invalidates the units and proofs that cover the
  bytes. If the store hit the running block, it STOPs after the storing instruction
  (decision-29 §4; SDM Vol 3A §14.6 p. 14-18, as the interpreter applies it). A/D CAS writes
  into a frame that also holds code take the interpreter's existing path (`resolve`,
  "A/D CAS can touch executable table storage"). Compiled code makes none.

### 5. Privilege

- **CPL in the key** (decision-27 §1): a unit compiled at CPL 3 is never entered at CPL 0.
  `io_permitted` (CPL ≤ IOPL) stays in the key, and no compiled op reads it yet.
- **CPL changes end blocks.** INT n, INTO, INT3, BOUND, IRET, far CALL/JMP/RET and task
  switches are not `flat32_ordinary` (`interp/legacy.rs:2690`), so each is an interpreted
  singleton. Interrupts and exceptions are delivered only at boundaries or after a DEFER
  (decision-29 §5). A compiled block never changes CPL, CR3 or any paging input.
- **Privileged and IOPL-sensitive instructions** are singletons or unconditional DEFERs.
  These are MOV CR/DR, LGDT/LIDT/LLDT/LTR, LMSW, CLTS, INVD, WBINVD, INVLPG and HLT (SDM Vol 3A
  §6.9 pp. 6-23-24; 80386 PRM §6.3.5.1 p. 6-17), and IN, INS, OUT, OUTS, CLI and STI (SDM Vol 1
  §20.5.1 pp. 20-3-4; 80386 PRM §6.3.5.2 p. 6-18). PUSHF and POPF DEFER as in J2. At CPL 3 the
  interpreter raises the #GP.
- **User-mode segment loads** (MOV/POP to DS, ES, FS or GS, LDS, LES, LFS, LGS) stay in
  ordinary blocks (`data_segment_load`, `interp/legacy.rs:2716`) and DEFER at the load, as in
  J2. The rest of that block runs interpreted. SS and CS loads end blocks.

### 6. The op set

J2's set (decision-30 §3) at both CPLs, plus two ops the target uses at CPL 3 (Q6):

| Op | Lowering | Exact because |
|---|---|---|
| `LegacyOp::I486Gate` (486, CPL 3, ops with an alignment mask; `lift/legacy.rs:1575-1593`) | Test CR0.AM and EFLAGS.AC live. If both are set, DEFER | The gate precedes every effect of its instruction (`interp/legacy.rs:2616`). #AC needs CPL 3, AM and AC (SDM Vol 3A §7.15, Event 17, p. 7-53), and the interpreter raises it. With AC clear the instruction completes inline |
| `LegacyOp::Address` for FS and GS (Flat32 leaves them live) | Load the cache: unusable, attributes, limit, base. Inline only if it is usable, a data segment (not code), expand-up, writable for a store, and `offset + bytes − 1 ≤ limit` without overflow. Then linear = base + offset, mod 2^32, into the probe. Otherwise DEFER | It is the passing subset of `legacy::access` (`legacy.rs:1655-1690`; SDM Vol 3A §§3.4.5.1, 6.3-6.4; 80386 PRM §§6.3.1.1-6.3.1.2 pp. 6-4-6). Every failure, and every expand-down or code-segment case, is left to the interpreter's #GP/#SS. A sum that crosses 4 GiB crosses a page and DEFERs |

Still deferred: strings and REP, x87, PUSHA/POPA, ENTER, LOCK forms, XCHG with memory,
CMPXCHG, XADD, segment loads, 16-bit SHLD/SHRD (decision-30 §3). The embedder census (§8)
chooses the next family.

### 7. Code that changes

| Place | Change |
|---|---|
| `jit_abi/machine.rs:33-125` | `FastEntry` with four tags (64 bytes, `ENTRY_SHIFT` 6). Identity fill sets both pairs. Paged fill from a `Tlb` hit. PG in the view's `reconcile` |
| `interp/legacy/paging.rs:396-516` | `Tlb` holds the view handle; `insert`, `remove`, `dirty`, `invalidate`, `undo_lane` clear its slots. A debug assertion after every helper fill checks the invariant |
| `interp/legacy/machine_jit.rs` | Paged arm of `access` (`fast_window`, then fill, in-block or DEFER). `lowered_op` adds `I486Gate` and FS/GS `Address`. The remainder walk runs `execute_flat32` (paged) from an op index |
| `vm/legacy/compiled.rs:397-401`, `:420-530` | Admission (§2, E9, E10). The paged `physical_pc`. A one-time Machine-map lookup on a fresh memo slot |
| `vm/legacy/paging.rs:226-375`, `:665-715` | `resolve` returns the block's first bus address. The paged turn calls `machine_progress` before `execute_paged` |
| `x86jit-cranelift/src/codegen/machine.rs` | Tag offset by CPL. `I486Gate` and FS/GS `Address` lowerings |
| `x86jit-bench/examples/machine32_loops.rs` | `machine-paged32-bg`, `machine-paged32-cpl3-bg`, and a `cr3switch` workload: two address spaces that share kernel pages, with a CR3 load every N iterations |

### 8. Expected costs and the workloads this unlocks

**Model** (decision-27 §5, decision-30 §9): per step = J2's measured compiled cost + t·ΔT +
a·ΔP. Here t is turns per step (0.2 / 0.2 / 0.6) and a is accesses per step (0 / 0.4 / 0.4).
**ASSUMED:** ΔT, the paged turn's extra over the unpaged compiled turn (the paged resolve,
reconcile and link identity), is **50-200** host instructions. The interpreter's paged-minus-
unpaged difference is 171 per step on `register`, and the ceiling study puts most of it in
per-instruction lane choice, which compiled code does not run. ΔP, the probe's paged extra, is
**0**: the probe is the same instructions at another offset. Misses are not in the steady
state of these loops.

| Workload | Interpreted paged32 | J3+J4 expected, no chaining | ÷ | decision-27 §5 paged range |
|---|---:|---:|---:|---:|
| register | 593 | 135-165 | 3.6-4.4× | 70-182 |
| memory | 881 | 148-178 | 5.0-6.0× | 97-213 |
| callret | 1481 | 400-490 | 3.0-3.7× | 226-566 |

CPL 3 costs the same (the interpreter differs by 0.5 %; compiled code differs by one offset).
J2b's chaining, if it lands first, takes t·T down in both tiers alike.

**Miss cost (ASSUMED).** A TLB miss DEFERs. The interpreter then walks, fills and runs the
rest of the block at paged interpreter cost: about 1,500-3,000 host instructions per miss. After
a CR3 load, every touched page misses once. After a code page's token changes, every paged
proof relifts once. That relift, not compiled code, is the cost that grows with the switch
rate (Q8).

**What it unlocks.** These are the hot paths of a late-1990s desktop OS: user code at CPL 3,
paged and flat, and kernel and driver code at CPL 0, paged and flat. J3+J4 also covers DOS
extenders that run their clients at ring 3 without paging, which is unpaged CPL 3. Cells that
do not change: `machine-paged16` CPL 3 `segmix` (1458), `machine-paged-v86` and `v86gp`
(1434), and every Segmented and V86 path. They stay interpreted until J5. The interpreter pays
for the view only where it mutates its `Tlb` (a slot clear per insert or remove, and a flush
per invalidate). That is **ASSUMED** at ≤ 0.5 % on the paged cells, and the perf gate checks it.

**Embedder census (as decision-30 §9, not a gate).** In a test-hooks copy of a PC-class
embedder running that OS: the share of retired instructions completed in compiled code, split
by CPL; the DEFER census by cause, with the new causes (TLB miss, clean write, rights, AC
gate, FS/GS refusal, page crossing); CR3 loads, paged relifts and TLB misses per million
steps; and host instructions per retired instruction before and after. Decision-27 §6's
"≥ 35 M retired/s" for the embedder's paged code becomes this census's target (Q10). At about
150 host instructions per step, 35 M retired/s needs about 5.3 G host instructions per second,
one performance core.

### 9. Witnesses and done-when

**Modes** are decision-30 §10's five (interpreter only; Forced; Delayed(1), Delayed(3);
background production at T1 = 1, 3, 50; Scripted), on x86-64 and ARM64, for the I80386 and the
I80486Dx. **Lockstep** (`machine_lockstep.rs`, full state including CR2, all RAM including
the page tables, `StepRun`, retired counts, the delivery log, and `Tlb::entries_for_test`
compared after every `run_steps`) gains paged and CPL-3 variants of every J2 case, plus:

- **#PF at every ordinal** of a compiled block, for each cause: PDE not present, PTE not
  present, user read and write of a supervisor page, user write of a read-only page,
  supervisor write of a read-only page with WP = 1 (486) and WP = 0, and the same on the 386
  (no WP). Each for a load, a store, a push, a pop, CALL [m] and RET, with the error code and
  CR2 compared, and with the 386/486 A-bit difference on protection faults;
- **A/D:** the first read sets A on both levels, the first write sets D, a write to a clean
  cached page in the middle of a block, a guest that clears A or D with no INVLPG (both tiers
  keep the stale entry), and page tables in a frame that also holds code;
- **CR3 loads mid-run** through `cr3switch`. The second address space reuses the kernel
  page's unit: no new compile, and the counter shows a compiled entry on the first execution
  after the relift. A page mapped differently in the two spaces gets its own unit;
- **INVLPG** of a data page in a loop, and of the running code page; PG, WP, CD and NW
  toggles;
- **U/S and privilege:** a CPL-3 loop calling the kernel through INT n and returning with
  IRET; privileged and I/O instructions at CPL 3 with IOPL 0 (#GP); `legacy_io_bitmap.rs`
  forced compiled;
- **AC gate** at CPL 3 on the 486 with AM and AC in all four combinations;
  **FS/GS:** null, limit, expand-down, code-segment and wrapping operands;
- **SMC through an alias linear mapping:** code at linear A, a store through linear B to the
  same frame, into another block and into the running block's next instruction, with TF clear
  and set. Also a store through a stale write translation into a frame that was retagged as
  code;
- **view invariant:** more than 64 and more than 256 data pages (conflicts and `Tlb`
  evictions), a `tlb_lane` replay (`undo_lane`), and the debug assertion on every fill;
- **paged in-process MMIO:** decision-28's census in modes 2-4. Positions and
  `linear_address` equal the interpreter's;
- **threads:** two vCPUs share page tables. One edits a PTE, and both run the guest's
  shootdown (an IPI-free flag protocol, then INVLPG). Plus D-bit races, and a store by one vCPU
  through its mapping into code the other runs compiled. Results equal the interpreter's
  under the same cross-vCPU schedule;
- **existing witnesses forced compiled:** `legacy_paging.rs`, `legacy_privilege.rs`,
  `legacy_task_switch.rs` (paged tasks), `legacy_code_alias_smc.rs`, `legacy_smc_selective.rs`,
  `legacy_alias.rs`, `machine_j2_devices.rs` paged, the `vm/legacy/paging.rs` unit tests
  (`:1164-1599`), `threads/same_block_smc.rs` and `threads/bg_tier.rs` paged.

**Done when:**

1. Every witness passes in every mode on both hosts. Forced and background witnesses report
   compiled-retired instructions > 0 at CPL 0 and at CPL 3, and a DEFER census that covers
   every new cause, so no pass is vacuous.
2. `machine32_loops` `machine-paged32-bg` and `machine-paged32-cpl3-bg` (T1 = 50, release and
   perf profiles) are inside decision-27 §5's paged range: register ≤ 182, memory ≤ 213,
   callret ≤ 566.
3. No cell that does not compile (paged16, `segmix`, V86, `v86gp`, real16, Segmented) is past
   the two-profile perf gate with compilation on or off. No Application cell is either, and
   the frozen ABI and `BlockKey` hash tests pass.
4. Production guards still reject Machine IR on Application entries, and `x86jit-core`
   depends only on `{iced-x86}`.
5. The embedder census (§8) is recorded.

## Options considered

| Choice | Options | Recommendation |
|---|---|---|
| Paged miss | (a) DEFER; the interpreter walks; (b) the helper walks and completes single-access instructions | **(a)**. It follows the fast-lane rule, and no new A/D or fault code is needed |
| View and TLB | (a) J2's 256-slot view as a subset of the `Tlb`; (b) a slot-for-slot mirror with the folded index; (c) a separate compiled TLB filled by its own walk | **(a)**. One probe for PG 0 and 1. (c) is the second TLB that decision-23 forbids |
| Rights | (a) a tag pair per privilege, chosen by the key's CPL; (b) one pair plus a run-time rights mask | **(a)**. Zero run-time cost |
| Fetch check | (a) the token through the proof or link (E10); (b) a translation probe at every compiled entry | **(a)**. It already holds and costs nothing |
| Unit reuse after relift | (a) a one-time lookup on a fresh memo slot; (b) count to T1 again; (c) adopt the unit's IR and skip the lift | **(a)** now, (c) after the census |
| FS/GS | (a) inline the passing subset; (b) DEFER, as J2 | **(a)** |

## Risks

- **The view drifts from the `Tlb`.** A missed clear makes a compiled access use a
  translation the interpreter no longer has, and that is a wrong result, not only a slowdown.
  Mitigation: one owner (the `Tlb` mutators), a debug assertion on every fill, the
  `Tlb::entries_for_test` comparison in lockstep, and conflict and replay witnesses.
- **Relift on every CR3 load.** Paged proofs are keyed by the token, so a process switch
  relifts every paged block, in both tiers. Compiled units survive (§4), but the lift remains.
  Mitigation: the census counts relifts per million steps. Q8 holds the fix.
- **TLB capacity.** The view cannot hold more than the `Tlb`'s 64 pages (256 KiB). Beyond
  that, misses DEFER at about 2,000 host instructions each. A 486 has 32 entries, but its
  miss costs cycles, not thousands of instructions. Mitigation: Q3.
- **Miss-heavy phases.** Right after a CR3 load, or with scattered data, the DEFER path runs
  often. Q1 (b) is the fallback if the census shows it dominates.
- **486 versus 386 paging details** (WP; A bits on protection faults). They stay correct by
  construction (the interpreter decides) and are checked per model in lockstep.
- **Undefined flags and model-only behaviour** are unchanged from J2. No new compiled op has
  undefined results.

## Maintainer questions

| # | Question | Options | Recommendation |
|---|---|---|---|
| Q1 | Paged view misses DEFER to the interpreter, and the helper never walks? This refines decision-27 amendment §3.2 ("a miss ... goes to one helper that runs the existing checked access") | (a) DEFER; (b) the helper walks single-access instructions | **(a)**. A/D order, fault precedence and CR2 stay in one place. Revisit with the miss census |
| Q2 | Is the compiled view a subset of the `Tlb`, filled only from `Tlb` hits and cleared by every `Tlb` mutation? This refines decision-30 §4, which proposed the view inside `Tlb` | (a) subset, J2's 256 direct-mapped slots; (b) a slot-for-slot mirror | **(a)** |
| Q3 | Grow the interpreter's `Tlb` from 64 to 256 entries (folded over 8 bits) as a separately gated first commit? The SDM allows any size (Vol 3A §5.10.2.2 p. 5-43: software "should not rely on the existence of TLBs or on the retention of TLB entries") | (a) yes, before the paged lowering; (b) keep 64 and decide from the census | **(a)**. It bounds the view, and it makes the token reissue of a replaced fetch entry rarer, which helps both tiers |
| Q4 | Per-privilege tag pairs in a 64-byte entry? | (a) yes; (b) one pair plus a rights mask | **(a)** |
| Q5 | Admit Flat32 at CPL 0 and CPL 3, paged and unpaged, with CPL 1-2 interpreted? | (a) yes; (b) all CPLs | **(a)** |
| Q6 | Add the `I486Gate` and FS/GS `Address` lowerings? | (a) both; (b) the gate only; (c) neither | **(a)**. CPL-3 code on the 486 carries the gate on every alignment-checked op, and FS-relative accesses are routine in user-mode code |
| Q7 | Does a fresh memo slot look up the Machine map once, so that compiled kernel code is entered on the first execution after a CR3 load? | (a) yes; (b) count to T1 again; (c) adopt the unit's IR without relifting | **(a)**. (c) needs the unit to keep its IR and the proof validation to change: a separate bead after the census |
| Q8 | Keep paged proofs across CR3 loads (an interpreter change: key by frame, not by token)? | (a) a separate bead, decided from the census; (b) part of J3+J4 | **(a)**. It changes the interpreter's paged identity (decision-20 amendment), and J3+J4 is correct without it |
| Q9 | In-process MMIO from paged compiled code through `in_block_access` with the linear address? | (a) yes; (b) always DEFER | **(a)**. A handled paged store costs about 21,000 host instructions interpreted (ceiling study 2) |
| Q10 | Done-when: decision-27 §5's paged range as the gate, and the embedder's 35 M retired/s as a census target rather than a gate? This changes decision-27 §6's J3+J4 row | (a) as proposed; (b) keep 35 M as a gate | **(a)**. The rate depends on the embedder's workload and host, which the library's gates cannot fix |
| Q11 | Order against J2b: decision-30 Q8 put J2b first. J3+J4 does not depend on chaining, and §8's estimates are inside the range at J2's measured turn | (a) land J2b first, as accepted, but start the `Tlb`/view work (Q3, §3) in parallel; (b) land J3+J4 first; (c) keep the order strictly | **(a)**. The view and `Tlb` changes touch other files than chaining does |

## Consequences

- The class that a late-1990s desktop OS runs most, flat paged code at CPL 0 and CPL 3,
  becomes compiled. The interpreter stays the reference for paging: compiled code only reads
  the TLB's answers.
- `FastEntry` grows to 64 bytes. `Tlb` mutators gain view clears. The paged turn gains the
  compiled call.
- J5 inherits the FS/GS segment-check lowering as its first Segmented piece, and a later
  P5/P6-class model inherits a view that already follows global entries.
- Revisit if the census shows TLB-miss DEFERs or relifts above about 10 % of host time (Q1
  (b), Q3, Q8), if CPL 1-2 code appears in a workload, or if paged cells miss done-when 2 by
  more than 2× (the turn, then J2b).
