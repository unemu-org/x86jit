---
id: decision-33
title: 'Compiled Machine blocks on sparse backing (amends decision-30 §10)'
date: '2026-11-28'
status: accepted
---

Decider: maintainer (2026-11-28), accepting every recommended answer (Q1-Q6).
Drafted 2026-11-28 for bead xj-to9.73, on top of J3+J4 (decision-32, 5293f6e9). It amends
one accepted text and one spec sentence, and each is a question below: decision-30's
implementation note 10 ("Sparse backing never compiles", the bead's "decision-30 §10"), and
spec §4.1's "Sparse backing executes **all** guest instructions through the checked
interpreter" (Q4). The Application profile's sparse rule is unchanged (Q3). Everything else in
decisions 27-32 stands. The implementation (§5) lands with this record; its numbers are
MEASURED.

# Compiled Machine blocks on sparse backing (amends decision-30 §10)

## Context

`Vm::sparse` and `Vm::with_backend_sparse` reserve a guest span with `PROT_NONE` (Windows:
`MEM_RESERVE`) and commit only what `map` maps (spec §4.1). A PC-class board for the 386 and
486 is a 4 GiB bus with a few committed regions (low RAM, extended RAM, option ROMs, the reset
ROM under 4 GiB), so sparse backing is its natural choice. `compiled::admits`
(`vm/legacy/compiled.rs:522`) refuses every sparse VM, so such a board never compiles a
Machine block. An embedder's flat-game probe measured 25-27 M steps/s on sparse backing against
65-70 M steps/s with owned backing and the JIT (84.9 % of steps compiled), a factor of 2.6.
The embedder would rather keep sparse backing: unmap returns host pages, a hole stays a hole
(no stray write can commit it), and the host commit charge is the mapped size, not the span.

The refusal was inherited from the Application profile, not derived for Machine code.
Application code checks an inline access against the span bound only (§8.2.3), so on sparse
backing it could touch an uncommitted page; its only recovery is the routed landing pads of
decision-5/decision-13, which a sparse VM does not install. The spec's second reason, "exact
retirement and step accounting without charging compiled suffixes that later defer", is
decision-29's contract for every Machine block (DEFER at an instruction start, nothing of it
committed), so it does not separate sparse from owned backing for Machine code.

### What compiled Machine code dereferences (5293f6e9)

Every host address a compiled Machine unit reads or writes inline comes from one of:

1. **The compiled view** (`jit_abi::machine::MachineTlb`): an inline load or store probes
   `entries[(linear >> 12) & 255]` and dereferences `addend + linear` only on a tag match for
   an operand inside the page (`codegen/machine.rs:537-640`). Two fills write entries,
   `fill` (PG = 0) and `fill_paged` (PG = 1, from a `Tlb` hit), both called only after a
   helper access completed inside that page (`jit_abi/machine.rs:502`), and both through
   `Memory::fast_frame` (`memory.rs:2382`): `Some` only when the whole 4 KiB bus frame lies
   in one RAM or ROM region inside the backing. Trap regions, holes, mixed frames and
   SoftMmu fill nothing.
2. **Live fields** (`LegacyOp::LiveField`, decision-30 §5): one load at host base + (bus −
   guest base) of bytes the interpreter fetched for this unit's block, inside its frame.
3. **Host metadata**, never guest backing: the code-tag word (`code_page`), the watch count
   and bitmap, the shadow return ring, `CpuState`, `MemCtx`. All are ordinary heap
   allocations (`try_zeroed_words`), committed by the allocator.

Everything else (holes, Trap pages, crossings, ROM writes, LOCK, strings) calls the checked
helpers or DEFERs to the interpreter, whose accesses check regions before dereferencing
(spec §4.1).

### What sparse backing guarantees (`memory.rs`, `memory/sparse.rs`)

- `Memory::map` commits the region's host pages, rounded out, **before** it publishes the
  region (`memory.rs:2746`). A failed commit publishes nothing.
- `Memory::unmap` removes the region, then decommits only whole host pages that no surviving
  region touches (`decommit_sparse`, `:2899`). A shared edge page stays committed.
- Both take `&mut Memory`, reached only through `&mut Vm` (`Vm::map`, `Vm::unmap`, or the
  public `vm.mem`). Every run holds `&Vm`, so no vCPU runs while a region is committed,
  decommitted or replaced. Child views borrow the VM (spec §4.1).
- A touched uncommitted page raises a host SIGSEGV/SIGBUS (Windows: an access violation).
  Machine VMs install no handler for it, so the process dies.

## Decision

### 1. The invariant

> **I.** Every host byte a compiled Machine unit dereferences inline lies in a region mapped
> in the VM's `Memory` at that moment, so on a host page committed by that region's `map`.

**Soundness argument.** By the list above, inline dereferences are view entries and live
fields.

- *View entries.* An entry is written only by `fast_frame`, whose frame lies in one RAM or ROM
  region; at fill time that region is mapped, so the frame's host pages are committed. The
  entry can outlive the mapping only across a `map`/`unmap`, which needs `&mut Vm`, so it
  happens between runs. Every compiling Machine run calls `State::begin_run`
  (`vm/legacy/compiled.rs:456`), which empties the view when `Memory::mapping_identity()`
  changed. `map` and `unmap` (and a replaced `vm.mem`) replace that identity, and the `Arc`
  held by the view's `State` keeps an old identity from being reused. So no run starts with
  an entry filled under another mapping.
- *Live fields.* A live field's bytes were fetched by the interpreter for the unit's block,
  so they lay in a mapped region at lift time. `Vm::unmap` drops every unit, proof and link
  overlapping the unmapped pages (`vm.rs:1718`, the J2 invalidation); the proof maps drop
  proofs on a changed mapping identity (`flat32.rs:133`, the paged map likewise), and a
  `Trap` `map` flushes the whole cache. So a unit that holds a live field runs only while its
  fetched bytes are mapped.
- *Commit.* Mapped implies committed: `map` commits before it publishes, and `unmap`
  decommits only pages that no surviving region touches.

A second, independent layer holds on every supported host: a fill follows a completed access
inside the frame, so at least one byte of the frame is mapped, and since the host page size
is a multiple of 4 KiB (4, 16 or 64 KiB), that byte's host page holds the whole frame. The
argument above does not need it.

The invariant is already true at 5293f6e9 for owned backing, where violating it gives a wrong
result instead of a host fault. Sparse backing turns a latent wrong result into a crash, so the
witnesses (§4) check the lifetime rule with a decommitted page.

### 2. The option: (b), commit-on-map plus the region-checked view

Admit Machine compilation on sparse backing with no change to generated code, the view, the
helpers or the memory layer. The admission becomes:

```text
admits(vm) = !vm.mem.host_fault_recovery()
             && vm.backend.compiles_machine()
             && !vm.backend.code_space_exhausted()
```

The Application paths keep every `is_sparse()` refusal (`vm.rs:4255`, `:4283`, `:4534`,
`:4736`, `test_support_force_region`).

**Costs (MEASURED, §6).** Host instructions per access: 0 added, the generated code is
byte-identical. Memory: 0 added; a 4 GiB sparse span already allocates its code-tag, watch and
dirty tables (3 × 128 KiB, lazily zeroed). Per run: 0 added, `begin_run` already compares the
identity.

### 3. Options considered

| Option | What it takes | Cost | Verdict |
|---|---|---|---|
| (a) A view or table of *committed host pages*, consulted at fill; a miss DEFERs | A commit bitmap kept by `Reservation::commit`/`decommit` (1 bit per host page: 32 KiB for 4 GiB at 16 KiB pages), read by `fast_frame` | One more test per fill (cold), a second source of truth beside the region map | Not needed: §1 shows the region map already implies commit, and the view already holds only region-checked frames. (b) *is* (a) with the region map as the table |
| **(b) Commit-on-map invariant + region-checked view** | The admission change, the witnesses, the spec amendment | None | **Recommended** |
| (c) Host-fault landing pads for Machine code | Routed config (`CONFIG_ROUTED`), a SIGSEGV/SIGBUS (and Windows) handler, per-site fault descriptors as decision-13 builds for Application, and an ordinal-exact DEFER from a faulting site | Every access then calls the helpers (routed Machine code inlines nothing), or new landing-pad metadata per access; a process-wide signal handler that spec §4.1 says sparse backing does not need | Rejected: it pays for faults that (b) shows cannot happen |
| (d) Keep refusing | Nothing | The embedder runs 2.6× slower, or switches to owned backing (a lazily zeroed 4 GiB allocation: no decommit on unmap, the whole span in the commit charge on Windows) | Rejected |

### 4. Witnesses and done-when

In `x86jit-tests/tests/legacy/machine_sparse.rs`, every one against the
interpreter on the same backing, through the J1 lockstep harness (state, flags, retired count,
TLB, written RAM and watched ranges after every call):

1. `compiled_programs_on_sparse_backing_equal_the_interpreter`: the J1 programs (§3 table of
   decision-29 and J2's additions: SMC, live fields, MMIO, ROM, splits, A20 off, watched
   pages, the entry refusals) on a 4 GiB sparse span, unpaged at CPL 0, paged at CPL 0 and
   CPL 3, in `Forced`, `Delayed(1)` and background T1 = 1, budgets 1, 7, 64. Compiled entries,
   paged entries, CPL-3 retirement and helper DEFERs are all nonzero.
2. `every_lowered_family_runs_compiled_on_sparse_backing`: every op family compiles (no
   unconditional DEFER but the final HLT).
3. `hole_accesses_on_sparse_backing_equal_the_interpreter`: a hot loop whose last iteration
   loads, stores or read-modify-writes 1, 2 or 4 bytes at an uncommitted hole, at a frame that
   shares a committed host page with a 4 KiB region (16 KiB-page hosts), across a region's end
   into a hole, across RAM's end, mid-span, from a hole into the top ROM page, past 4 GiB, and
   inside the lone region and the ROM (inline); unpaged at CPL 0 and 3, and paged at CPL 0 and
   3 through PTEs that point at those frames. Each gives the interpreter's exit or fault; the
   "device or hole" DEFER count is nonzero.
4. `a_region_unmapped_between_runs_is_never_touched_inline`: compiled code fills the view for
   a 4 KiB region alone in its host page; the region is unmapped (decommitted); the next run
   gives the interpreter's exit, and after a remap compiled code runs again on the zeroed
   region. Unpaged, paged and paged CPL 3, `Forced` and background. **Mutation check:** with
   `begin_run`'s flush removed, this test dies of SIGBUS on macOS arm64, so it witnesses the
   lifetime rule of §1.
5. `application_vms_on_sparse_backing_stay_interpreted` (Q3): an Application VM with a JIT
   backend, eager or tiered, compiles nothing on sparse backing and compiles on owned backing.
   `machine_j1_lockstep::sparse_backing_compiles` (formerly `sparse_backing_stays_interpreted`)
   enters compiled units in every mode; `memory::sparse_backing` (Application refusals) is
   unchanged.

Done-when: 1-5 pass on the three CI lanes; `machine32_loops` `-bg` cells on a 4 GiB sparse span
equal owned backing in host instructions per step within 1 % (§6); and the embedder's flat-game
probe on sparse backing is recorded (a census, not a gate, Q6).

### 5. Implementation

- `compiled::admits` drops its sparse test and states I (Q2 (a): no knob). The prototype's
  provisional `Vm::set_sparse_machine_compilation` opt-in is gone.
- Spec §4.1 states the invariant and keeps the Application refusal (Q3, Q4); decision-30's
  implementation note 10 points here.
- `Setup::sparse_compile` in the J1 lockstep harness: `Vm::with_backend_sparse` over
  `VmConfig::flat(1 << 32)`, the J1 regions plus a lone 4 KiB RAM region at 200000h and a ROM
  page at FFFFF000h.
- `MACHINE32_SPARSE=1` in `machine32_loops` runs the `-bg` paths on a 4 GiB sparse span.
- Three of the new tests (> 5 s) join `.config/slow-tests.txt` (Q5).

### 6. Measured

`machine32_loops`, 2,000,000 iterations, three rounds, release, I80386, macOS arm64, one
process per cell; host instructions per guest step (median of three; every round within 0.2 %):

| Path | Backing | register | memory | callret |
|---|---|---:|---:|---:|
| protected32-bg | owned, `flat(128 KiB)` | 19.80 | 35.40 | 65.61 |
| protected32-bg | sparse 4 GiB, compiled | 19.80 | 35.41 | 65.61 |
| protected32-bg | sparse 4 GiB, interpreted (before) | 424.3 | 565.9 | 1047.8 |
| paged32-bg | owned | 19.81 | 35.41 | 65.61 |
| paged32-bg | sparse 4 GiB, compiled | 19.80 | 35.41 | 65.61 |
| paged32-bg | sparse 4 GiB, interpreted | 528.9 | 740.4 | 1287.9 |
| paged32-cpl3-bg | owned | 19.80 | 35.21 | 65.41 |
| paged32-cpl3-bg | sparse 4 GiB, compiled | 19.80 | 35.23 | 65.41 |
| paged32-cpl3-bg | sparse 4 GiB, interpreted | 531.9 | 743.8 | 1291.3 |

The interpreted rows were measured with the prototype's opt-in off, the behaviour before this
record. Sparse compiled equals owned to the measurement's resolution. Re-measured on the
landed implementation (no opt-in, on 61ba04b4), the nine cells' medians differ from owned by
−0.01 % to +0.02 % (done-when, Q6). Wall-clock rates follow (for
example protected32 register about 1,070 M steps/s both ways, 48 M interpreted).

## Risks

- **A future inline access outside the view.** A later stage (J5 Segmented, a string or
  LOCK lowering, a region tier) that dereferences guest memory inline without the view
  breaks I on sparse backing with a host fault. Mitigation: the invariant is stated at
  `admits` and in spec §4.1 (Q4); a new inline access class must name how it satisfies I,
  and the hole and unmap witnesses run on sparse backing for every compiled mode.
- **A mapping change without `&mut Vm`.** An interior-mutable map or a device handler that
  remaps during a run would bypass `begin_run`'s flush. None exists; the `&mut` signatures of
  `Memory::map`/`unmap` are the guard, and the unmap witness fails (SIGBUS) if the flush is
  lost.
- **OS overcommit.** A committed anonymous page can still fail at first touch under memory
  pressure (spec §4.1). That holds for owned backing and the interpreter alike.

## Maintainer questions

| # | Question | Options | Recommendation |
|---|---|---|---|
| Q1 | Admit Machine compilation on sparse backing by the commit-on-map invariant and the region-checked view? | (a) a committed-page table; (b) invariant + view, no code change; (c) landing pads; (d) keep refusing | **(b)**. Zero cost, and §1 shows (a) holds already |
| Q2 | Gating after acceptance | (a) drop the knob, admit sparse unconditionally; (b) keep `set_sparse_machine_compilation`, default off; (c) keep it, default on | **(a)**. The generated code is identical, the API is pre-1.0, and a knob doubles the matrix the witnesses must cover |
| Q3 | Application VMs on sparse backing stay interpreted? | (a) yes, unchanged; (b) extend I to Application code | **(a)**. Application inline accesses check the span, not regions; extending it needs a region-checked probe or landing pads, a separate decision if a workload asks |
| Q4 | Amend spec §4.1: "Sparse backing executes Application code through the checked interpreter; compiled Machine blocks run, because they dereference guest memory inline only through frames proved inside a mapped region (decision-33 §1)"? | (a) yes, with the invariant; (b) leave the spec and rely on this record | **(a)** |
| Q5 | Witness placement | (a) the new module in the `legacy` binary, three tests in the slow tier; (b) all in the fast tier | **(a)**. The hole witness (3 s) stays fast |
| Q6 | Done-when | (a) `machine32_loops` sparse = owned within 1 %, the embedder's flat-game probe as a census; (b) the flat-game probe's 65 M steps/s as a gate | **(a)**. The library cannot fix the embedder's workload or host |

## Consequences

- PC-class boards on a 4 GiB sparse bus compile Flat32 code at J2/J3+J4 speed, keeping
  decommit-on-unmap and hole semantics.
- `admits` loses its sparse test (Q2 (a)); spec §4.1 names the invariant; the
  Application profile is unchanged.
- Every later compiled Machine stage inherits I as an admission condition for its inline
  accesses.
- Revisit if a stage needs inline guest accesses outside the view, or if a mapping change
  ever becomes possible during a run.
