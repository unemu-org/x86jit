---
id: decision-23
title: >-
  CPU model as the primary axis for 16- and 32-bit machine execution, with
  cycle-counted timing
date: '2026-11-11'
status: accepted
---

# CPU model as the primary axis for 16- and 32-bit machine execution, with cycle-counted timing

**Status: accepted (2026-11-11).** This record changes no code. It replaces the
Application-Real16 / LegacySystem split with one execution path selected by a CPU
model, and adds an optional cycle-counted timing level. It amends
[decision-18](decision-18%20-%20Legacy-386-protected-mode-CPU-state-and-execution-context-contract.md)
§§1, 3 and 4 and builds on decisions 16, 20, 21 and 22; what it keeps from each is
stated where it applies. Bus-level accuracy (prefetch queue, bus cycles, DMA and
refresh stealing, lockstep with devices) is out of scope and stays out: it is a
separate future project.

The maintainer answered the first draft's open questions on 2026-11-11. The
answers are folded into the body and listed under Resolved questions. The
maintainer accepted the record as a whole the same day.

## Context

### Two paths execute 16-bit guests today

READ at `5c95116`.

**Application Real16.** `CpuMode` has exactly `Long64`, `Compat32`, `Real16`
(`x86jit-core/src/lift/mod.rs:91-95`). Real16 is interpreter-only
(`vm.rs:3998-4003`, `vm.rs:4459-4466`) and uses the VM's shared translation cache,
keyed by physical fetch address and mode (`cache.rs:22-27`, `vm.rs:3986-3997`), with
multi-instruction blocks (`lift/mod.rs:877`). It has no descriptor state: a segment
base is `selector << 4`, computed in IR, and the segment limit is the constant
`FFFFh` (`lift/mod.rs:5133-5170`). Its behaviour is an implicit, fixed CPU
personality that nothing names:

- 80286 rules are hard-coded: the ten-byte instruction limit
  (`lift/mod.rs:392-397`, `:841`), #GP on a word access at `FFFFh` where the 8086
  wraps (`lift/mod.rs:5141-5161`), the 286 decoder permissiveness
  (`lift/mod.rs:303-314`), MOV CS as #UD "because we target the 286"
  (`lift/control.rs:809`).
- Yet the decoder accepts post-286 encodings, and 35 of those forms run with
  16-bit semantics: 66h segment pushes and pops, `PUSH imm32`, 66h near and far
  returns, `LES/LDS r32`, `ENTERD/LEAVED`, `LOOP` with 67h, `CALL/JMP r/m32`,
  and `POPF/IRET` dropping IOPL/NT
  (`x86jit-tests/tests/legacy_real_parity.rs:195-241`; tracked as xj-5t9).
- CPU exceptions are delivered in the guest through the IVT (`vm.rs:2729-2744`).

So Application Real16 is a 286 with a superset decoder. It matches no processor.

**LegacySystem.** Decision-18 added `ExecutionProfile::{Application(CpuMode),
LegacySystem}` (`legacy.rs:7-10`), `CpuGeneration::{I8086, I286, I386}`
(`legacy.rs:16-20`) and `LegacyCpuModel { generation, model_semantics_id }`
(`legacy.rs:25-28`). The path has descriptor caches, CR0, table registers and the
A20 line in `CpuState::legacy` (`legacy.rs:120-140`, `state.rs:449`), derives a
`LegacyDecodeContext` from them (`legacy.rs:144-232`), and runs in a separate
dispatcher (`vm/legacy.rs:425-604`) and a separate interpreter walk
(`interp/legacy.rs:814-867`, `:869`). It is selected by
`Vm::set_legacy_cpu_model` (`vm.rs:1199-1206`); `run`, `run_steps` and
`step_instruction` branch on the profile at entry (`vm.rs:2764`, `:2528`, `:2406`).
`I8086` is accepted by the context derivation but refused at run entry
(`vm/legacy.rs:310-316`). CPU exceptions are returned to the embedder, not
delivered (`legacy_real_parity.rs:498`).

The legacy path pays per instruction for generality it rarely needs. Every
instruction: re-derives the context (`vm/legacy.rs:469`); rebuilds and compares a
stamp that copies the 136-byte system state (`vm/legacy.rs:29-39`, `:530-533`);
fetches its bytes one at a time, each through a segment check, with a fresh decode
after every byte (`vm/legacy.rs:223-295`); looks up a one-instruction block in a
per-vCPU hash map (`vm/legacy.rs:186-195`; `lift/legacy.rs:227-235` sets
`icount: 1`); scans its ops for support (`interp/legacy.rs:827`); snapshots
registers, flags, selectors and the system state for rollback
(`interp/legacy.rs:836-842`); and zeroes the temp scratch
(`interp/legacy.rs:848`). The code says so: "Legacy performance has no application
target" (`interp/legacy.rs:835`). The tree holds no measurement of this path
against Application Real16 on the same guest code.

### The two paths drifted

xj-srh.19 has landed as commit `5c95116`. It measured the gap when an embedder
moved its AT-class board from Application Real16 to LegacySystem I286: 152 refused forms on I286, 287
on I386, and wrong REPNE string results. It closed the gap by making legacy real
mode call the Application lowering (`lift/legacy.rs:800-844`), and pinned equality
with `legacy_real_parity.rs`, whose allowlist names the 35 forms above. For those
forms the *legacy* path follows the manuals and the Application path does not.

The duplication that remains is structural: two lowerings of stack, control-flow
and flag-image instructions (`lift/mod.rs:4183-4553` against
`lift/legacy.rs:438-795`), two interpreter walks, two dispatch loops, two interrupt
delivery helpers (`interp::deliver_interrupt`, `interp/legacy.rs:576`), two segment
models and two exception policies. Each new model would multiply it.

Work continues on the legacy path while this record is open. xj-srh.20, in
progress, adds real-mode single-step (TF) through the IVT, with per-model rules
for the instruction after a segment load; at `5c95116` a set TF is still refused
when the context is derived (`legacy.rs:209-211`). Each such bead adds
model-dependent behaviour to a path this record replaces, which is why the bead
plan fixes where each one lands.

### "Legacy" is the wrong axis

Both paths already choose behaviour by processor: the Application path by a
hard-coded 286, the legacy path by `generation` tests spread over the lifter and
interpreter (`lift/legacy.rs:113-177`, `legacy.rs:735-748`, `:801-811`,
`interp/legacy.rs:290`, `:1403`). The xj-5t9 forms are not defects with one right
answer: on a 286 they are #UD (decision-21), on a 386 they have 32-bit semantics.
The answer is a property of the model. Decision-18 §1 anticipated this: it made
xj-50w the owner of "one shared model-profile authority" and `CpuGeneration` only
a projection of it. This record defines that authority.

### What the consumers need

- A 64-bit userland embedder uses Application Long64 and Compat32 only. It must
  see no change: no source change, no change in emitted code.
- A PC-class embedder runs real-mode guests on Application Real16 and is moving an
  AT-class board to LegacySystem I286. Its order of need is: 80286 now; 80386DX
  next (386 real-mode set, big real mode, no CPUID); later 486- and Pentium-class
  guests running a 32-bit protected-mode operating system with paging (CPUID, TSC,
  MMX); and 8088/V20 for an XT-class board. It selects the model and the FPU per VM
  at construction and does not want to know about Application versus LegacySystem.
  Its virtual clock is the retired-instruction count, which must keep its exact
  meaning; cycles may be offered beside it but must stay optional and must cost
  nothing when unused. Results and step counts must be identical for identical
  inputs, independent of host and of the interpreter/JIT choice. It accepts the
  manual-correct behaviour for the 35 xj-5t9 forms.
- The maintainer plans 8086, 8088 and NEC V20 models (epic xj-50w), after the 386
  system features and the compiled route that the PC-class embedder needs.

The measured interpreter cost on the Application Real16 path is 385.0 host
instructions per guest step on `alu` and 472.8 on `dos-mixed`
(`x86jit-bench/examples/real16_loops.rs`; commit `4d88291`, xj-193.3, which reduced
them from 416.9 and 530.5). Those gains must survive.

### Primary sources

| Authority | Fact used here |
|---|---|
| SDM Vol 3B §23.1.3 | The 8086 core instruction list, and the instructions later processors added that execute in real-address mode. |
| SDM Vol 3B §25.13.1 | Instructions added in the Intel386 and Intel486 processors and the Intel 387 coprocessor. |
| SDM Vol 3B §25.15 | Reserved opcodes with defined behaviour (82h, group 2 `/6`, group 3 `/1`, D6h SALC). |
| SDM Vol 3B §§25.16, 25.16.1; §25.13.1 and Table 25-1 | AC (bit 18) is not implemented on the Intel386 and is always clear; ID (bit 21) arrived later. BSWAP, XADD, CMPXCHG, INVD, WBINVD and INVLPG arrived with the Intel486; CPUID, RDTSC, RDMSR and WRMSR later still. |
| SDM Vol 3B §25.17.1 | PUSH SP: the 8086 pushes the decremented SP; the Intel 286 and later push the value before the decrement. |
| SDM Vol 3B §25.17.2; Vol 1 §21.1.2 | FLAGS bits 15:12 as stored: 8086 all ones; 286 real mode all zeros; 32-bit processors bit 15 zero, bits 14:12 last loaded. |
| SDM Vol 3B §25.18 | 8087, 287 and 387 differences. |
| 80286 PRM Appendix B, "Clocks", p. B-6 | What a 286 clock count includes, its addressing and odd-address additions, its labels (`mem`, `noj`, `pm`), and its stated 5-10 % shortfall. |
| 80286 PRM Appendix C, software items 1-17, pp. C-1-C-3; hardware items 1-2, p. C-3 | The 8086/8088 differences as Intel lists them for the 286. |
| 80386 PRM §14.7 items 1-18, pp. 14-5-14-7; Table 14-1 | The 8086/8088 differences as Intel lists them for the 386; real-mode exceptions. |
| 80386 PRM §14.8 | 80286 real-mode differences. |
| 80386 PRM §17.2.2.3, pp. 17-10-17-11 | What a 386 clock count assumes; the `n`, `m` and `pm=` symbols. |

The 80286 PRM is `oracles/intel/i286-prm.pdf`, order 210498-005; the 80386 PRM is
`oracles/intel/i386-prm.pdf`, order 230985-001; the SDM is 325462-092US. Both PRMs
are scans. Page images were checked for 80286 PRM pp. B-6 and C-2 and for 80386 PRM
pp. 14-6 and 17-10. The other PRM pages cited in this record were read from the
OCR text layer only; a bead that encodes a value from them checks the page image
first. No fact about the 8086, 8088, 80186 or V20/V30 that is absent from these
pinned sources is decided here; such facts are listed as *to derive*, with the
manual that must be pinned first.

## Decision

### 1. Two profiles: Application and Machine

**Application** means userland execution in which the embedder is the system. The
engine models no descriptor tables, control registers, privilege or interrupt
delivery; segments are flat except configured FS/GS bases; exceptions and system
calls leave through `Exit`; blocks compile. Application is exactly `Long64` and
`Compat32`, driven by `GuestCpuFeatures` and the CPUID tables (decisions 9 and 11).
Nothing in this record changes its lifting, IR, cache keys, code generation or API.

**Machine** means the engine executes a processor: segmentation, real and
protected operation, interrupt and exception delivery, port I/O, and the model's
system registers. Every 16-bit guest and every guest that owns its own protected
mode runs here. `ExecutionProfile::LegacySystem` is renamed `Machine`; the old name
stays as a deprecated alias through the migration.

Real16 was never an application context: it has an IVT, IF, IRQ injection and port
I/O. It moves to Machine. `CpuMode::Real16` stays in the enum with its discriminant,
because `BlockKey` hashes it (`cache.rs:29-36`) and `IrBlock::mode` carries it
(`ir.rs:3575`), and `Vm::set_cpu_mode(CpuMode::Real16)` stays as a compatibility
alias defined in §6.

### 2. The CPU model selects everything a processor defines

Add `CpuModel`, `#[non_exhaustive]`, chosen once per VM before any vCPU exists:

```rust
vm.set_cpu_model(CpuModel::I80286);   // selects Machine; immutable afterwards
vm.set_fpu_model(FpuModel::None);     // VM default, inherited by its vCPUs
```

`set_cpu_model` is the one entry point. An embedder never names a profile; real
and protected switching is architectural inside the model, from CR0/MSW and the
descriptor caches as decision-18 §1 fixed; A20 stays per vCPU (decision-22).

Each model is one entry in a static catalogue, `ModelProfile`. The catalogue is
the "shared model-profile authority" of decision-18 §1 and the content of epic
xj-50w. An entry defines:

| Field | Meaning |
|---|---|
| `isa` | The decode set: which encodings exist, and what an absent one does (#UD, a defined alias, or a typed refusal — see §3). Replaces the decoder-feature filters at `lift/legacy.rs:113-170`. |
| `quirks` | The semantic differences of §3, as named fields with one cited value each. |
| `system` | `None`, `Msw286` (16-bit protected mode, 24-bit bases, LMSW-only entry; decision-21) or `Cr386` (32-bit protected mode, paging, big real mode; decisions 18 and 20). |
| `physical_bits` | 20, 24 or 32: the width of a formed physical address. |
| `max_insn_len` | None, 10 or 15. |
| `fpu` | Which `FpuModel` values the model accepts (§4). |
| `timing` | The cycle table of the Timing model section, or none. |
| `semantics_id` | A stable identity of the entry's lifting-relevant content; it replaces the embedder-supplied `model_semantics_id`. |

`CpuGeneration` remains as a projection of `isa` for the existing checks until
they read the catalogue directly.

| Model | `isa` | `system` | Phys. | FPU pairing | Timing source | Wave |
|---|---|---|---|---|---|---|
| `I80286` | 286 real + protected set | `Msw286` | 24 | none, 287 | 80286 PRM Appendix B | 1 (exists as LegacySystem I286) |
| `I80386` (DX) | 386 set | `Cr386` | 32 | none, 287, 387 | 80386 PRM ch. 17 | 1 (exists in part as LegacySystem I386) |
| `I8088` | 8086 core set | `None` | 20 | none, 8087 | 8086 family manual (to pin) | 3 |
| `NecV20` | 8086 core + 186 additions + NEC extensions | `None` | 20 | none, 8087-class | NEC V20/V30 manual (to pin) | 3 |
| `I8086` | as `I8088` | `None` | 20 | none, 8087 | as `I8088`, 16-bit-bus counts | 3, with `I8088` |
| `NecV30` | as `NecV20` | `None` | 20 | as `NecV20` | as `NecV20`, 16-bit-bus counts | 3, with `NecV20` |
| `I80186`, `I80188` | 8086 core + 186 additions | `None` | 20 | none, 8087 | 186/188 manual (to pin) | 4, no consumer yet |
| 386SX | as `I80386` | `Cr386` | 24 | none, 387SX | 386SX reference (to pin) | not a separate entry until asked: `I80386` semantics with `physical_bits = 24` and its own table |
| `I80486` class | named only | — | 32 | integrated or none | — | 2, under its own later decision |
| Pentium class | named only | — | 32 | integrated | — | 2, under its own later decision |

**Order of work**, as the maintainer set it:

1. The unified path with `I80286` and `I80386`. The AT-class board needs the
   first now and the 386DX board needs the second next.
2. The 386 system features (epic xj-srh) and the compiled route of §8, which the
   PC-class embedder's later guests depend on; and the 486- and Pentium-class
   entries.
3. `I8088` and `NecV20`, with `I8086` and `NecV30` if they prove to be table-only
   variants.
4. `I80186`/`I80188`, when a consumer appears.

**486- and Pentium-class models are in scope for the library but not for this
record.** They are named in the catalogue so that nothing here forecloses them,
and they get their own later decision. That is where the model axis meets
`GuestCpuFeatures`, the CPUID tables (decisions 9 and 11) and `CpuSource`
(decision-12): a CPUID-era entry refers to those facilities rather than
duplicating them. Nothing about them is designed here.

`I8086` and `NecV30` are listed with their 8-bit-bus siblings on the expectation
that they differ only in bus width, which at this record's timing levels is a
different cycle table and nothing else. That expectation is *to derive* from the
manuals once pinned; if it fails they become separate work.

### 3. Model quirks: one named field per difference

A quirk is resolved **at lift time**. The lifter reads the catalogue entry and
emits IR that already contains the model's answer, as a constant or as a choice of
lowering. The interpreter's per-op code contains no model test. Adding a model
therefore adds no work to the hot loop of any other model. This is the central
rule of spec.md §16 — encode a semantic trap once, centrally — applied to models.

Licensed by sources that are pinned today:

| Quirk | 8086/8088 | 80286 | 80386 | Source |
|---|---|---|---|---|
| Value stored by `PUSH SP` | SP after the decrement | SP before | SP before | SDM Vol 3B §25.17.1; 80286 PRM App. C item 8, p. C-2; 80386 PRM §14.7 item 4, p. 14-6 |
| FLAGS bits 15:12 stored by PUSHF, interrupts, exceptions | all ones | zeros in real mode | bit 15 zero, 14:12 last loaded | SDM Vol 3B §25.17.2; Vol 1 §21.1.2 (already in `legacy.rs:801-811`) |
| Shift and rotate count | not masked | masked to 5 bits | masked to 5 bits | 80286 PRM App. C item 9, p. C-2; 80386 PRM §14.7 item 5, p. 14-6 |
| Saved CS:IP of a divide exception | next instruction | the DIV, with prefixes | the DIV | 80286 PRM App. C item 3, p. C-1; 80386 PRM §14.7 item 2, p. 14-5 |
| `IDIV` quotient 80h / 8000h | interrupt 0 | produced | produced | 80286 PRM App. C item 13, p. C-3; 80386 PRM §14.7 item 11, p. 14-7 |
| Instruction length limit | none | 10 bytes | 15 bytes | 80286 PRM App. C item 10, p. C-2; 80386 PRM §14.7 item 6, p. 14-6 |
| Operand crossing offset `FFFFh` | offset wraps | interrupt 13 | 12 on SS, else 13 | 80386 PRM §14.7 item 7, p. 14-6; 80286 PRM Table C-1, p. C-1 |
| Sequential execution past `FFFFh` | IP wraps to 0 | *to derive* | exception 13 | 80386 PRM §14.7 item 8, p. 14-6 |
| Address above 1 MiB | truncated to 20 bits | formed in 24 bits | formed in 32 bits | 80386 PRM §14.7 item 18, p. 14-7; 80286 PRM App. C hardware item 2, p. C-3 |
| Processor-raised interrupts beyond 0-4 | none of those listed for the later processors | 5, 6, 7, 8, 9, 13, 16 added | 5, 6, 7, 8, 12, 13 listed as new against the 8086 | 80286 PRM App. C item 1 and Table C-1, p. C-1; 80386 PRM §14.7 item 17 and Table 14-2 |
| `POP CS`, `MOV CS, r/m` and other encodings the 8086 leaves undefined | undefined operations; no invalid-opcode interrupt exists | interrupt 6, or a 286 instruction | exception 6, or a 386 instruction | 80286 PRM App. C item 6, p. C-2; 80386 PRM §14.7 item 3, p. 14-5 |
| `LOCK` | differs; not specified by the pinned sources | meant for XCHG, MOV, MOVS, INS, OUTS; asserted on every memory XCHG | #UD outside the listed forms | 80286 PRM App. C item 11, p. C-2; 80386 PRM §14.7 item 9, p. 14-6 |
| x87 operand past the segment end | — | interrupt 9 | 12 or 13 | 80286 PRM Table C-1, p. C-1; 80386 PRM Table 14-1 |
| Numeric error | external interrupt through the controller | interrupt 16 | interrupt 16 | 80286 PRM App. C items 4, 15; 80386 PRM §14.7 items 14, 16 |
| Single-step against a pending external interrupt | the interrupt handler can be single-stepped | single step has the higher priority; the handler is not stepped | as the 286 | 80286 PRM App. C item 12, p. C-2; 80386 PRM §14.7 item 10, p. 14-6 |
| EFLAGS AC (bit 18) and ID (bit 21) | — | — | not implemented; always clear | SDM Vol 3B §§25.16, 25.16.1 |

The operand-crossing and x87-operand rows retire the "deliberate divergence"
recorded in spec.md §17.9 and `state.rs:463-469` (x87 SS-relative operands faulting
with 13 whatever the processor): each model gets the vector its manual gives.

**Identity is catalogue data too.** What makes a 386 a 386 to software that probes
it — AC and ID fixed at zero, and #UD for CPUID, RDTSC, RDMSR/WRMSR, INVD, WBINVD,
INVLPG, BSWAP, CMPXCHG and XADD (SDM Vol 3B §25.13.1, Table 25-1, §§25.15-25.16.1)
— is part of the `isa` and `quirks` fields of `I80386`, with the corresponding
gating on `I80286` where the 286 lacks a form. This is the content of xj-srh.17,
which becomes catalogue attributes instead of checks in the lifter. A 386DX board
whose firmware expects no CPUID depends on it.

The single-step row and the rules for the instruction after a segment load are the
per-model part of xj-srh.20; they become quirk fields when the catalogue exists.

*To derive*, each blocked on the manual named in Provenance. No value is decided
here:

- 80186/80188: which of the rows above follow the 8086 and which the 286; the
  exact set of added instructions; its undefined-opcode trap.
- 8086/8088: the saved state when an interrupt arrives inside a repeated string
  instruction that carries more than one prefix; the flags left undefined by each
  instruction; `AAM`/`AAD` with an immediate other than 0Ah, if the manual
  documents the second byte at all.
- NEC V20/V30: the extension instructions behind the 0Fh escape and at 64h-67h;
  the `AAM`/`AAD` immediate; the mode (MD) flag and how it reads and writes in
  the flag image; the entry and exit instructions of 8080 emulation mode and
  which instruction stream each belongs to; the register mapping between the two
  modes.

**Undocumented behaviour needs its own rule.** On the 8086 every opcode executes
something, and no Intel manual says what most of the undefined ones do. A manual
cannot license a behaviour it does not describe, and a hardware-captured corpus can
falsify a claim but cannot license one (PROVENANCE.md §2). Decided: an encoding
that the model's manual does not define ends in `Exit::UnknownInstruction` — a
typed refusal, not a guessed semantic and not a fabricated #UD on a processor
that has none. PROVENANCE.md is not changed, and no "observed, not licensed"
category is introduced. If real software for these machines needs one of these
encodings, that is a new question for the maintainer with the software's evidence.

**Amendment (compatibility aliases, 2026-11-14, accepted by the maintainer).** SDM
Vol 3B §25.15 (p. 25-5) lists reserved opcodes that, "for compatibility with prior
generations", behave as defined instructions: 82h as 80h, F6h/F7h /1 as TEST /0,
shift group 2 /6 as /4. Such an alias is a defined alias of a model when both hold:
SDM §25.15 lists it, and the hardware-captured corpus of that processor shows it as
an alias. It is not one when the model's own manual or its corpus says otherwise.
On that basis, F6h/F7h /1 is TEST and 82h is 80h on `I8088`, `I8086`, `NecV20`,
`NecV30` and `I80286`. The 80286 PRM, Appendix B p. B-12, lists F6h /1 as an invalid
extension, but the 80286 corpus shows it executing as TEST, so the 286 entry keeps
TEST. Shift group 2 /6 is not an alias on the 8088 or the V20: their corpora mark it
undocumented, not as the SHL alias. It stays a typed refusal there, as the rule
above says.

The same holds for an encoding the SDM's instruction pages define as an ordinary
instruction, valid in real-address mode, which the model's manual omits but the
model's hardware-captured corpus shows executing as that instruction. Case: 83h /1,
/4, /6 (OR, AND, XOR r/m16 with a sign-extended imm8). The 8086 manual's tables do
not list them. The SDM defines them (Vol 2A §3.3 AND; Vol 2B §4.3 OR; Vol 2D §6.1 XOR), the
8088 corpus marks them normal, and assemblers emit them in 16-bit code. They execute
on `I8088`, `I8086`, `NecV20` and `NecV30`. Other encodings stay typed refusals on
the 8088 even though its corpus shows them executing:
- 60h-6Fh, C0h/C1h and C8h/C9h, where the 8088 does something other than the
  instruction the SDM defines for that encoding;
- FFh /7 and D6h, which the SDM reserves;
- shift /6.

**8080 emulation mode on the V20/V30 is split between two cores.** Executing 8080
code is not x86jit's job: it is a second instruction set, and it lives in a
separate core outside this repository: the 8080-family project x80 (working
name; a sibling repository, 8080 first, then 8085 and Z80). That keeps
`x86jit-core`'s dependency set at `{iced-x86}` (`x86jit-tests/tests/boundary.rs`).
What belongs to the V20/V30 model in x86jit is the native side of the mode switch:

- The model decodes BRKEM, RETEM and CALLN and maintains the MD flag.
- On entry to 8080 mode x86jit returns a typed `Exit`. The embedder runs the 8080
  core on the same memory and the same registers, and resumes x86jit on RETEM or
  on an interrupt that forces native mode.
- The exit documents the register mapping the two cores share: A = AL, BC = CX,
  DE = DX, HL = BX, the 8080 stack pointer = BP, memory through DS. This mapping
  is the maintainer's statement of the design; it is a hardware fact about the
  V20/V30 and is verified against the NEC manual once that is pinned, before any
  code relies on it.
- The exact `Exit` contract is designed with the V20 bead (C4): the fields of the
  exit, interrupts that arrive in 8080 mode, and how steps and cycles are
  accounted across the two cores so that the embedder's clock stays one clock.
- Until that bead lands, BRKEM ends in a typed refusal.

### 4. The FPU is a separate axis

`FpuModel` stays a separate, `#[non_exhaustive]` choice (`state.rs:470-483`). It is
selected per VM with `Vm::set_fpu_model` and inherited by vCPUs; the per-vCPU
`Vcpu::set_fpu_model` (`vm.rs:2277`) remains as an override and is not deprecated.
That is the decided shape: a VM default fixed at construction, which is what the
PC-class embedder asked for, without removing the per-vCPU control that vCPUs
sharing memory and cached IR were given it for (`state.rs:459-461`). The catalogue only
constrains the pairing: selecting a coprocessor the model cannot attach is a
configuration error at `new_vcpu`.

- `None` — an empty socket, as today.
- `Present` — the existing x87 engine with no generation claim, as today. It stays
  the default so that no existing configuration changes.
- `I8087`, `I80287`, `I80387` — generation-faithful coprocessors, added as
  variants when their beads land. Their differences come from SDM Vol 3B §25.18,
  the 80286 PRM Part II (80287) and, once pinned, the 80387 and 8087 references.

On `Msw286` and `Cr386` models MSW/CR0 own EM, MP and TS; `Vcpu::set_fpu_emulation`
(`vm.rs:2288`) remains only for models with `system = None`, where no such register
exists, and for the compatibility alias.

### 5. One execution path, with two segment-access lowerings

There is one lifter front end, one IR, one interpreter walk, one dispatcher loop
and one interrupt-delivery helper for every Machine context. They take a
`MachineContext` (the present `LegacyDecodeContext`, renamed, with `generation`
replaced by the catalogue entry) and nothing else selects behaviour.

**Lowering.** The width-aware legacy lowering of stack, control-flow, flag-image,
string, frame and segment-load instructions (`lift/legacy.rs:438-795`) becomes the
only implementation for 16- and 32-bit Machine contexts. It already takes the
operand size from the instruction and the stack size from the context, which is
why it gets the xj-5t9 forms right. The Application `*_real16` helpers
(`lift/mod.rs:4183-4553`, `lift/control.rs`) are retired. Ordinary scalar
instructions keep the shared lowering they already use on both paths
(`lift/legacy.rs:800-818`).

**Segment access.** `with_segment` is already the single place where a memory
operand meets its segment, and it already has two forms (`lift/mod.rs:5120`,
`:5133`). Both forms stay, and the context chooses between them at lift time:

- **Direct**: the context proves that every segment is an ordinary real-mode
  segment — base `selector << 4`, limit `FFFFh` — so the lifter emits today's
  Application Real16 lowering: the base computed inline, the model's limit rule as
  a constant (wrap for `physical_bits = 20` models, the model's fault vector for
  the others), no cache read.
- **Checked**: anything else — protected mode, big real mode, a retained
  non-standard cache, a reset-origin code segment. The lifter emits the
  cache-backed access of decision-18 (`LegacyOp::Address`, checked by
  `legacy::access`, `legacy.rs:699-770`).

The class is part of `MachineContext`. It can change only where the context
already changes and already ends the block (decision-18 §3): a CR0/MSW write, a
segment load that installs a non-standard cache, an embedder state edit, a reset.
A real-mode segment load replaces the base and keeps the limit (decision-20), so it
keeps a Direct context Direct.

**What a model without system features pays.** For `system = None` the context is
Direct by construction and never changes. None of the per-instruction work listed
in Context exists for it: no context derivation, no stamp, no byte-wise refetch,
no private hash map, no rollback snapshot, no cache read, no limit load. Its
"cached limit" is the constant `FFFFh` in the lowering, and its 20-bit wrap is a
constant mask. `CpuState::legacy` stays inert for these models, exactly as it is
for Application today.

For `Msw286` and `Cr386` models the class is derived at run entry and after each
context-changing instruction: six cache comparisons, not per instruction. A
segment load in a Direct context also writes the cache, so the hidden state is
correct the moment PE is set.

**A20 in the Direct class.** AT-class firmware leaves A20 disabled, so real mode
with A20 disabled must be Direct, not Checked. In a Direct context the physical
address is the real-mode linear address under a mask: constant for 20-bit models,
the per-vCPU A20 mask otherwise. Blocks do not bake the line state, so toggling it
needs no relift. A transfer that crosses the wrap boundary leaves the fast path
and uses the split transfer of decision-22. The boundary rule of decision-22 —
when a queued change applies — is unchanged.

**Dispatch and caching.** Direct contexts use the VM's shared translation cache
with multi-instruction blocks and the existing `BlockKey`, as Application Real16
does today; the model is VM-wide, so the key needs no model field. Checked contexts
keep decision-20's per-vCPU retained-context namespace and one-instruction blocks.
Making Checked contexts faster is separate, measured work, not a condition of this
record.

**Interpreter.** The generic walk executes the Machine ops. The legacy walk and
its dispatcher loop are removed once parity is proven. The per-instruction rollback
snapshot (`interp/legacy.rs:836-857`) is taken only for instructions the lifter
marks as not restartable by lowering order alone; Direct-class scalar instructions
never take it.

**Exception delivery.** The two paths differ today (in guest against outward).
The unified path has one explicit per-VM option: `InGuest` delivers through the
real-mode table, and through the protected IDT once xj-srh.6 lands; `Outward`
returns the fault as `Exit::Exception`. `InGuest` is the default for
`set_cpu_model` and is what the alias does: it is what the processor does.
`Outward` survives as an explicit option in real and in protected mode, including
after xj-srh.6, for an embedder that wants to see every fault. The deprecated
`set_legacy_cpu_model` selects `Outward`, so its callers keep today's behaviour.

The mechanism in "Segment access", "A20" and "Dispatch and caching" is the target
structure. The first unification bead measures it before the rest build on it; if
the measurement contradicts it, this section is amended, not worked around. What
is fixed regardless is: one lowering and one interpretation of every instruction,
model facts resolved at lift time, and the performance requirement below.

### 6. `CpuMode::Real16` becomes an alias

`Vm::set_cpu_mode(CpuMode::Real16)` keeps working and keeps its meaning for
existing callers. It selects the Machine path with an **internal** catalogue entry
— not a public `CpuModel` — that reproduces today's personality: 80286 real-mode
quirks, the superset decoder, selector-derived segments (an embedder may keep
writing `cpu.ds` directly between runs; the engine derives the caches at run
entry), `InGuest` delivery, `FpuModel::Present`, A20 not consulted. The entry is
frozen: it gains no features — single-step from xj-srh.20 included — and is the
only place the hybrid survives.

The alias stays until the PC-class embedder has moved fully to models. Its removal
is decided then, on that evidence, and is not tied to a migration step.

One deliberate change reaches the alias: when the lowering is unified, its 35
xj-5t9 forms take the manual-correct operand-size behaviour, because the code that
got them wrong no longer exists. The PC-class embedder has accepted this.

**Amendment (U3, 2026-11-12, accepted by the maintainer).** U3 serves the alias
with the same width-aware lowering of stack, control-flow, flag-image, frame and
far-pointer instructions as every Machine context. Besides the 35 xj-5t9 forms,
the alias therefore follows the manuals in these corner cases of those families:

- the stack-segment limit check and the SP wrap at 0000h and FFFFh;
- POP addresses the stack through SP, not ESP, when the high bits of ESP are set
  and the stack is 16-bit;
- a far CALL and an IRET check their whole frame before the first access and
  commit nothing until every access has completed;
- POPA does not read the stack slot that holds the discarded SP;
- XLAT honours the 67h address-size prefix and an FS or GS segment override;
- LDS, LES and BOUND with a 67h memory operand check its whole offset against the
  segment limit.

Instructions outside those families keep the frozen alias behaviour. A memory
operand of such an instruction with a 67h prefix passes the limit gate on the low
16 bits of its offset and is then accessed at the full linear address, as before
U3. One named policy implements this (`alias_low16` of `SegLimitCheck`).

`run_steps`, `irq_ready`, `inject_irq`, `retired_instructions`, `StepOptions`,
`StepRun` and the decision-16 contract are unchanged for the alias and are the
contract of every Machine model.

### 7. What stays from the earlier decisions

| Decision | Kept | Amended |
|---|---|---|
| 16 | The whole step, readiness and shadow contract. | Its scope: "Real16 interpreter" becomes "every Machine context". |
| 18 §1 | Context derived from CR0 and hidden caches; no writable duplicates; `Code16` never implies real addressing. | `LegacySystem` → `Machine`; `LegacyCpuModel` → `CpuModel`; "Application Real16 keeps its current interpreter path" is withdrawn. |
| 18 §2 | `CpuState::legacy` layout, every frozen offset, the field name. | Nothing. |
| 18 §3 | Guard dimensions; block-ending instructions; physical-bus access chain. | Direct contexts use the shared cache; the interpreter-only rule gets the staged exception of §8. |
| 18 §4 | One event path; delivery chosen at acceptance; typed unsupported exits. | The Application Real16 row merges into the Machine row. |
| 18, scope | V86 mode, debug registers and the 387 error integration stay out. | Real-mode single-step and hardware task switches are no longer out of scope: xj-srh.20 and xj-srh.18 narrow that exclusion, each in its own bead. |
| 20 | Retained contexts, big real reload rule. | Applies to Checked contexts only. |
| 21 | Every 286 fact and witness. | None; `I80286` is its model. |
| 22 | The A20 API, boundary rule, fault evidence. | The line is also consulted in Direct contexts of 24- and 32-bit models. |

### 8. A route to compiled execution for 386-and-later protected mode

Decision-18 keeps system execution interpreted "with every backend". That is right
for the 286 and tolerable for a 386 running firmware and a real-mode operating
system. It is not tolerable for the 486- and Pentium-class guests on the PC-class
embedder's path: an interpreter that spends several hundred host instructions per
guest instruction (385-473 measured on `real16_loops`) cannot sustain a 32-bit
protected-mode operating system with paging. The first step below replaces that
estimate with a measurement.

The maintainer approved the staged route below as direction. This record still
does **not** authorise any Machine context to enter a backend: each stage needs
its own decision record, with evidence, before implementation. Those decisions
start from the mechanisms named here:

- a software TLB whose write entries exclude code-tagged physical pages;
- one guest page per compiled block, with cross-page transfers through the
  dispatcher;
- a Flat32 class whose segment assumption is carried in the block key;
- CPL and the IOPL comparison in the block key, with privilege transfers through
  helpers;
- an entry-time step-budget check, with the interpreter finishing a block the
  budget cannot cover.

The 486- and Pentium-class models of §2 are what this route is for; their own
decision builds on it.

| Stage | Content | Depends on |
|---|---|---|
| J0 | Decision record and measurement: instruction rate of the interpreter on a flat 32-bit protected-mode workload, against the Compat32 JIT on the same code. Defines the compiled Machine block key. | unified path (U2), xj-srh.6 |
| J1 | Instruction-exact accounting in compiled Machine blocks: a step budget decremented by the block's instruction count at entry, with the interpreter finishing a block the budget cannot cover; the retired count; exact counts at a mid-block fault from the faulting instruction's ordinal. | J0 |
| J2 | First compiled class, **Flat32**: PE set, CS.D and SS.B set, CS/DS/ES/SS with base 0 and a 4 GiB limit, paging off, CPL 0. Segment checks vanish because the key carries the assumption; code generation is today's Compat32 plus system instructions through helpers. | J1 |
| J3 | Paging in compiled code: a per-vCPU software TLB with an inline hit path and a page-walk helper. | J2, xj-srh.7 |
| J4 | Privilege: CPL in the key, user/supervisor TLB split, privilege transfers through helpers. | J3, xj-srh.9, xj-srh.10 |
| J5 | Non-flat segments in compiled code (16-bit protected code, FS-style bases with limits). | J4 |
| — | A compiled tier for Direct (real-mode) contexts remains deferred (deferred.md, "Real-mode JIT tier"). The 286 and the XT-class models stay interpreted. | — |

Constraints on the unification work, so that the stages are possible:

- **Machine IR is ordinary IR.** A Machine op is defined by its effect on
  `CpuState` and memory, executable by a helper call from compiled code. No op may
  depend on interpreter-private state. The guards that reject a block with a
  Machine context before backend entry (`vm.rs:187`, `:206`, `:229`, `:250`,
  `:495`, `:4447`; `x86jit-cranelift/src/lib.rs:4223`, `:4296`) stay until J2.
- **Compiled identity is physical and context-flagged, not per-vCPU.** The J0 key
  is `{ physical pc, decode IP, context flags, mxcsr mode }` in its own namespace
  of the shared cache; the Application `BlockKey` and its hash stream do not
  change. Decision-18 §3 already requires both the decode IP and the fetch
  address in identity. Context flags carry what the code assumes: CS.D, SS.B, the
  segment class, CPL, and whether IOPL permits the I/O-sensitive instructions.
- **A compiled block covers one guest page.** A block's bytes then depend on one
  physical page and one mapping, so a changed mapping or a write to the page
  invalidates exactly the blocks it should. An instruction that straddles two
  pages is a block of its own with both pages checked.
- **Paging is behind one translation interface from its first day.** xj-srh.7
  implements `translate(linear, access, cpl) → physical` with an explicit
  invalidation entry point (CR3 write, PE/PG change), so that the same interface
  later backs the software TLB.
- **Steps are exact in every tier.** The determinism requirement — identical
  results and step counts whatever executes the code — binds every Machine
  context. It does not bind Application, where compiled blocks do not tick the
  retired counter (`vm.rs:1818-1827`) and no step API exists.

Why a software TLB and not host page tables: guest pages are 4 KiB and the
primary development host uses 16 KiB pages, so a guest mapping cannot be
expressed as a host mapping there. A software TLB also carries the answer to
self-modifying code: `Memory` is the physical bus in Machine mode, so the existing
code-page tags (`memory.rs:560`) are already tags on physical pages; a write TLB
entry for a tagged page is simply never installed on the fast path, and the store
reaches the existing invalidation through the slow path.

The cost is stated plainly: every compiled memory access in a paged Machine
context pays a TLB probe that Application code does not. A Machine guest will be
slower than the same code under Application Compat32. How much is a J0/J3
measurement, not a claim made here.

## Timing model

Three levels. Level 1 exists and is the default. Level 2 is added by this record
and is optional. Level 3 is not built here.

| Level | Unit | Status |
|---|---|---|
| 1 | Completed instructions and REP iterations (decision-16) | Exists. Unchanged in meaning, API and cost. |
| 2 | Clock cycles, computed from the manufacturer's per-instruction tables | Added, opt-in per VM. |
| 3 | Bus cycles, prefetch queue, wait states per access, DMA and refresh, device lockstep | Out of scope. A separate future project. |

### What level 2 is

A cycle count is a deterministic function of the model, its table revision and
the executed instruction stream. It is the same on every host and in every tier.
It is exactly what the manual's table computes, and it inherits the manual's own
limits. The 80286 PRM says its counts assume no bus delays and that a real program
"will average 5-10% more than the calculated clock count due to instruction
sequences that execute faster than they can be fetched from memory" (Appendix B,
p. B-6). The 80386 PRM's counts assume the instruction is already prefetched and
decoded, no wait states, no bus HOLD, no exceptions and aligned operands
(§17.2.2.3, p. 17-10). Level 2 makes no stronger claim than its source. An
embedder that needs board-exact time still calibrates; level 2 replaces a
hand-tuned instructions-per-tick rate with one frequency and, at most, one scale
factor. The maintainer accepted that outcome: anything closer is level 3.

### Cost components

| Component | Source of the value | Resolved |
|---|---|---|
| Fixed cost of the instruction form; register against memory operand | Table entry (`mem` label, 80286 PRM p. B-6; slash notation, 80386 PRM p. 17-10) | Lift time |
| Effective-address calculation | 286: one clock for base + index + displacement (p. B-6). 8086 class: the manual's EA table (to pin). 386: none given | Lift time |
| Real or protected mode | `pm` / `pm=` variants | Lift time, from the context |
| Conditional branch taken or not | `noj` label (286); paired counts (386) | Run time, at the branch |
| Refill after a taken transfer | 286: one clock per byte of the next instruction executed. 386: `m`, the component count of the next instruction (p. 17-11) | Run time, when the target instruction starts |
| REP | Per-iteration count times `n`, plus the fixed part | Run time, per completed iteration |
| Count-dependent shifts, rotates | Per-bit count where the table gives one | Run time |
| Data-dependent multiply | 386: the manual's early-out formula on the multiplier (ch. 17, IMUL/MUL notes) | Run time |
| Odd-address word operand | 286: two clocks per 16-bit memory reference at an odd physical address (p. B-6). 386: not given, not modelled | Run time, on the access |
| Uniform memory wait states | 286: one clock per wait state per memory read (p. B-6) | Lift time, from a per-VM setting |
| Interrupt and exception entry | The table's interrupt entries | Run time, at delivery |
| x87 instructions | 80287: "typical" column of Table 2-14 (80286 PRM Part II) | Later bead |

Where a table gives only a range and no formula — the 8086-class multiply and
divide are expected to — the entry stores the range and one engine-wide policy
function charges the midpoint, rounded down. The policy is engine behaviour, not
a hardware claim, and is labelled as such; because the table keeps the range, the
policy can change without re-deriving a table.

### Carried in the IR

Costs are computed by the lifter from the catalogue's table and stored in the IR
block as a side table, `IrBlock::timing`, one compact record per instruction in
`InsnStart` order: base cost, taken-branch difference, per-iteration cost, the
refill count of this instruction as a transfer target, and a tag for a
run-time-dependent part. The op stream itself is unchanged, so IR snapshots and
every backend that ignores timing are unaffected. `timing` is `None` unless the VM
selected level 2; the lifter then computes nothing. The level is fixed before vCPU
creation, so it needs no key dimension.

### Run API

```rust
vm.set_timing_level(TimingLevel::Cycles);       // default: Instructions; before new_vcpu
vcpu.cycles() -> u64                            // cumulative, monotone; 0 at level 1
vcpu.add_cycles(n)                              // embedder-charged time
StepOptions::with_cycle_limit(Option<u64>)      // optional second budget
StepRun::cycles() -> u64                        // charged in this call
StepRun::final_unit_cycles() -> u32             // cost of the last charged unit
```

The counters and the optional cycle limit are delivered together.

- **Steps do not change.** `StepRun::steps`, `retired_instructions`, the
  retirement relation and every row of decision-16 §2 are untouched at both
  levels. Cycles are reported beside steps, never instead of them.
- **Cycles follow steps.** A unit is charged its cycles when it is charged its
  step. An incomplete instruction or iteration — an MMIO request, a fault, a
  fetch failure — is charged no step and no cycles; its completed predecessors
  keep theirs. `IN`/`OUT` are charged at the port handoff, as their step is.
- **Delivery is the one exception.** Delivering an interrupt or a fault costs zero
  steps (decision-16) but is charged the table's interrupt-entry cycles, because
  the hardware spends them. The manuals give no cost for the abandoned part of a
  faulting instruction; it is charged nothing, as engine policy.
- **Cycle limit.** With a limit, execution stops at the first unit boundary at
  which the cycles charged in this call reach it. The overshoot is less than one
  unit and is visible in `StepRun::cycles`. The stop is
  `Exit::BudgetExhausted`; the two counts tell which budget ended the call. The
  step limit keeps its exact meaning and both may be set.
- **Attribution.** An embedder that time-stamps a device access at the start of
  the stopping instruction subtracts `final_unit_cycles` for the stops where
  decision-16 counts the stopping instruction (`PortIo`, `Hlt`).
- **Halt and idle time** stay with the embedder, as today; it advances its clock
  and may record it with `add_cycles`.
- Selecting level 2 on a model without a table, or on Application Long64/Compat32,
  is refused at configuration.

### Wait states, DMA and refresh: no per-access hook

Level 2 exposes no callback on memory access. A hook on every access would put an
indirect call in the hottest path and would be level 3 by another name. Two things
are offered instead. A per-VM count of uniform memory-read wait states is folded
into each instruction's base cost at lift time; this is exactly the adjustment the
80286 PRM describes and it costs nothing at run time. And `add_cycles` lets the
embedder charge what only it knows — wait states of a device it serviced at a
port or MMIO exit, DMA and refresh stealing, halted time — between runs. Per-region
wait states, the prefetch queue and cycle stealing inside an instruction are not
modelled.

### Interpreter and compiled accounting

The interpreter gets a timing specialisation selected at API entry, in the manner
of decision-16 §5: the level-1 walk contains no cycle load, add, branch or store,
and must compile to the code it has today. In the timing walk, `InsnStart` adds
the base cost (and the refill count if the previous instruction transferred),
branch ops add the taken difference, the REP helper adds per iteration, and word
accesses add the odd-address term branch-free.

Compiled Machine blocks do not exist yet (§8). When they do, a block adds its
static cycle sum at entry and its run-time parts where they occur, under the same
"exact in every tier" rule as steps; the witness is `jit == interp` on the cycle
count.

### Table revisions

A corrected table value changes cycle counts for every guest. Each catalogue entry
carries a timing revision, readable by the embedder, and every change to a table
is recorded with the citation that justified it. Step counts are not affected by
table revisions.

## Performance requirement

The model-driven path must keep the interpreter performance that xj-193.3 reached
on Application Real16, and level 2 must be free when it is off.

**Measure.** `x86jit-bench/examples/real16_loops.rs`, release build, the host
instruction counter the benchmark already reads (`real16_loops.rs:10-34`), host
instructions per guest step, median of three serial rounds; and wall time per step
of the `dos-mixed` workload, median of five rounds interleaved between baseline and
candidate, because the PC-class embedder's own measure is the wall time of a
session of which the interpreter is about 57 %.

**Variants.** The first unification bead extends the benchmark to run every loop
under: the alias (`set_cpu_mode(Real16)`); `I80286` real mode with A20 enabled and
with A20 disabled; `I80386` real mode; and, when it exists, `I8088`. It records the
present LegacySystem numbers before anything changes; that baseline does not exist
today.

**Gates** (thresholds confirmed by the maintainer):

| When | Requirement |
|---|---|
| Every bead touching the shared lifter, IR, walk or dispatcher | Alias: host instructions per step within +0.5 % of the parent commit on every loop; `dos-mixed` wall time within +2 %. Decision-22 measured run-to-run differences of −0.02 % to +0.16 % on this metric. |
| Machine real mode gains the Direct class (U2) | `I80286` and `I80386` real mode, A20 enabled or disabled, within +2 % of the alias on every loop of 64 or more guest steps per `run` call. |
| The single dispatcher lands (U5) | The same, on every loop, including one guest step per `run` call (`ip-boundary-1`). |
| The alias switches to the Machine path | Alias within +0.5 % of its pre-switch numbers. A regression blocks the switch; it is not traded against the cleanup. |
| Level 2 lands | Level 1 numbers unchanged within noise, and the level-1 walk's disassembly compared as in decision-16 §5. Level-2 overhead is measured and recorded; target at most 5 % on `dos-mixed`. |
| Every bead | Application Long64/Compat32 emitted code unchanged, by decision-18 witness 6. |

**Amendment (U2, 2026-11-12, accepted by the maintainer).** U2 measured Machine
real mode within +2 % of the alias on every loop except `ip-boundary-1`, which runs
one guest step per `run` call: about +36 % (U0: +113 %). The remaining cost is per call:
the entry validates the public `CpuState`, derives the context and enters the
separate legacy run wrapper. U5 removes that duplication, so the one-step gate
moves to U5. U2 meets the gate on loops of 64 or more steps per call. The alias
gates are unchanged.

**Amendment (U4, 2026-11-13, accepted by the maintainer).** A U4-only alias switch
failed the alias gate. The best prototype measured `ip-boundary-1` at +1.54 % over the
pre-switch alias, which is about 39 host instructions per `run` call. The other 14
loops were 0.2 % to 2.2 % faster. The extra cost is the Machine run entry, and U5
removes it. U4 and U5 therefore land together, as one change. That change must meet
the U4 alias gate on every loop, `ip-boundary-1` included, against the pre-switch
(U3) numbers, and the U5 gates as well. The alias switch is not landed without
its gate.

**Amendment (U5, 2026-11-13, accepted by the maintainer).** After the switch, the
alias runs on the Machine path through a private profile whose context is fixed.
On most loops it is 4 % to 15 % faster than the U3 alias. The U5 Machine gate
therefore counts against the pre-switch (U3) alias numbers, the same baseline as
the U4 alias gate: `I80286` and `I80386` real mode, A20 enabled or disabled, within
+2 % of the U3 alias on every loop, `ip-boundary-1` included. The comparison with
the current alias is still measured and reported.

**Amendment (U4+U5 landing, 2026-11-13, accepted by the maintainer).** The change
lands with one measured miss, a slowdown and not a defect: Machine `ip-boundary-1`
is +2.56 % to +2.63 % over the U3 alias (2605.9 to 2607.7 host instructions per
call against 2540.8), 14 to 16 instructions per call above the gate. The fix for
an abandoned REP flag origin (xj-srh.5, found in the final review) added them;
the round before it measured +1.86 %. Every other gate is met. Bead xj-s3p.11
carries the miss, with the U5 gate as its acceptance; the gate itself is
unchanged.

**Amendment (xj-ir6.12 landing, 2026-11-14, accepted by the maintainer).** The split
of an access across adjacent RAM and ROM-kind regions lands with a measured
slowdown, not a defect. Against d4a58e2a, the worst real16_loops cell is +2.44 %
(53 of 80 cells exceed +0.5 %). The store probe is +3.35 % at width 8 and +5.43 %
at width 32. The maintainer accepted it because a Real16 embedder needs the split
now. Bead xj-ir6.31 carries the miss, with this requirement's +0.5 % gate as its
acceptance; the gate itself is unchanged.

**Amendment (C4a landing, 2026-11-15, accepted by the maintainer).** C4a, the NEC V20/V30
models and the 8080-mode exit contract, lands with a measured slowdown, not a defect. Against
060d446a, 106 of 112 real16_loops cells are within +0.5 %. The six misses are on SMC workloads:
- smc-next-1, I8088/I8086: +1.35 % and +1.33 %;
- smc-next-64, alias/I8088/I8086: +0.50 %, +0.81 % and +0.79 %;
- smc-rmw-64, alias: +0.55 %.

The store probe is within +0.02 %. The measured mechanism is a change in rustc's codegen-unit
partition, triggered by the new public `Exit::I8080Mode` variant; the paths with misses run no
NEC code. The maintainer accepted the slowdown because the sibling 8080 project needs the
contract for feature completeness. Bead xj-50w.17 carries the miss, with this requirement's
+0.5 % gate as its acceptance; the gate itself is unchanged.

**Amendment (xj-ir6.31 closing, 2026-11-15, accepted by the maintainer).** xj-ir6.31 moves
the xj-ir6.12 region split off the hot path, which removes the slowdown the "xj-ir6.12 landing"
amendment accepted. It was measured against a373f11b with 060d446a reverted, that is, main
without the split:
- 140 of 144 real16_loops cells are within +0.5 %;
- the store probe is below the base in all 8 cells;
- the four misses are smc-next-64 on the four twenty-bit paths, at +0.50 % to +0.53 % on the
  median of 15 processes.

The base is bimodal per process in those cells, with two values about 0.55 % apart, and the
candidate is within −0.06 % of its upper value. The maintainer accepted these four cells as
measurement noise and closed xj-ir6.31. The gate itself is unchanged.

## Migration

Steps are ordered, not dated. Each is one or more beads from the plan below and
lands whole with its witnesses and the full gate.

| Step | Change | 64-bit userland embedder | PC-class embedder |
|---|---|---|---|
| M0 | Today. | Application Long64/Compat32. | Application Real16; AT-class board moving to `set_legacy_cpu_model(I286)`. |
| M1 | API and catalogue (U1). `CpuModel`, `Vm::set_cpu_model`, `Vm::set_fpu_model`, `ExecutionProfile::Machine`. Old names become deprecated aliases with identical behaviour. | No change. No touched code path. | May switch to `set_cpu_model(CpuModel::I80286)` + `set_fpu_model(FpuModel::None)`. Old calls compile, with deprecation warnings, and behave as before. One difference to choose at the switch: `set_cpu_model` delivers faults in the guest by default; selecting `Outward` reproduces LegacySystem. |
| M2 | Direct class for Machine real mode (U2). | No change. | `I80286`/`I80386` real mode reaches Application Real16 speed. No API change. Real-mode A20 wrap now works at full speed. |
| M3 | Unified stack/control/flag lowering (U3); xj-5t9 closes. | No change. | **Behaviour change on the alias only:** the 35 forms become manual-correct. On `I80286` they were already #UD; on `I80386` already correct. |
| M4 | Alias switch (U4): `set_cpu_mode(Real16)` runs on the Machine path. | No change. | Alias behaviour identical to M3. `execution_profile()` still reports `Application(Real16)` for an alias VM. `cpu.legacy` of an alias vCPU is no longer all-zero: it holds selector-derived caches. |
| M5 | One walk, one dispatcher (U5). Internal. | No change. | No change. |
| M6 | Models beyond 286/386 (C-beads), level 2 (T-beads), each additive. | No change. | New `CpuModel` variants and the timing API appear; nothing existing changes. |
| M7 | Removal of the LegacySystem names, after both embedders confirm: `set_legacy_cpu_model`, `LegacyCpuModel`, `legacy_cpu_model()`, the `LegacySystem` alias, the public `CpuGeneration`. The `set_cpu_mode(Real16)` alias is **not** part of this step. | No change, provided it never named them. | Compile errors on the removed names only. |
| not scheduled | Removal of the `set_cpu_mode(Real16)` alias. Decided when the PC-class embedder has moved fully to models. | No change. | Decided with it. |

Rules of the migration:

- **Zero change for Application Long64/Compat32** is a gate, not an intention: no
  bead in this plan edits their lowering, keys or code generation, and each proves
  it with emitted-code comparison.
- **`CpuMode::Real16` is never marked `#[deprecated]`.** An Application embedder
  with an exhaustive `match` on `CpuMode` would get a warning for a variant it does
  not use. The deprecation is documentary until M7, and the variant itself stays.
- **`ExecutionProfile`** gains `Machine` by rename with a deprecated associated
  constant `LegacySystem`, so existing expressions and patterns keep compiling;
  U1 proves it with a compile test. The enum becomes `#[non_exhaustive]` at M7, not
  before.
- **`CpuState::legacy`** keeps its name and layout (decision-18 §2). Renaming a
  public, offset-frozen field buys nothing.
- **Type names** (`LegacyDecodeContext`, `LegacyRegime`, `LegacyContextError`)
  get neutral names with deprecated aliases at M1 and lose the aliases at M7.
- **The alias outlives M7.** It stays until the PC-class embedder has moved fully
  to models; its removal is decided then.

What breaks, and when: nothing at M1, M2, M4, M5, M6; the 35 forms on the alias at
M3; the removed LegacySystem names at M7. The alias keeps working throughout.

## Provenance

### Rule

A model's behaviour is licensed by its manufacturer's manual, cited by section and
page. The differences an Intel manual lists against an earlier processor
(80286 PRM Appendix C, 80386 PRM §14.7, SDM Vol 3B ch. 25) license the *later*
processor's side and Intel's statement of the earlier one; where they suffice, the
quirk table of §3 uses them. A cycle count is licensed only by a table.

Hardware-captured corpora — the SingleStepTests 80286 corpus pinned at
`oracles/ss286`, and its 8088 and V20 counterparts, which carry per-cycle bus
traces — are comparands. They can show that an implementation or a reading of a
manual is wrong. They cannot supply a behaviour or a cycle count that no manual
states. A disagreement is recorded with the measured bytes and the manual wins,
as PROVENANCE.md §2 requires.

This record does not amend PROVENANCE.md. In particular it adds no category for
behaviour that is observed on hardware but stated by no manual (§3).

### Manuals to pin before the facts that need them

Nothing below was downloaded or pinned for this record. Titles and file names are
as they appear in the public archive's directory listings (read on the
`bitsavers.trailing-edge.com` mirror, which `oracles/MANIFEST.md` already uses for
the two PRMs); contents, section numbers and table numbers are unverified until
the document is pinned and read. `…` stands for that mirror's root.

| Document | Identifier | Public source | Needed for |
|---|---|---|---|
| *The 8086 Family User's Manual*, Intel, October 1979 | order 9800722-03 | `…/components/intel/8086/9800722-03_The_8086_Family_Users_Manual_Oct79.pdf` | 8086/8088 instruction set, undefined encodings, string-interrupt behaviour, instruction timing and EA tables |
| *iAPX 86, 88 User's Manual*, Intel, 1981 | (listed without an order number) | `…/components/intel/8086/1981_iAPX_86_88_Users_Manual.pdf` | Second edition of the same; choose one as the pin, keep the other as a cross-check |
| *iAPX 86/88, 186/188 User's Manual, Programmer's Reference*, Intel, 1983 | order 210911-001 | `…/components/intel/80186/210911-001_iAPX86_88_186_188_Programmers_Reference_1983.pdf` | 80186/80188 additions and differences, their timing; a second witness for the 8086 |
| *iAPX 86/88, 186/188 User's Manual*, Intel, 1985 | order 210912-001 | `…/components/intel/8086/210912-001_iAPX_86_88_186_188_Users_Manual_1985.pdf` | Hardware companion to the above |
| *V20/V30 User's Manual*, NEC, October 1986 | (NEC document number to record when pinned) | `…/components/nec/V-Series/V20_V30_Users_Manual_Oct86.pdf` | V20/V30 instruction set, extensions, 8080 emulation mode, timing |
| A 16-bit V-series user's manual, NEC, June 1997 (title as the file names it) | U11301EJ5V0UM00 | `…/components/nec/_dataBooks/U11301EJ5V0UM00_16-BIT_V_Series_Jun97.pdf` | Later NEC reference; a cross-check for the 1986 manual |
| *80387 Programmer's Reference Manual*, Intel, 1987 | order 231917-001 | `…/components/intel/80386/231917-001_80387_Programmers_Reference_Manual_1987.pdf` | `FpuModel::I80387` |
| *386 SX Processor Programmer's Reference*, Intel, 1989 | (listed without an order number) | `…/components/intel/80386/1989_386SX_Processor_Programmers_Reference.pdf` | Only if a 386SX entry is requested |
| An 8087 reference (the numerics supplement to the 8086 family manual) | to locate | to locate | `FpuModel::I8087` |
| *i486 Processor Programmer's Reference Manual*, Intel, 1990 | to record when pinned | `…/components/intel/80486/i486_Processor_Programmers_Reference_Manual_1990.pdf` | The later 486/Pentium-class decision, which also names its Pentium-class references |

All are © their publishers with no redistribution licence, so each would be a
fetch-only entry in `oracles/MANIFEST.md`, like the two PRMs already there. The
SingleStepTests 8088 and V20 corpora would be pinned as the 80286 one is:
documentation and revocation list vendored, corpus fetched by the consumer.

### The 80286 over-length fault: two statements in one manual

The 80286 PRM states the over-length instruction fault two ways: Appendix B lists
"exceeding the maximum instruction length of 10 bytes" under #GP (13), which the
engine follows (decision-21; `lift/mod.rs:392-397`); Appendix C item 10, p. C-2,
says "Exception 6 occurs".

Decided: the engine keeps #GP, per Appendix B, and this section is the record of
the discrepancy. A bead (X1 below) runs the over-length case against the
hardware-captured 80286 corpus. The corpus can falsify the Appendix B reading; it
cannot license the Appendix C one. If it contradicts Appendix B, the finding goes
back to the maintainer with the measured bytes, and nothing changes until then.

## Bead plan

Keys below are provisional; the supervisor assigns ids. "8086" in the last column
marks a prerequisite of any 8086/8088 work.

### Structure

- **xj-50w** keeps its role from decision-18 §1 and is restated: *the CPU-model
  catalogue and the models before the 286*. The C-beads are its children. The
  catalogue itself (C1) is early, because the 286 and 386 need it; the pre-286
  models (C2-C5) come after the 386 system features and the compiled route.
- A new epic, **machine-path unification**, holds the U-beads. **xj-5t9** is
  re-parented to it and is closed by U3.
- A new epic, **cycle timing**, holds the T-beads.
- A new epic, **compiled Machine execution**, holds the J-stages of §8.
- **xj-srh** remains *system features of the 286 and 386*. Its open children keep
  their content; their ordering against the unification is fixed below.
- Pinning manuals is work in the `oracles` repository, tracked there, with a
  submodule bump here.

### Order and dependencies

| Key | Title | After | 8086 |
|---|---|---|---|
| U0 | Baseline: `real16_loops` variants for the alias, LegacySystem I286/I386 real mode, A20 off; record numbers; no engine change | — | yes |
| U1 | `CpuModel`, `ModelProfile` catalogue for `I80286`/`I80386`, `Vm::set_cpu_model`, VM-level FPU default, `Machine` rename, deprecated aliases; behaviour unchanged | U0 | yes |
| U2 | `MachineContext` with the segment class; Direct lowering, shared-cache multi-instruction blocks and A20 mask for Machine real mode; measure and confirm or amend §5 | U1 | yes |
| U3 | One width-aware lowering of stack, control, flag-image, frame and far-pointer instructions; retire the `*_real16` helpers; empty the parity allowlist; close xj-5t9 | U2 | no |
| U4 | Alias switch: `set_cpu_mode(Real16)` on the Machine path through the internal entry; explicit exception-delivery option; remove the Application Real16 dispatch branch | U3 | no |
| U5 | One interpreter walk and one dispatcher loop; lift-marked rollback; remove `interp::legacy::walk` and `legacy_loop`. Lands together with U4 (Performance requirement, U4 amendment) | U4 | no |
| U6 | Remove deprecated names (M7) | both embedders migrated | no |
| P1 | Pin the 8086 family manual | — | yes |
| P2 | Pin the V20/V30 manual | — | no (V20 yes) |
| P3 | Pin the 86/88, 186/188 programmer's reference | — | no (second witness) |
| P4 | Pin the SingleStepTests 8088 and V20 documentation | — | no |
| C1 | `isa` decode tables and the undefined-encoding policy per model, replacing decoder-feature filters; quirk fields with the values licensed today; the 386 and 286 identity attributes (this is xj-srh.17, re-parented) | U1 | yes |
| R1 | Review xj-srh.16 against `5c95116`: list the 386 integer forms that Machine real mode on `I80386` still refuses or gets wrong, and re-scope the bead to that list | — | no |
| X1 | Run the 80286 over-length instruction case on the hardware-captured corpus; record the vector it shows against Appendix B and Appendix C | — | no |
| C2 | `I8088` and `I8086`: the 8086 quirk set, `system = None`, 20-bit addressing, permanent Direct context | U2, C1, P1; scheduled after the xj-srh system features and the J stages the PC-class embedder needs | — |
| C3 | 8088 differential lane on the hardware corpus, in the shape of `ss286` | C2, P4 | — |
| C4 | `NecV20` and `NecV30`: extensions; BRKEM/RETEM/CALLN and the MD flag; the 8080-mode `Exit` contract of §3 (until then BRKEM is a typed refusal) | C2, P2 | — |
| C5 | `I80186`/`I80188` | C2, P3 | — |
| C6 | `FpuModel::I80287`, then `I80387`, `I8087`: generation differences | U1; manuals | — |
| T1 | Timing table format, policy function for ranges, the 80286 table from Appendix B, value-pinning witness | U1 | no |
| T2 | `IrBlock::timing`, the timing walk, `cycles`, `add_cycles`, cycle limit, wait-state setting; level-1 code unchanged | U2, T1 | no |
| T3 | 80386 table (chapter 17), `pm=` and `m` | T2 | no |
| T4 | 8086/8088 table; V20/V30 table | T2, C2/C4, P1/P2 | — |
| T5 | 80287 execution times | T2, C6 | no |
| J0-J5 | §8 | U2, then xj-srh.6/.7/.9/.10 | no |

The maintainer ordered the 8086-class wave after the 386 system features and the
compiled route. The prerequisites below say what that wave needs, not when it
starts; P1 can be pinned at any time.

**Amendment (order, 2026-11-13, accepted by the maintainer).** The sibling
8080-family project waits on C4 for its joint witnesses. C1, C2 and C4 therefore
move ahead of the 386 system features and the J stages: C1 starts when U4 and U5
have landed (C1 changes decoding, so it may not land between U3 and U5), then C2,
then C4. P1, P3 and P4 are pinned now. C3, C5, C6 and the T stages keep their
places.

**Amendment (C4 split, 2026-11-14, accepted by the maintainer).** C4 is split in two
so that the 8080-family project's joint witnesses do not wait for the V-series
extension instructions. C4a: the `NecV20`/`NecV30` entries with the 8086 core and
the 80186 additions, the MD flag, BRKEM, RETEM and CALLN, and the 8080-mode `Exit`
contract agreed with that project. Three points were decided in its review:
- the core decodes RETEM and CALLN when the adapter hands back at the ED escape;
- BRKEM and CALLN clear IE and BRK as interrupt entry does (NEC UM p. 6-1; ASSUMED);
- `halted` is part of the exchanged view.

C4b: the native-mode extension instructions (the bit, BCD-string and nibble-rotate
groups and the rest the NEC manual lists). Until C4b lands, they end in
`Exit::UnknownInstruction`.

Prerequisites of 8086/8088 work, in order: **U0, U1, U2, C1, P1.** U3-U5 are not
prerequisites: an 8086 model needs the Direct class and the catalogue, not the
removal of the old path. C2 should still land after U3 if U3 is close, so that the
8086 stack and control quirks are written once, in the lowering that survives.

### The open xj-srh children

xj-srh.19 has landed (`5c95116`) and is not listed. The placement of .16, .17,
.18 and .20 follows their bead texts as the maintainer relayed them. The rule
behind every row: a child that adds or changes *lowering or interpretation* lands
either before U3 starts or after U5 ends, never between, so that nothing is ported
twice; a child about state, memory or the embedder API is independent of the
U-beads.

| Bead | Placement |
|---|---|
| xj-srh.5 (exact stepping in both regimes) | Stays. Lands before U5, or is folded into it: U5 is where the two dispatchers become one. |
| xj-srh.6 (protected IDT delivery) | Stays. After U3: it extends the single delivery helper. `Outward` remains selectable after it. Prerequisite of J0. |
| xj-srh.7 (paging) | Stays. Independent of U3-U5. Must implement the translation interface of §8. Prerequisite of J3. |
| xj-srh.8 (LDT), xj-srh.9 (TSS, privilege), xj-srh.10 (I/O privilege) | Stay. After U3, on the unified lowering. .9 and .10 are prerequisites of J4. |
| xj-srh.12 (reset fetch) | Stays, independent. The catalogue gains each model's reset image as data. |
| xj-srh.13 (checkpoint) | Stays. Adds the model identity, FPU model, timing level and cycle counter to what is exported. After T2 if level 2 has landed. |
| xj-srh.16 (386 integer instruction set in legacy real mode) | Largely delivered by xj-srh.19: legacy real mode accepts the full Application Real16 set, with 386 semantics for the 66h/67h forms on I386. R1 reviews what remains and re-scopes the bead to it. The remainder is lowering, so it follows the rule above. |
| xj-srh.17 (386 identity: AC/ID fixed at 0; #UD for CPUID, RDTSC, RDMSR/WRMSR, WBINVD/INVD/INVLPG, BSWAP, CMPXCHG, XADD; 286 gating) | Becomes catalogue attributes. Re-parented to xj-50w and delivered with C1, after U1. Wave 1: the 386DX board needs it. |
| xj-srh.18 (hardware task switch through a 286/386 TSS) | Stays in xj-srh: a system feature of `I80286` and `I80386`. After .8 and .9, on the unified lowering (after U3). In the compiled route it is reached only through a helper (J4). |
| xj-srh.20 (real-mode TF single-step through the IVT, per-model inhibit rules after segment loads; in progress) | Stays, and finishes on the legacy path now, before U3 starts. Its per-model rules move into catalogue quirk fields with C1. U4 and U5 carry its delivery into the single dispatcher. The alias does not gain single-step. At timing level 2 the trap is charged as any delivery is. |

## Consequences

- One implementation of each instruction serves every 16- and 32-bit Machine
  model. A model is data in a catalogue plus the lowerings its quirks select.
- xj-5t9 stops being a list of defects and becomes a consequence of the model:
  #UD on the 286, 32-bit semantics on the 386.
- Machine real mode on the 286 and 386 reaches the Application Real16 speed, and
  models with no system features never pay for descriptor state.
- The embedder API has one selection call and no profile to understand.
- A cycle count is available beside the step count at no cost to those who do not
  ask for it.
- Decision-18's interpreter-only rule is no longer permanent. It holds until a
  later decision admits a first compiled Machine class.

Risks:

- **The alias switch is the dangerous step.** It moves every existing Real16 test,
  the hardware corpus harness and the PC-class embedder onto different code. It is
  gated on instruction-count parity, the frozen IR snapshots (updated once,
  deliberately, at U3), the corpus pass rate and the benchmark.
- **The Direct/Checked boundary is new state to get wrong.** A context wrongly
  classified Direct skips a limit check. The class is derived in one function and
  witnessed against every transition decision-18 §3 lists.
- **Lift-time quirks multiply IR variants, not interpreter code.** The cost is in
  the lifter's size and in witness coverage per model, which the catalogue makes
  enumerable.
- **Level 2 can be mistaken for hardware timing.** It is the manual's arithmetic.
  The documentation of the API says so with the manual's own caveats.
- **The compiled route is large.** Paging, privilege and exact accounting in
  compiled code are each a project. This record commits to keeping the route open,
  not to walking it.

## Alternatives considered

| Choice | Assessment |
|---|---|
| Keep both paths; add 8086 and V20 to each | **Rejected.** Doubles every model's work and repeats the drift that xj-srh.19 measured. |
| Make Application Real16 model-aware and leave LegacySystem as it is | **Rejected.** The 286 and 386 would still have two real modes with two segment models, and the board moving between them would keep paying the parity tax. |
| Move everything to the legacy path as it is | **Rejected.** It discards the xj-193.3 result; the per-instruction costs in Context are structural. |
| Resolve model differences at run time, in the interpreter | **Rejected.** Every model would tax every other model's hot loop, and a compiled tier would inherit run-time model tests. |
| One public `CpuModel` entry for today's Real16 hybrid | **Rejected.** It would present a processor that never existed as a peer of real ones. It is an internal entry reachable only through the alias. |
| Put cycle costs in `InsnStart` or in a new op | **Rejected.** It changes the op stream for every VM, the IR snapshots and every backend's match arms. A side table is absent when unused. |
| A per-access timing callback | **Rejected.** It is level 3 by another name and costs an indirect call on the hottest path. |
| Derive cycle counts from hardware-captured traces | **Rejected as a source.** A corpus is a comparand; it may show a table reading is wrong. |
| Host page tables for guest paging | **Rejected.** Guest pages are smaller than the primary host's. |
| Extend Application Compat32 into a system mode for 486-class guests | **Rejected,** as in decision-18: it changes flat contracts and compiled code the userland embedder depends on. |

## Resolved questions

Answered by the maintainer on 2026-11-11. Each answer is folded into the section
named.

| # | Question | Answer | Why |
|---|---|---|---|
| 1 | Which models first, and where does the 8086-class wave go? | `I80286` and `I80386` first; then the 386 system features and the compiled route; `I8088`/`NecV20` after those (§2). | The order follows the one consumer with a schedule: 286 now, 386DX next, 486/Pentium-class after. |
| 2 | Are 486- and Pentium-class models in scope? | In scope for the library, named in the catalogue, designed by their own later decision (§2). | That is where the model axis meets `GuestCpuFeatures`, the CPUID tables and `CpuSource`; it deserves its own record. |
| 3 | Is the `set_cpu_mode(Real16)` alias removed at M7? | No. It stays until the PC-class embedder has moved fully to models; removal is decided then (§6, Migration). | A removal date set before the migration it depends on would be a guess. |
| 4 | Default exception delivery for `set_cpu_model`? | `InGuest`. `Outward` stays an explicit option, including after xj-srh.6 (§5). | The default should be what the processor does; some embedders still want every fault. |
| 5 | Is the staged compiled route right? | Approved as direction. Each stage needs its own decision; the mechanisms of §8 are the starting point. | The direction can be fixed now; the evidence for each stage does not exist yet. |
| 6 | Undocumented 8086-class opcodes? | Typed refusal. PROVENANCE.md unchanged (§3). | A manual cannot license what it does not describe, and a corpus cannot license at all. |
| 7 | V20/V30 8080 emulation mode? | The model decodes BRKEM, RETEM and CALLN and keeps the MD flag; 8080 execution is a separate core outside x86jit, entered through a typed `Exit` with a documented register mapping; the contract is designed with the V20 bead; BRKEM is a typed refusal until then (§3). | A second instruction set does not belong in a core whose dependency set is `{iced-x86}`. |
| 8 | Timing ranges without a formula? | The midpoint, rounded down (Timing model). | The least biased single value when the manual gives no rule; the table keeps the range. |
| 9 | Performance thresholds? | Confirmed: +0.5 % host instructions per step and +2 % wall time for the alias; Machine real mode within +2 % of the alias (one-step-per-call loops from U5, amended at U2); 5 % target for level-2 overhead. | They sit just above the measured run-to-run noise. |
| 10 | Cycle limit now, or counters only? | Both now: `StepOptions::with_cycle_limit` and the counters (Timing model). | The limit is optional and costs nothing when unset. |
| 11 | Is "one frequency and at most one scale factor" enough? | Yes, with the 80286 PRM's own 5-10 % caveat stated (Timing model). | Anything closer is level 3. |
| 12 | FPU per VM or per vCPU? | A VM default at construction; the per-vCPU setter stays and is not deprecated (§4). | It gives the embedder its one-place selection without removing an existing control. |
| 13 | The 80286 over-length fault: #GP or exception 6? | Keep #GP per Appendix B; record the Appendix C discrepancy; bead X1 checks the hardware corpus (Provenance). | The corpus can falsify Appendix B but cannot license Appendix C. |
| 14 | Where do xj-srh.16, .17, .18 and .20 go? | .16 is re-scoped after a review (R1); .17 becomes catalogue attributes under C1; .18 stays a 286/386 system feature after .8 and .9; .20 finishes on the legacy path before U3 (Bead plan). | Each follows from its bead text and the rule against porting lowering twice. |

## Open points

Two points arose from the answers. Neither blocks the unification work.

1. **Where RETEM and CALLN are decoded.** Answer 7 has the V20/V30 model decode
   BRKEM, RETEM and CALLN. BRKEM is met in the native instruction stream. If the
   NEC manual shows that RETEM and CALLN are encodings of the *emulation-mode*
   stream, the 8080 core meets them, not x86jit's decoder; x86jit would then
   expose their native-side effects — the return to native mode, the call to a
   native handler, the MD flag — as operations the embedder invokes when the 8080
   core stops on them. Which it is cannot be settled until the manual is pinned;
   the V20 bead (C4) settles it and shapes the `Exit` contract accordingly.
2. **The 486/Pentium-class decision against the 8086-class wave.** The answers
   place `I8088`/`NecV20` after the 386 system features and the compiled route,
   and give the 486/Pentium-class models their own later decision. They do not
   say whether that decision and its first model precede the 8086-class wave or
   may run beside it. §2 lists them in the same step as the compiled route; the
   maintainer confirms or reorders when that decision is opened.
