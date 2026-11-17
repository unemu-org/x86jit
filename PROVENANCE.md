# Provenance — where this engine's behaviour comes from

An emulator is a claim about how somebody else's hardware behaves. The claim is only
worth what its sources are worth, so this document records them: what we treat as
authoritative, what we merely compare against, what we deliberately do not model, and
what we did not copy.

The rule is the one [`oracles`](https://github.com/unemu-org/oracles) exists to serve:
**every behavioural fact must be derived from, and cite, a clean primary source, and the
citation has to be checkable by someone who was not there.** `oracles` is vendored here
as a submodule so the paths a citation names stay stable and the version is pinned by
commit:

```sh
git submodule update --init
cd oracles && ./fetch-oracles.sh verify   # offline; the vendored sources are already there
cd oracles && ./fetch-oracles.sh fetch    # adds the fetch-only ones (the Intel SDM, the APM, the psABI)
```

## 1. What is authoritative

| Source | Role | Pin |
|---|---|---|
| **Intel SDM**, combined volumes 1, 2A-2D, 3A-3D, 4 | **The** authority for instruction semantics, flag effects, the x87 environment and tag-word images, exception behaviour | Order Number `325462-092US`, June 2026, 5363 pp — `oracles/intel/sdm.pdf`, fetch-only |
| **AMD64 APM**, volumes 1-5 | **Second witness** where the SDM is silent or says "undefined". Also the natural authority for the Jaguar-class (x86-64-v2) target | Publication `40332`, revision 4.08 — `oracles/intel/amd64-apm.pdf`, fetch-only |
| **Intel 80286 and 80287 Programmer's Reference Manual** | **Source for 286- and 287-specific behaviour** the current SDM does not describe: MSW and LMSW-only protected-mode entry, the 286 descriptor format, 286 exception conditions, the 80287 coprocessor. A 1987 scan with an OCR text layer: check cited values against the page image | Order Number `210498-005`, 1987, 513 pp — `oracles/intel/i286-prm.pdf`, fetch-only |
| **Intel, Undocumented iAPX 286 Test Instruction** | **Source for the 80286 LOADALL instruction** (0F 05), which the 80286 PRM omits: the 102-byte table at 000800H, its entry formats, the CPL 0 rule and the effect of a later segment load (decision-23 amendment, LOADALL). A scan; pages are cited by the numbers printed on them (page 1 carries none). The document contradicts itself on the opcode: p. 2 encodes LOADALL as `00001111 00000101` (0F 05), while p. 13 calls it "opcode 0F04H", says that 0F05H stops the 80286 until RESET, and then that 0F05H "may be executed at any time in real address mode". x86jit follows p. 2's bit encoding and the description on pp. 1-3 (DERIVED; spec.md §17.10), and keeps 0F 04 #UD (80286 PRM App. B p. B-12) where p. 13 gives it exception 13 at CPL > 0 | Undated, 15 pp — `oracles/intel/i286-loadall.pdf`, fetch-only; the PCjs archive copy, sha256 `5c5a128ea7d5500c62f75efb24d1bac7ac392bb4f2fa21dce9a0277ac9d8e13d` |
| **Intel 80386 Programmer's Reference Manual** | **Second witness for 386-specific behaviour** the current SDM no longer describes as a 386 feature set (CR0 and EFLAGS masks, two-level paging permissions and reserved bits, reset state); required by decision-18 before such a fact is implemented. A 1986 scan with an OCR text layer: check cited values against the page image | Order Number `230985-001`, 1986, 523 pp — `oracles/intel/i386-prm.pdf`, fetch-only |
| **NEC µPD70108 (V20) and µPD70116 (V30) User's Manual** | **Source for the V20/V30 models** (decision-23 C4): the 8080 emulation mode (entry with BRKEM, return with RETEM, native-mode calls with CALLN, the MD flag) and the V-series instruction-set extensions, which the Intel manuals do not describe. A scan with an OCR text layer: check cited values against the page image | Stock No. `500350`, October 1986, 228 pp — `oracles/nec/v20-v30-um.pdf`, fetch-only |
| **Intel, The 8086 Family User's Manual** | **Source for the 8086/8088 models** (decision-23 C2): instruction set, flags, bus and the 8086-specific behaviours the current SDM does not describe. A 1979 scan with an OCR text layer: check cited values against the page image | Order Number `9800722-03`, October 1979, 748 pp — `oracles/intel/i8086-um.pdf`, fetch-only |
| **Intel, iAPX 86/88, 186/188 User's Manual, Programmer's Reference** | **Second witness for the 8086/8088 models** and the source for the 80186/80188 models (decision-23 C5). A 1983 scan with an OCR text layer: check cited values against the page image | Order Number `210911-001`, May 1983, 550 pp — `oracles/intel/i186-prm.pdf`, fetch-only |
| **x86-64 psABI** | Process initialisation: the initial stack, the auxiliary vector, the exception→signal map, and the syscall calling convention (RDI, RSI, RDX, **R10**, R8, R9 — not RCX) | commit `e1ce0983`, LaTeX sources — `oracles/psabi/`, fetch-only |
| **Linux x86 syscall tables and process/signal uapi** | Syscall numbers for both the x86-64 and the i386 `int 0x80` tables; the `siginfo`/`sigcontext` layouts | `v6.12` — `oracles/linux-x86/`, vendored |

Cite by **volume and section**, not by "the SDM". `[SDM Vol 1 §8.1.7]` is a citation;
"per Intel" is not — the second cannot be checked, and an unfalsifiable citation is worse
than none because it looks like one.

**Where this repository stands today, stated honestly** (counted, not estimated — rerun the
count when you change this paragraph): **537** internal `§spec` references against **62**
mentions of an external authority, of which only **8** name a volume and a section, and
**one** URL across the crates' Rust sources (35 across the whole tree once docs, scripts
and manifests are included — the figure that matters here is the Rust one, because a URL
in a doc is a link while a URL in a comment is a citation standing in for a real source). The sources above are what a precise citation can now
point at; retrofitting the vague ones is standing work, not a finished state.

### Witness tests — a citation you can execute

Prose citations rot silently: nobody notices when a comment and the code drift apart. So a
cited constant is also **pinned by value in a test**, which makes the claim falsifiable by
anyone, with **no stash and no network** — the manuals are for deriving *new* facts and for
human verification, never a build dependency.

`features.rs::cpuid_feature_bits_match_the_documented_positions` is the worked example. It
restates the CPUID feature-bit tables from SDM Vol 2A and asserts that a set containing
exactly one feature projects to exactly the documented bit. Trusting the `if_has(f, n)`
calls would be circular — the test would pass whatever the code said. Instead, moving AVX
from bit 28 to 27 fails it by name. That matters because advertising a bit we do not lift
surfaces as "glibc picked a string routine that traps", a very long way from its cause.

The same shape appears wherever a value comes from a document rather than from reasoning:
`x87::rc` (control-word bits 11:10, SDM Vol 1 §4.8.4) is witnessed by a test that `fistp`s
`(0.75, -0.75)` under each of the four rounding modes, a pair chosen because it separates
all four — a mis-decoded field cannot pass it.

A citation can also settle **how much to guarantee**, not just what a value is.
`x86jit-tests/tests/cross_modifying.rs` runs the two-processor protocol from
**SDM Vol 3A §11.1.3** ("Handling Self- and Cross-Modifying Code") — modifier stores the
code then raises a flag, executor polls, executes a serializing instruction, then runs it
— because that is the case the architecture actually defines. The same section calls the
*unsynchronized* form "model-specific", which is what stopped this engine from being held
to a stronger rule than the hardware offers: an acceptance criterion asking that a stale
translation can never run was replaced on the strength of that sentence (task-323). The
manual is as useful for bounding an obligation as for supplying a constant.

## 2. What is an oracle — and what each one can actually judge

An oracle is something we compare against. **None of them is an authority.** The
distinction is load-bearing here, because this engine has repeatedly been right where an
oracle was wrong.

| Oracle | Judges | Cannot judge |
|---|---|---|
| **Real host CPU** (`x86jit-tests/src/native.rs`) — forks, runs the snippet on bare hardware, captures state from a signal handler | Anything the silicon does, including VEX/EVEX semantics | x87 registers (not captured); needs an x86-64 host, so it is absent on the ARM lane |
| **Unicorn** (QEMU TCG), behind the `unicorn` feature | Broad cross-platform agreement, and the only oracle available on ARM | Anything QEMU gets wrong — see below. Its build predates TCG AVX |
| **The interpreter** | The JIT (`jit == interp` is the JIT's oracle) | Nothing, when both tiers share a bug — which is why neither is the ground truth |
| **SingleStepTests 80286** — captured from a real Harris N80C286-12 through ArduinoX86 | Real-mode per-instruction behaviour, from silicon | Only the 80286 subset; pinned at `oracles/ss286`, corpus fetched separately |
| **SingleStepTests 8088 and V20** — captured from real chips; documentation and opcode metadata only | Their capture formats, masks and stated limits, for the 8088 differential lane (decision-23 C3) and the V20 native mode | Not yet run against the engine; no V30 or 8080-mode coverage; documentation pinned at `oracles/ss8088` and `oracles/ssv20`, corpus not fetched |
| **objdump** | Our disassembly against binutils | Semantics |

**Unicorn is a comparand, not an authority.** The tree records around twenty places where
QEMU is wrong and hardware agrees with us. Three worked examples, all measured rather
than argued:

- `fnstenv` writes the reserved half-words as `0xFFFF`; QEMU writes zero. Silicon agrees
  with us.
- `fnstenv` masks all six FP exceptions after the store, per the SDM. QEMU omits the side
  effect entirely.
- On an MMX write, Intel sets the x87 exponent bytes to all-ones; Unicorn leaves them
  zero.

Where an oracle disagrees, the SDM settles it, and the divergence is written down with the
measured bytes — never quietly excluded.

**The host CPU's x87 pointer fields depend on the host, and the manuals say so.** A native
witness compares FIP, FDP, FOP, FCS or FDS only where `native::host_x87_pointers` measured
that the host stores the field the way the engine does, and prints why it skipped one.
Measured so far:

- AMD Zen 4 (Ryzen 7 7840HS): FXSAVE with FSW.ES = 0 stores zero FOP, FIP and FDP (APM
  Vol 2 §11.4.4.3 p. 354; Vol 5 FXSAVE p. 26). FNSTENV stores them regardless of ES
  (Vol 5 FSTENV p. 306).
- An Intel GitHub-hosted x86-64 runner: after `fld qword [0x220000]`, FXSAVE64 stored
  FDP = 0, because that CPU updates FDP only on an unmasked x87 exception (SDM Vol 1
  §8.1.8 p. 8-9, CPUID.(EAX=07H,ECX=0):EBX[6]). FNSTENV after an FLDENV of FCS = 0x4567
  and FDS = 0x89AB stored 0x0000 for both: the selectors are deprecated (same section,
  EBX[13]).
- The same runner, given FIP = 0x1234_5678_9abc_def0 by FXRSTOR64, saved
  0x0034_5678_9abc_def0 with FXSAVE64. A load of FIP keeps only the bits within the
  maximum linear-address width, here 57, and sign-extends the top one (SDM Vol 3A §4.5.3
  p. 4-6). That is not a vendor difference: the section applies to every Intel processor.
  The FDP of the same image, 0x2345_6789_abcd_ef01, came back whole: FDP is not one of
  the registers that section lists.
- The engine loads FIP the same way at its own width, 48 bits, on FXRSTOR64 and
  XRSTOR64 in both tiers and in `x87::exec_fxstate` (`memory::canonical`). That is the
  width it enforces on every Long64 memory access and the one its CPUID implies: LA57
  clear with Intel 64 set (Vol 3A §4.5 p. 4-4; §5.1.4 p. 5-6). The probe measures the
  host's width. The native witness compares the host's FIP with the engine's on a 48-bit
  host. On a 57-bit host it checks the host against the host's width and prints why it
  does not compare the two. The engine-side witnesses check 48 bits on every host.

## 3. What we do not model, and say so

Refusing to lift is a design position, not an omission, when the alternative is a
plausible-looking wrong answer:

- **x87 `FIP`/`FCS`, `FDP`/`FDS` and `FOP` are modelled**, so they are no longer listed
  here as a gap. Non-control x87 instructions record them as SDM Vol 1 §§8.1.8–8.1.9
  describe, and FXSAVE/FXRSTOR, XSAVE/XRSTOR, `fnstenv`/`fldenv` and `fnsave`/`frstor` carry them.
  A REX.W restore loads FIP canonical at 48 bits and FDP whole (Vol 3A §4.5.3 p. 4-6; see
  §2). What remains unmodelled: `fnsave`/`frstor` are lifted only in 16-bit code; the 32-
  and 64-bit forms remain **unlifted**.
- **The six FP exception flags ARE set** by arithmetic since task-328, and each rule is
  witnessed against the host rather than restated from the manual — including the two that
  are easy to get backwards: masked underflow needs the result to be both tiny and
  inexact, and ES follows the *masks*, not the exception. An unmasked exception is now
  **delivered** as `#MF` on the following waiting instruction, and the instruction that
  raised it is abandoned — both from SDM Vol 1 §8.6, whose enumeration of the six
  non-waiting instructions is what keeps a guest's own exception handler able to read and
  clear the status word. The stack-fault flag with C1, the condition codes C0/C2/C3 (which
  is what let `ficom`/`ficomp` be lifted at last) and the denormal-operand exception all
  followed. `TASK-328` is closed.

  One correction is worth keeping, because it is the shape this discipline exists for:
  §4.9.1.2 says the denormal exception is reported "if an **arithmetic** instruction
  attempts to operate on a denormal operand", which reads as excluding `fld`. The host
  says otherwise — `fld qword` of a denormal, alone in a program, leaves the status word
  at `0x3802`. `fld m32/m64` *converts*, and the conversion is what meets the denormal.
  The manual was not wrong; the reading was, and only the oracle could say so.
- **MXCSR** is stored as per-vCPU state and round-trips through `ldmxcsr`/`stmxcsr`
  (including VEX forms) and `fxsave`/`fxrstor`. Reserved bits fault before state commits
  (SDM Vol 1 §§10.2.3, 11.6.6; Vol 2A §3.3 LDMXCSR/FXRSTOR). The reset value is
  `0x1F80` (§11.6.4, Table 11-2). RC/FTZ/DAZ are stored, not yet applied to arithmetic,
  and arithmetic does not set sticky exception flags (xj-pi7.2). Oracles seed and capture
  the real register; the general comparator excludes the six sticky flags, while storage
  witnesses assert the complete value.

**Machine string access ordering.** SDM 325462-092US Vol 2B §4.3 MOVS
(Operation, pp. 4-105–106) defines `DEST := SRC`; its Real-Address Mode
Exceptions include a segment-limit violation. The 80386 PRM §17 MOVS,
pp. 17-112–113 (scan checked), gives the corresponding source-to-destination
assignment and whole-operand limit rule. Neither entry explicitly orders an
observable source device read against a destination segment-limit fault. The
Machine model forms/checks/reads the source first, then forms/checks/accesses the
destination, consistently in Direct and Checked execution. This source-first
ordering is an explicit engine policy for that unspecified interaction. The
`legacy_direct_parity` MOVSW source-MMIO/destination-limit witness pins both the
read request and the subsequent fault. Application Real16 keeps its existing
string engine and ordering.

## 4. What we did not copy

One third-party source is vendored: `vendor/cranelift-jit`, a modified fork of
Cranelift's `cranelift-jit` 0.135.8 (§5). It is compiler integration and supplies no fact
about how an x86 behaves. Its files keep their upstream licence, and each file the project
changed says so at its top. Outside `vendor/`, no file carries a foreign copyright or SPDX
header — checked, not assumed. Two further pieces of other projects' work are recorded in
§5: the V20 contract vectors from the sibling 8080-family project, and the musl libc that
`pthreads.elf` links statically. Nothing here is a port of another emulator's
implementation. Where another project is named in a comment it is in one of two roles:
*this is what mature translators do* (the memory-ordering design cites Box64's `STRONGMEM`
levels and FEX's RCpc use as precedent), or *this oracle is wrong here*.

`x86jit-core`'s dependency set is exactly `{iced-x86}` — the x86 decoder, the one thing a
recompiler legitimately needs — and `x86jit-tests/tests/boundary.rs` fails the build if
that ever changes. That tripwire is why extracting the Linux userland into `unemulinux`
was a move rather than an untangling.

## 5. Licence surface

This project is `MIT OR Apache-2.0`. The dependency graph, by crate:

| Crate | Direct dependencies | Licence |
|---|---|---|
| `x86jit-core` | `iced-x86` | MIT |
| `x86jit-cranelift` | `x86jit-core`, `cranelift{,-module,-native}`, `x86jit-cranelift-jit`, `memmap2`, `region`, `libc`, `wasmtime-internal-jit-icache-coherence` | Apache-2.0 WITH LLVM-exception; MIT OR Apache-2.0 |
| `x86jit-elf` | `x86jit-core`, `goblin` | MIT |
| `x86jit-tests` | `serde`, `ron`, `hex`, `serde_json`, `flate2`, `iced-x86`, `libc`, **`unicorn-engine`** | permissive, except as below |
| `x86jit-bench` | `serde`, `serde_json`, `iced-x86` | permissive |

`vendor/cranelift-jit` is a modified fork of Cranelift JIT 0.135.8, kept under its
original Apache-2.0 WITH LLVM-exception licence (the complete licence is alongside the
sources). The fork point is the crates.io `cranelift-jit` 0.135.8 package, checksum
`31d67d6c6a02ffd860599e55b03d4f37c6cf9be678c1d68dbfa6d1f61946e4ae`, published from
wasmtime commit `7b6c29ec3e9a49c97da231665cbd8eead9ccbda0` (path `cranelift/jit`). As
Apache-2.0 §4(b) requires, every file changed from that package carries a notice at its
top saying the x86jit project modified it: `Cargo.toml`, `README.md`, `src/lib.rs`,
`src/backend.rs`, `src/compiled_blob.rs`, `src/memory/mod.rs` and
`src/memory/system.rs`. `LICENSE`, `examples/jit-minimal.rs`, `src/memory/arena.rs`
and `tests/basic.rs` are byte-identical to upstream. The fork's `README.md` and
`Cargo.toml` name it a fork and point at this repository; the README lists each
file's change.
The package is named `x86jit-cranelift-jit` so embedders also resolve the explicit
path dependency when this workspace is consumed as a dependency; a root-only
Cargo patch would not propagate. The local changes add dual addresses, scoped
writes, provider relocation validation, allocation-generation queries and
finalized-definition retirement, never-published-definition discard, cleanup
of failed blob initialization and code-range entries for functions defined from
bytes (decision-17), plus a narrow system-provider
publication callback that reports protection/cache/pipeline errors as
`ModuleError::Allocation` instead of panicking or returning `Backend`. The callback
lets `x86jit-cranelift` attach operation/errno diagnostics and inject failures at
the OS boundaries; allocation policy and failure counters stay there. This
compiler integration supplies no x86 hardware semantics.

**`unicorn-engine` is GPL-2.0**, and this needs stating plainly because `spec.md §15` said
for a long time that "all core dependencies are permissive", which stopped being true when
the Unicorn oracle was added. It is:

- **optional**, behind the `unicorn` feature, off by default;
- confined to `x86jit-tests`, which is `publish = false` and which nothing else in the
  workspace depends on at build time;
- dynamically linked against the system `libunicorn.so` rather than statically built in;
- **enabled in CI**, so the default CI test binary does link it.

Nothing shipped from this repository links it. A consumer of `x86jit-core`,
`x86jit-cranelift` or `x86jit-elf` never sees it.

The one committed guest binary, `x86jit-tests/programs/pthreads.elf`, is our own program,
but it is statically linked against **musl libc**, which is MIT-licensed. Redistributing
the binary therefore requires musl's copyright and permission notice. It belongs in
[`x86jit-tests/programs/README.md`](x86jit-tests/programs/README.md), which holds a
placeholder until the text from musl's `COPYRIGHT` file is inserted.
Every third-party guest binary left with the Linux userland when `unemulinux` was split
out — which is also how this repository stopped shipping busybox (GPL-2.0), the glibc
loader (LGPL-2.1), and lua and CPython, both of which statically link GNU Readline and
are therefore GPL-3.0 combined works.

### Contract vectors from the sibling 8080-family project

`x86jit-tests/tests/legacy_nec_v20.rs` embeds, byte for byte, the V20 mode-switch
contract vectors (format `x80-v20-contract-2`) of `x80`, the sibling 8080-family
project:

- **Source:** `x80`, <https://github.com/unemu-org/x80>, file `x80-v20-contract/vectors/mode-switch.txt` at commit
  `2d5a5354ea1b3b3005a13e4594d753a1896006e5`.
- **Licence:** MIT OR Apache-2.0, at the user's option. That is the x80 repository's
  licence (its workspace declares `license = "MIT OR Apache-2.0"`); no separate
  licence covers the vectors' directory.
- **Copyright:** Copyright (c) 2026 Mikołaj Mikołajczyk, the same holder as x86jit.
- **Compatibility:** x86jit has the same licence and the same holder, so the copy is
  compatible with this repository's licence.

The vectors are an interface contract between the two engines (the 8080-mode exit and
hand-back), not a hardware authority: the V20 behaviour they exercise is cited to the
NEC manual (§1). A comment next to the embedded text in the test file repeats the
copyright line. The MIT licence asks for the copyright and permission notice to
accompany a copy; it is:

```text
Copyright (c) 2026 Mikołaj Mikołajczyk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALING IN THE
SOFTWARE.
```

## 6. A clean source can still be second-hand

`oracles` documents a case where one file in an otherwise-clean upstream tree cited
another emulator as the origin of its own constants; its fetcher excises that block and
re-checks it on every verify. The general lesson holds here too: **"it came from the
stash" is not by itself proof that a fact is first-hand.** Read
`oracles/MANIFEST.md` § *Second-hand blocks in a clean source* before leaning on a
citation.

This bit us concretely, and the fix is worth recording. The 80286 corpus fetcher
(`x86jit-tests/vendor/80286/fetch.sh`) used to pull from a moving `main` with no checksum,
and never downloaded `revocation_list.txt` — which lives at the repository root rather than
under `v1_real_mode/` — so every test the corpus author had marked bad was silently trusted,
because the loader reads an absent file as an empty list. The fetcher now pins the same
commit `oracles` records and downloads the revocation list; CI fetches the corpus and sets
`SS286_REQUIRED=1`, so its absence fails instead of skipping.
