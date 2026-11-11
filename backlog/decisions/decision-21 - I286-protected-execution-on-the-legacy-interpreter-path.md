---
id: decision-21
title: I286 protected execution on the legacy interpreter path
date: '2026-11-11'
status: accepted
---

## Context

Decision-18 reserves `CpuGeneration::I286` but only activates the I386 system
path. An embedder needs ordinary real entry, LMSW protected entry, same-CPL
16-bit protected execution, and an externally requested reset back to real mode.
This proposal extends that scope without changing the application path or ABI.

## Proposed I286 amendment to decision-18

Activate I286 on `ExecutionProfile::LegacySystem`, interpreter only. Continue
refusing paging, nonzero CPL, LDT selectors, gates/tasks, and protected event
delivery under the existing typed unsupported-system contract. Same-CPL segment
loads and far JMP/CALL/RET/IRET use the existing access and fault boundaries.
Faults return outward; they do not enter a protected handler or emulate a device.

Use `legacy.cr0` as the backing MSW controls: PE/MP/EM/TS in bits 3:0. LMSW
changes only these controls and cannot clear PE. SMSW supplies a word image with
reserved bits 15:4 set, following the reset image; this is the engine's fixed
reserved-bit policy. CLTS clears TS. MOV CR/DR/TR and other 386-or-later forms
raise #UD before operand accesses. Decode remains 16-bit; 64h–67h and FS/GS
operands cannot grant 386 addressing, registers or sizes. The 286 instruction
length limit is ten bytes (#GP on overflow). Existing supported 286 forms retain
the current scalar lowering; this is not complete ISA coverage.
Available 8087/287 coprocessor forms retain the existing typed unsupported exit.

LGDT/LIDT consume six-byte images with 16-bit limits and 24-bit bases, ignoring
the sixth byte. SGDT/SIDT write six bytes and fill the sixth with `ff`. Decode
segment descriptors using only bytes 0–5: a 16-bit inclusive limit, 24-bit base,
and access byte. Reserved bytes 6–7 supply no 386 extension fields; ignoring
their contents is a deterministic engine policy, not a conformance claim for
software supplying nonzero reserved bytes. Hidden cache injection rejects
limits/bases/attributes outside the 286 format and 32-bit code/stack sizes.
Protected FLAGS includes IOPL/NT; real-mode FLAGS keeps the existing 286 mask.
NT restoration does not activate task return.

Reset remains two embedder actions: call `Vcpu::reset_execution_state`, then
install the register/cache image. This API does not synthesize hardware reset
registers. I286 accepts reset-origin execution through the supplied hidden CS
base, including `ff0000:fff0`; a real far reload replaces the base and clears
the reset-origin marker. Memory, configuration and accounting follow the existing
reset API. Reused IR is revalidated under the complete current context as in
decision-20; no stale protected cache is rebuilt from visible selectors.

## Primary-source evidence

Intel **80286 and 80287 Programmer's Reference Manual**, order **210498-005**,
1987, `oracles/intel/i286-prm.pdf`; cited values checked against page images:

| Fact | PRM section/page |
|---|---|
| Sixteen-bit effective offsets and selectors | §6.2 p. 6-2 |
| Descriptor byte positions; reserved upper word | §6.3 Fig. 6-3 p. 6-5 |
| Accessed bit; segment type/privilege/limit checks | §7.4 pp. 7-11–13; Appendix B MOV pp. B-73–74 |
| Same-level transfers and word frames | §7.5 p. 7-15; Appendix B CALL p. B-23, IRET pp. B-51–52, RET pp. B-94–95 |
| #UD and restart at leading prefixes | §9.6.1 p. 9-10; Appendix B #UD p. B-12, MOV p. B-73 |
| Ten-byte instruction limit | Appendix B #GP p. B-9 |
| Available coprocessor control forms (not activated) | Part II Ch. 2, Processor Control Instructions, p. 2-16 |
| Protected IOPL/NT | §10.1 p. 10-1; Appendix B IRET pp. B-51–52 |
| Table-register image | §10.2 Fig. 10-3 p. 10-3; Appendix B LGDT/LIDT p. B-65, SGDT/SIDT p. B-101 |
| MSW controls, sticky PE, CLTS | §10.2.2 Table 10-1 p. 10-4; Appendix B LMSW p. B-67, SMSW p. B-103 |
| Reset MSW, hidden CS, IP, IVT and real regime | §10.4 p. 10-6 |

The PRM calls SGDT/SIDT's sixth byte undefined. **Intel SDM 325462-092US,
Vol 2B §4.3 SGDT and SIDT, IA-32 Architecture Compatibility** explicitly
specifies ones for the 286, supplying that value. Instruction-generation tags
from iced route the manual's encodings; they are not semantic authority.

## Consequences

`CpuState` and `jit_abi` retain their layouts. Application lowering and code
generation are unchanged. The public reset API remains a transient-state reset,
with model register initialization owned by the embedder. Hardware reset does
not imply that the reset API itself clears architectural PE.

The witnesses in `x86jit-tests/tests/legacy_i286.rs` assert images, widths,
faults and restart state by value, using a backend that panics if invoked.
They require no manuals, network, external firmware or hardware oracle at runtime.
Complete 286 ISA/system emulation, protected event delivery and peripheral reset
policy remain outside this amendment.
