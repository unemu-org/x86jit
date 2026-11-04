---
id: decision-11
title: CPUID tables and per-vCPU overrides amend decision-9
date: '2026-11-04 18:53'
status: accepted
---

**Deciders:** Mikołaj Mikołajczyk

## Context

[Decision-9](decision-9%20-%20CPU-features-are-embedder-configured-CpuFeatures-not-hardcoded-%E2%80%94-supersedes-the-global-model-of-decision-2-8.md)
lets an embedder choose feature bits, but vendor, family/model, cache, topology,
brand, address widths and frequency information still need whole CPUID replies.
The feature preset cannot describe those fields. Its frozen default also advertises
instruction families the lifter does not fully cover; changing the identity must not
silently change that existing contract.

## Decision

Use a data table plus per-vCPU overrides. `Vm::set_cpuid_table` installs a complete
`CpuidTable` for newly created vCPUs. Each leaf either ignores ECX or indexes a
subleaf map with an explicit invalid-subleaf reply. The table supplies a missing-leaf
reply and an out-of-range policy: a fixed reply or the highest basic leaf while
retaining input ECX. Maxima come from the table's leaf 0 and 0x8000_0000 EAX.
Explicit entries also permit software-defined leaves. The range rule follows
SDM Vol 1 §21.1.3 for the redirect. Retaining ECX is an engine choice:
Vol 2A CPUID, Operation calls outputs for invalid leaves and subleaves
"Reserved". The table owns vendor policy.

`Vm::new_vcpu_with_cpuid_overrides` accepts masks and replacement values keyed by
the resolved leaf/subleaf. A leaf ignoring ECX uses subleaf zero. Lookup redirects
out-of-range queries before applying overrides, so an alias of the highest basic
leaf sees that leaf's logical-processor fields too. This covers initial APIC ID
(leaf 1 EBX[31:24]) and x2APIC ID (0xB/0x1F EDX), per SDM Vol 1 §21.1.4 and
Tables 21-8/31/70. The same data can carry runtime fields (OSXSAVE and leaf 0xD
sizes, 07H ECX[4] OSPKE and 19H EBX[0] AESKLE; SDM Vol 1 §21.1.5,
Table 21-1), together with an optional XCR0 value used by XGETBV(0).
The embedder also maintains 80000001H EDX[11] SYSCALL when modeling mode
transitions (Table 21-90). The legacy projection sets it in every mode as a
frozen deviation: Intel reports zero outside 64-bit mode.
The embedder maintains their consistency and updates them through `CpuState` when
its modeled runtime state changes; this adds no CR4/XSETBV emulation.

The table is immutable and shared by `Arc`; CPU-state cloning preserves table and
overrides, and `fork_with_backend` inherits the table. The data derives equality
and hashing and has public fields for embedder serialization; there is no new
core dependency or serialization format. Both backends use `CpuState::cpuid`
through the existing shared helper. The installed table is the CPUID source of
truth; missing entries never fall back to preset feature bits. Presets still
supply XCR0 unless a per-vCPU override replaces it.

With no table, retain decision-9's frozen feature projection, including CX8,
and correct leaf 7's invalid-subleaf reply to zero (SDM Vol 1 §21.3,
CPUID.07H and Table 21-17). Its zero reply outside the enumerated ranges deliberately
deviates from SDM Vol 1 §21.1.3 for compatibility. CX8 becomes removable; SSE4A,
ADX, RDRAND, RDSEED and RDTSCP become selectable bits without entering the default.

Resolve **spec §14's hooks question for CPUID** with data: no CPUID callback and
no CPUID exit. The core remains return-based for existing exits. This decision
does not decide future debugging hooks unrelated to CPU identification.

## What this amends in decision-9

The default-only advertise ⊆ lift guard becomes a check of every preset with
explicit per-preset waivers for real lift gaps and unknown probe coverage.
CX8 leaves the always-on scalar baseline and becomes a selectable feature;
the default and preset baselines retain it. Decision-9's trigger to revisit
per-preset waivers has fired. Its frozen default identity remains the compatibility
contract, apart from the corrected invalid leaf-7 subleaf reply.

## Alternatives considered

- A callback is flexible but opaque to equality, serialization, lockstep, VM forks
  and future AOT keys. It would introduce the core's first embedder hook for a
  problem representable as data.
- A CPUID exit would make every query require an embedder round trip and would
  move a CPU-description lookup out of the shared execution path.
- A plain leaf map cannot specify ignored ECX, invalid-subleaf policy, or fields
  that differ between logical processors without duplicating the CPU description.

## Consequences

An embedder can present an entire CPU identity and validate it with
`x86jit_tests::compat::cpuid_lift_gaps`, which accepts any CPUID query function.
That test utility reads instruction-feature bits from the cited Intel/AMD tables
and probes iced's per-Code feature requirements directly, counting a Code only
when all requirements with known CPUID bits are advertised (with iced's
HLE_or_RTM and SKINIT_or_SVM alternatives), excluding forms not
available at CPL 3. It checks memory alternatives too and reports unencodable
probes as unknown coverage. Old vendor-specific leaves outside those manuals are
outside the checker. It validates lift coverage, not instruction correctness or
internal consistency of the CPU description.

All four presets (default/v2/v3/v4) now require reasoned per-preset waivers for
partial or unknown coverage. Existing advertisements are preserved, including
CX8, XSAVE and v2-and-up CX16 gaps. In particular, exposing real leaf 0xD sizes
can move software onto unlifted XSAVE/XRSTOR instructions; installing a table does
not implement those instructions. Richer presets remain explicit caller choices.

## Trigger to revisit

Revisit when new instruction lifts close waivers, when a CPU field needs dynamic
behavior that cannot be represented by embedder-updated data, or when an AOT cache
introduces a concrete CPU-description serialization/key contract.
