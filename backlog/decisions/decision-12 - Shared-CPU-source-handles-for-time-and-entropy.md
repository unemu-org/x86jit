---
id: decision-12
title: Shared CPU source handles for time and entropy
date: '2026-11-04 19:47'
status: accepted
---
## Context

Decision-11 resolves CPU identity through immutable CPUID data. Time and entropy
also belong to the embedder, but a lookup table cannot supply an advancing host
clock or fresh random samples. The existing constant RDTSC is useful for replay
and differential tests, while calibration loops need an advancing counter.

## Decision

Install an optional `Arc<dyn CpuSource>` with `Vm::set_cpu_source` for newly
created vCPUs. Each CPU state retains the handle and an engine-allocated
`source_id`, independent of guest CPUID IDs. The source receives that identity
on each request. An embedder may replace either field between runs.

`read_tsc` returns one `TscSample { counter, aux }`, preserving the paired
observation required by SDM Vol 3B §20.17.2. RDTSC and RDTSCP use the same source;
only RDTSCP writes AUX. `read_random` receives RDRAND versus RDSEED and the
destination width, returning `Some(value)` or `None` for unavailable data.
Both backends call the same execution helpers exactly once per executed
instruction, including reads whose result is unused. A failed random request
does not retry internally or fall back to deterministic data.

Without a source, TSC remains `0x12345678`, AUX is zero, and both random
instructions consume one shared per-vCPU deterministic stream with CF=1.
The stream starts at `0x4d595df4d0f33173` and advances by
`state = state * 6364136223846793005 + 1` modulo 2^64 for each instruction;
the destination takes its low 16/32/64 bits. This is reproducible test data,
not cryptographic entropy. Cloning CPU state copies the fallback stream.
Cloning source-backed CPU state preserves its source identity and shares the
handle without snapshotting external state. VM forks share the source and ID
allocator, so fresh vCPUs receive distinct identities across forks.

This is a narrow exception to spec §1's exit-only boundary. A callable source is
justified for values that depend on the moment of execution, including host
clocks, external entropy and replay streams. It receives no guest memory or
mutable CPU-state access. Methods must not panic or reenter execution because
the JIT invokes them through an extern-C helper. This does not introduce
general instruction hooks or change CPUID's data-only contract.

RDTSC/RDTSCP write zero-extended EDX:EAX, RDTSCP also writes zero-extended ECX,
and flags are preserved (SDM Vol 2B RDTSC/RDTSCP, Description/Operation/Flags
Affected). Before calling the TSC method for RDTSCP, the engine guarantees that
all earlier guest instructions have retired and their memory accesses are ordered
by a full host fence. The source owns the ordering of the counter read itself:
a hardware-counter source must use execution-synchronising reads so that sampling
cannot run ahead of earlier guest loads (SDM Vol 2B RDTSCP, Description). On
x86-64, use RDTSCP or LFENCE;RDTSC (SDM Vol 2B RDTSC/RDTSCP, Description). On
AArch64, use ISB before CNTVCT_EL0, or the self-synchronising CNTVCTSS_EL0 with
FEAT_ECV (see the Arm Architecture Reference Manual, generic timer). A memory
fence alone does not synchronise counter execution. RDTSC has no ordering
requirement and adds no fence (SDM Vol 2B RDTSC, Description). Both instructions
call the same source method, so hardware-counter sources must provide RDTSCP
ordering on every call.

RDRAND/RDSEED return zero on failure, set CF from availability and clear
OF/SF/ZF/AF/PF (Vol 2B RDRAND/RDSEED, Operation/Flags Affected). Their IR is a
full-flag helper barrier; prior flag producers stay materialized and the helper's
writes supersede cold fault-recovery recipes.

The source owns TSC scale, offset, cross-vCPU monotonicity and entropy quality.
The embedder must keep invariant-TSC and CPUID leaves 0x15/0x16 consistent with
its source (SDM Vol 3B §20.17/§20.17.1, Vol 1 §21.3 CPUID.15H/16H).
Installation does not advertise features. RDTSCP/RDRAND/RDSEED remain selectable
features absent from the frozen default preset. This adds no CR4.TSD policy or
runtime CPUID-based instruction gating.

## Alternatives considered

- A data-only counter/step description can provide synthetic time but cannot
  express host-time sampling and externally supplied entropy without additional
  update protocols. It would constrain the embedder's clock and replay design.
- An exit per read retains an entirely return-based boundary but leaves compiled
  code for every timestamp in a spin or calibration loop.
- Calling host instructions directly would couple the guest clock, entropy and
  replay behavior to the host and would not supply per-vCPU AUX consistently.

## Consequences

The core gains no dependencies. Dynamic reads incur a shared helper call and,
when installed, source dispatch. Sources may block, so embedders control their
latency. Handles are runtime state and cannot be compared, serialized or
snapshotted by the core; source-specific replay/snapshot protocols remain the
embedder's responsibility. Generated code carries no source addresses or values.

## Trigger to revisit

Revisit when a measured workload needs a concrete data-backed fast clock, or
when VM snapshot/replay gains a source-state capture contract. This decision is
accepted and reflected in spec §1/§14.
