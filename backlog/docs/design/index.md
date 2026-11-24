---
id: doc-11
title: 'Design'
type: guide
created_date: '2026-07-06 11:25'
---

# Design

Authoritative design documents for `x86jit`, in English.

| Document | What |
|----------|------|
| [`spec.md`](spec.md) | The engine spec (v0.3): public contract, guest state, IR, lift, backends, cache, dispatcher, SMC, multithreading, milestones, and the consolidated pitfalls list. Source of truth. |
| [`testing.md`](testing.md) | Testing architecture (v0.1): the oracle problem, test vectors, differential fuzzing, optimization testing axes, and per-milestone test integration. |
| [`fpu-error-routing.md`](fpu-error-routing.md) | Design proposal: external x87 numeric-error routing for 287/387-era Machine contexts and a future 486, with CPU/board boundaries, API choices and witness plan. |
| [`paging.md`](paging.md) | Design proposal: 386/486 Machine two-level paging, precise page faults, A/D and WP, translation invalidation, physical aliases/SMC, bus boundaries and witness plan. |
| [`v86.md`](v86.md) | Design proposal: virtual-8086 execution on the 386 and both Intel486 Machine models, entry/exit frames, privilege and I/O rules, paging, continuations, source conflicts and witness plan. |
| [`doc-24 - Lockstep-interp-vs-hardware-tracer.md`](doc-24%20-%20Lockstep-interp-vs-hardware-tracer.md) | The lockstep tracer (`scripts/lockstep.sh`): finds bugs where interp and JIT agree but both differ from a real CPU. When to use it, how it works, blind spots. |

Section references throughout the wiki and source comments (`§7.1`, `§8.2.1`, …) point at `spec.md`; `T§`-prefixed references point at `testing.md`. The [`../../tasks/`](../../tasks/) backlog is derived from both.
