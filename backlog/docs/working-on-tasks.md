---
id: doc-10
title: 'Working on tasks'
type: guide
created_date: '2026-07-06 11:25'
---

# Working on tasks

Open work is tracked in **beads** (`bd`, a Dolt-backed issue tracker; issue prefix `xj`). Knowledge
— design, decisions, guides — stays as markdown under `backlog/`; a bead links to a doc section and
never copies it. Roles, review and the Sol (Codex) workers are in
[`agent-workflow.md`](agent-workflow.md); this page is the day-to-day reference. `bd <cmd> --help`
gives the flags. `bd prime` prints the tool's generic workflow; where it differs from these docs
(who claims and closes, `bd remember`, pushing), these docs win.

## Statuses

| Status | Meaning |
|--------|---------|
| `open` | Filed, scoped, not started. Default for every new bead. |
| `in_progress` | Claimed and actively worked. Claim it **before** you start writing code. |
| `closed` | Landed on `main`. The close reason names the commit and the doc or spec section changed. |

`bd` derives "blocked" from dependencies: a bead with an open blocker does not show in `bd ready`.

```sh
bd ready --exclude-type epic               # unblocked work, highest priority first (plain `bd ready` lists the epics too)
bd list --status in_progress               # what is claimed right now
bd show <id>                               # one bead with its dependencies and parent
bd list --parent <epic-id>                 # the children of one epic
bd list --external-ref backlog:TASK-363    # the bead an old Backlog.md id became
```

Priorities are `P0`…`P4` (`P0` highest). The import mapped Backlog.md high/medium/low to
`P1`/`P2`/`P3`.

## Epics

Each theme is an epic (`bd list --type epic`); every bead belongs to one, through `--parent`. The
epic carries the theme's label; children keep their own labels (create them with
`--no-inherit-labels`). A dotted id (`xj-i2n.4`) shows the parent a bead was created under; after a
reparent `bd show` is authoritative.

| Epic key | Theme |
|----------|-------|
| `m6-x87` | x87 fidelity — exceptions, rounding control, environment |
| `m8-simd` | SIMD correctness — EVEX masking and broadcast, ZMM upper bits, vector MMIO |
| `code-review` | defects from the module-by-module review |
| `fault-precision` | exact guest state at every fault |
| `isa-identity` | the guest CPU the engine claims to be |
| `fp-fidelity` | MXCSR and SSE/AVX floating-point semantics |
| `memory-order` | x86 TSO on every host |
| `smc` | self-modifying code and write barriers |
| `aot` | ahead-of-time translation units |
| `embedding` | what an embedder needs from the JIT |
| `tests-ci` | test infrastructure and CI lanes |
| `guest-perf` | performance measured on real guest workloads |
| `toolchain` | compiler and toolchain defects |

The epic keys `m6-x87` and `m8-simd` are labels inherited from the old board. `m8-simd` matches
`spec.md` §12 M8+ (SIMD); `m6-x87` does not — spec M6 is SMC invalidation. The delivered
milestones `m0-skeleton` … `m8-simd` are Done history under `backlog/completed/`.

## Bead lifecycle

```sh
bd create --title "<title>" --description "<why and what>" --acceptance "<criteria>" \
          --type task --priority P2 --parent <epic-id> --no-inherit-labels --labels <theme labels>
bd update <id> --claim                              # before writing code
bd update <id> --append-notes "MEASURED: <command> → <result>"
bd close <id> --reason "<commit> — <what landed; doc/spec section changed>"   # only once it is on main
```

- **Never `--notes` on an existing bead**: it replaces the whole notes field. Use `--append-notes`.
- Label evidence: **MEASURED** (command run, give it), **READ** (`file:line`), **ASSUMED** (not
  checked yet).
- Work found along the way is a new bead with `--deps discovered-from:<id>`, not folded into the
  current one.
- Do not close until `main` actually contains the work — an early close misleads `bd ready`.

## Definition of Done

A bead closes only when all of these hold on `main`:

1. `cargo nextest run --features unicorn -E 'not binary(fuzz_robustness)'` is green.
2. `cargo clippy --all-targets --all-features -- -D warnings` is clean.
3. `cargo fmt --check` is clean (the nix-pinned rustfmt).

Plus, when the change warrants it: `cargo deny --all-features check licenses bans sources`, the
real-program ladder ([`commands.md`](commands.md)), and a hot-path measurement against
`bench/baseline.json` for codegen changes.

## Git flow

One bead, one branch (`<type>/<bead-or-topic>`, Conventional Branch), one worktree outside the main
checkout. The supervisor squash-merges each bead into `main` as one commit: **Conventional Commits**
(`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, `test:`, `ci:`, `build:`), **GPG-signed** (the
pre-commit `gpg-uid-guard` refuses a sign-key/email mismatch). Commits and pushes happen only at the
maintainer's request. Never bypass hooks. Details: [`agent-workflow.md`](agent-workflow.md) § From
bead to `main`.

`origin` is the maintainer's private Forgejo. The issue database lives in `.beads/` (gitignored);
`bd dolt push` stores it in `refs/dolt/data` on `origin` (plus bd's `__dolt_remote_info__`
branch) and it goes nowhere else. Push it after a batch of bead changes, with the maintainer's
agreement like any push.

## Decision capture

- Bead-bound decision → on the bead: `bd update <id> --append-notes "Decided: X over Y — <reason>."`
- Cross-cutting / architecture-grade → `backlog decision create "<summary>"` (decisions stay
  markdown under `backlog/decisions/`). See [`decisions.md`](decisions.md).

## Session handoff

```sh
bd update <id> --append-notes "Session pause $(date -I). Done: <X>. Next: <Y>. Blocker: <Z|none>."
```

The next session runs `bd list --status in_progress`, then `bd show <id>`, and picks up without
rediscovering state from the diff.

## History

Backlog.md was the tracker until 2026-11-04. Its Done history stays read-only under
`backlog/completed/` and `backlog/archive/`. Every migrated bead carries its old id as
`external_ref` (`backlog:TASK-NNN`) and its old file name and dates in `metadata`. Text that
mentions `TASK-NNN` resolves through `bd list --external-ref backlog:TASK-NNN` (exact, upper case;
`--external-contains task-nnn` ignores case). No bead means the task was Done before the migration:
read `backlog/completed/task-NNN - *.md`.
