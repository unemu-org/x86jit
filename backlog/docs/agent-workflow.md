---
id: doc-29
title: 'Agent workflow — supervisor, workers, beads, review'
type: guide
created_date: '2026-11-04 14:30'
---

# Agent workflow

How the maintainer works with coding agents on this repository. It describes roles, how work is
tracked, how a change goes from a task to `main`, and how the Codex workers (Sol) are driven. The
same method runs on the maintainer's other projects; this page is the x86jit copy.

The repository's remote is the maintainer's private Forgejo; it becomes public only when the
maintainer publishes it, and publishing is their step alone. Write as if it were public already:
everything an agent writes here — code, comments, docs, task and bead text, commit messages — must
be safe to publish and guest-agnostic (README "What this is",
[`PROVENANCE.md`](../../PROVENANCE.md)). Never name a proprietary guest, a private host, a private
path or anyone's personal setup. Never push to any remote other than the configured `origin`, and
never create, mirror or publish a repository.

## Migration from Backlog.md (2026-11-04)

Backlog.md was the tracker until 2026-11-04; beads (`bd`, prefix `xj`) is the tracker now. What
the migration did, so a later session can check it:

- Every open task (61: 60 `To Do`, 1 `In Progress`) became one bead. Title, Description,
  Acceptance Criteria (numbering and checked state kept) and Implementation Notes were copied
  verbatim (one title's YAML `''` escape was fixed afterwards); labels were kept; priority high/medium/low became `P1`/`P2`/`P3` (unset became `P2`).
  The old id is the bead's `external_ref` (`backlog:TASK-NNN`); the old file name, dates and
  ordinal are in its `metadata`.
- The project used labels, not milestones, so the epics follow the thematic labels: thirteen
  epics, each open bead a child of exactly one. The nine `code-review` findings carried no theme
  label; after review, five of them moved to the theme epic they belong to (fault-precision,
  fp-fidelity, smc). The list is in [`working-on-tasks.md`](working-on-tasks.md) § Epics. The one milestone field still set
  held only Done performance work; that theme's epic is `guest-perf`.
- The 24 `dependencies:` entries became `blocks` dependencies. `bd ready --exclude-type epic`
  showed the same 43 tasks the old board called unblocked. Three orderings written only in task prose
  were added after review, at the maintainer's request: `xj-obx.4` waits for `xj-i2n.7` and
  `xj-i2n.8`, `xj-pi7.3` for `xj-562.1`.
- The per-task Definition of Done was identical on every task; it moved to
  [`working-on-tasks.md`](working-on-tasks.md) § Definition of Done.
- Done history stays read-only markdown in `backlog/completed` and `backlog/archive`. The ten
  Done tasks that still sat in `backlog/tasks` were moved to `backlog/completed`, not imported.

The bead database was pushed to the private Forgejo `origin` on 2026-11-04 and restores from a
fresh clone with `bd bootstrap`. Still open: once the maintainer has checked the migration, the
migrated files in `backlog/tasks` move to `backlog/archive` in one commit. `bd dolt push` goes
to `origin` only, never to a public mirror. When the repository is published, the beads data
goes with it, so bead text follows the same publication rules as code.

## Roles

- **Maintainer** — decides scope, priorities and anything outward-facing (pushes, publishing,
  upstream reports). Talks to the supervisor in Polish; everything in the repository is English.
- **Supervisor** — the agent session the maintainer talks to. Turns goals into narrow beads, writes
  worker prompts, reviews every result, runs the gates, and is the only one that commits, merges and
  closes beads. It keeps going on its own from one bead to the next and asks the maintainer only for
  decisions, approvals and anything that needs their hands.
- **Workers** — Codex Sol sessions (default) and Claude agents. A worker implements one bead in
  its own worktree, or reviews one change. Workers never commit, push, merge, close beads or edit
  the design spec; they report back to the supervisor.

## Beads conventions

- `bd` for all task tracking. No markdown TODO lists, no agent-internal todo tools.
- Pick work with `bd ready`; claim it with `bd update <id> --claim` before writing code.
- Record findings with `bd update <id> --append-notes "…"`. **Never `--notes`** on an existing
  bead: it replaces the whole notes field.
- Label evidence in notes, reviews and task text: **MEASURED** (command run, give the command or
  log), **READ** (source read, give `file:line`), **ASSUMED** (not checked yet).
- Work found along the way becomes a new bead with `--deps discovered-from:<id>`; it is not folded
  into the current one.
- A shortcut taken to make progress gets its own "proper path" bead, so the shortcut is not
  forgotten.
- Close with a reason that names what landed: the commit, and the doc or spec section changed.
- **Spikes** (`--type spike`, title prefixed with the spike key) answer a question before code
  depends on it. The description holds: question, what is known, hypothesis, method, time box,
  done-when, what happens if the hypothesis fails, what it feeds. A spike closes only after the doc
  or spec it feeds has changed. A failed hypothesis files its plan-B bead with
  `--deps discovered-from:<spike>`. Past the time box: write down what is known and ask the
  maintainer whether to extend, take plan B, or accept the risk.
- `bd remember` only for facts every contributor needs (tooling traps, repository conventions) that
  are not already in a doc. Personal setup never goes there.

## From bead to `main`

1. **One bead, one branch, one worktree.** Branch `<type>/<bead-or-topic>` (Conventional Branch);
   worktree outside the main checkout. Independent beads may be written in parallel in separate
   worktrees.
2. **Heavy jobs run one at a time per machine** — the full nextest run, the fuzz target, benches,
   the real-program ladder. Parallel writing, serial testing.
3. **The worker implements** with tests that fail before and pass after (on both backends, and on
   an aarch64 host or cross-check where the change touches codegen or memory ordering), runs the
   fast loop, and reports files, tests and results.
4. **A fresh session reviews.** The implementing session never reviews its own change. Small
   changes: one fresh Sol session, read-only. Larger ones: a multi-agent review — independent
   reviewers per dimension (correctness, provenance, tests, performance), then adversarial
   verification of every finding (verifiers try to refute it; unsure means refuted). Confirmed
   findings go back to a worker; the fix is reviewed again.
5. **The supervisor runs the gate** (the Definition of Done in
   [`working-on-tasks.md`](working-on-tasks.md): nextest with `--features unicorn` minus `fuzz_robustness`, clippy
   `-D warnings`, nix-pinned `cargo fmt --check`; plus `cargo deny` and the ladder when the change
   warrants it) and measures hot-path cost against `bench/baseline.json` when codegen changes.
6. **Squash merge** by the supervisor: one commit per bead on `main`, Conventional Commits, GPG
   signed, with a message that describes the whole change. Commits and pushes need the
   maintainer's request (AGENTS.md hard rules); the maintainer may grant it for a session.
7. **Never bypass hooks** (`--no-verify`). The pre-commit hooks need the nix devShell; if a hook
   cannot run, stop and say so.
8. Close the bead with the commit in the reason; file follow-ups as new beads.

Correctness comes before speed: a full review before every merge. During performance work, a fast
measurement loop per step is fine; the full gate runs at merge.

## Driving Sol (Codex) workers

Sol is OpenAI's `gpt-6.1-sol` model run through the Codex CLI, driven from Claude Code with the
Codex plugin's companion script. It performs about like the strongest Claude model and is the
default worker for implementation and review.

```sh
C=~/.claude/plugins/cache/openai-codex/codex/<version>/scripts/codex-companion.mjs
cd <task worktree>
node $C task --model gpt-6.1-sol --effort high --write --background "<prompt>"   # implementation
node $C task --model gpt-6.1-sol --effort high --background "<prompt>"           # read-only review
node $C status <job>      # phase, progress, log path
node $C result <job>      # the worker's final report
```

- **Watch every job.** A Codex job sends no harness notification. Right after starting one, start a
  background loop that polls `status <job>` every 30 s and exits on the job's terminal state. Match
  the job line itself (`^- <job> \| (completed|failed|cancelled|error)`): progress lines also
  contain the word "completed" and end a naive loop at once. Act on the result immediately.
- **Narrow prompts.** Name the worktree and branch, the exact finding or goal with `file:line`, the
  files the worker may change (and that everything else is read-only), the tests to add, the
  commands to run, and the report format. Restate the rules that matter for the change: no commit,
  no push, no merge; provenance (cite the SDM/APM by volume and section, pin with a witness test,
  never take a semantic from another emulator); `x86jit-core` depends on `iced-x86` only; no
  guest-specific code in the core.
- **The sandbox** cannot write `.git` and cannot take the beads lock: the supervisor commits and
  updates beads. If `sccache` fails inside the sandbox, run cargo with `RUSTC_WRAPPER=`.
- **One implements, another reviews.** The review is a separate, fresh Sol session (or a Claude
  agent), never the implementing one.
- **Escalate a wrong approach.** If a worker keeps patching around a problem instead of solving it,
  spell out the intended design in the next prompt. If that still fails, give the bead to a Claude
  worker.
- **Fallback model.** If Sol fails or crashes on a task, rerun the same task with
  `--model gpt-6-astra`.
- **Usage window.** Sol has a rolling five-hour limit. When a job fails on it, read the reset time
  from the error, schedule a one-shot wake-up for reset + 5 minutes, and resume with
  `task --resume-last` (or rerun). Do not wait idle meanwhile: run implementation and review with
  Claude agents until the window reopens.
- **Check the diff scope** after every Sol job (`git diff --stat`): a worker that touched files
  outside its brief is a finding, not a convenience.

## Claude agents and workflows

- Use a Claude agent when the work needs several tools across machines, long investigation, or
  when Sol is unavailable. Brief it like a peer: goal, facts already measured, hypotheses in order,
  what it may touch, and what to return. It sees only the prompt.
- Multi-agent workflows are for reviews, audits and broad research: fan out by dimension, verify
  adversarially, synthesise. Keep each workflow to one well-scoped phase and read its result
  before starting the next.
- A delegated check is not repeated by the supervisor; a delegated result is not trusted blindly —
  the supervisor re-reads the cited code before acting on a surprising claim.

## Reporting to the maintainer

Short status in Polish: what landed (commit, bead), what is running, what failed with the shortest
decisive error line, and the one decision the maintainer needs to make, if any. Every number comes
from a measurement or is labelled as an estimate.
