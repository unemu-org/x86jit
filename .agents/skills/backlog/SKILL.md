---
name: backlog
description: >-
  Docs and decision records with the Backlog.md CLI (`backlog doc`, `backlog decision`). Use when
  creating or listing a project doc or a decision under `backlog/`, or when reading the Done
  history in `backlog/completed`. Not for open work: x86jit tracks tasks in beads (`bd`) since
  2026-11-04 — use `bd ready`, `bd show`, `bd create` for those.
---

# Backlog.md

> **x86jit: tasks moved to beads (`bd`) on 2026-11-04.** Use this skill only for docs
> (`backlog doc`) and decisions (`backlog decision`). Open work: `bd ready`, `bd show <id>`; see
> `backlog/docs/working-on-tasks.md`. Never create a task with `backlog task create`.

[Backlog.md](https://github.com/MrLesk/Backlog.md) keeps docs and decisions as plain markdown
**inside the repo** (`backlog/`), versioned with the code. The CLI is `backlog`; `backlog <cmd>
--help` gives the exact, version-current flags. In x86jit it no longer tracks tasks: open work is in
beads (`bd`), and `backlog/completed/` + `backlog/archive/` are read-only Done history.

## Layout

```text
backlog/
  config.yml            # project_name, task_prefix, … (task fields are historical)
  tasks/                # empty: open work is in beads
  completed/            # read-only Done history of the old task board
  archive/tasks/        # the 61 task files migrated to beads on 2026-11-04 (source of record)
  docs/                 # guides & reference — backlog doc create → doc-N - Title.md
  decisions/            # architecture/tech records — backlog decision create → decision-N - Title.md
```

## Docs & decisions

```sh
backlog doc create "<title>" [-p <subpath>]   # project documentation → backlog/docs/
backlog doc list
backlog decision create "<title>"             # architecture/tech decision → backlog/decisions/
ls backlog/decisions/                          # list existing (no `decision list` subcommand)
```

Decision vs bead note: cross-cutting / architecture-grade → `backlog decision`; bound to one bead →
`bd update <id> --append-notes "Decided: …"`. See `backlog/docs/decisions.md`.

## Reading the Done history

An old id (`TASK-NNN`) that has no bead (`bd list --external-ref backlog:TASK-NNN` is empty) was
Done before the migration: its file is `backlog/completed/task-NNN - *.md` (or
`backlog/archive/`).
