---
id: decision-26
title: GitHub as the primary forge
date: '2026-11-16'
status: accepted
---

Decider: maintainer (2026-11-16). It completes the republish recorded in
[doc-28](../docs/doc-28%20-%20Republish-handoff-%E2%80%94-state-known-defects-and-what-must-be-fixed-before-the-repo-is-created.md)
and follows the setup the sibling project x80 adopted the same day
(decision-22 (x80)).

## Context

Until 2026-11-16 x86jit was developed in a private repository on the
maintainer's Forgejo instance. That repository was `origin` for code and for the
beads database (`refs/dolt/data`), and it ran CI: a manual Forgejo workflow with
one x86-64 job, whose last step posted the verdict as a commit status
(`scripts/ci-report-status.sh`). A GitHub Actions workflow with x86-64 and
aarch64 Linux lanes was ready for the public repository, but there was no
repository for it to run in.

doc-28 planned the publication under the `unemu-org` organisation on GitHub as a
single orphan commit. On 2026-11-16 the maintainer changed the plan to keep the
history.

Keeping a private Forgejo origin beside a public GitHub copy would mean two CI
systems with different lanes, a mirror credential, and a beads database apart
from the code, for a project whose text is written to be publishable anyway
(AGENTS.md hard rules). x80 ran that arrangement on 2026-11-16 and left it.

## Decision

1. **The publication keeps the history.** On 2026-11-16 the existing history
   was published at <https://github.com/unemu-org/x86jit> with only the
   third-party binaries filtered out of it, and every commit was re-signed.
   Re-signing gives every commit a new id. The private pre-publication history
   is kept under the tag `archive/pre-republish` on the Forgejo backup, so the
   old commit ids, including consumers' pins, still resolve there.
2. **GitHub is the primary forge** for code, CI, releases and planning:
   <https://github.com/unemu-org/x86jit> is `origin`. Landings push `main`
   there, and releases are tags pushed there. The first release is git-only;
   nothing is published to crates.io yet, so there is no release workflow.
3. **The beads database is public in the same repository.** `bd dolt push`
   writes `refs/dolt/data` and bd's branch `__dolt_remote_info__` on `origin`; a
   fresh clone recovers the database with `bd bootstrap`. Bead text stays
   English, publish-safe and guest-agnostic, as before.
4. **Pull requests are disabled** on GitHub while the project matures. Issues
   are welcome, and development happens in the open there.
5. **CI runs only on GitHub Actions** (`.github/workflows/ci.yml`: x86-64 and
   aarch64 Linux lanes, gates in `scripts/ci-gate.sh`). The Forgejo workflow and
   `scripts/ci-report-status.sh` are removed; GitHub reports the job checks
   itself.
6. **Forgejo keeps a backup.** The Forgejo copy becomes a read-only pull mirror
   of the GitHub repository, beads data included, and the former development
   repository is archived. Nothing is pushed to Forgejo any more.

## Alternatives

- **Forgejo primary with a push mirror to GitHub** (decision-21 (x80), run
  there on 2026-11-16). Rejected: two CI systems and a mirror credential,
  for a copy that is public anyway.
- **A single orphan commit** (doc-28's earlier plan). Replaced on 2026-11-16 by
  the maintainer's choice to keep the history with the third-party binaries
  filtered out.
- **GitHub primary with beads on a private Forgejo remote.** Rejected: bead
  text is written to be publishable, and one remote keeps `bd bootstrap` simple.
- **Pull requests closed by a bot.** Unnecessary: GitHub can disable pull
  requests for a repository.

## Consequences

- AGENTS.md, `backlog/docs/commands.md` § Git remote and CI,
  `working-on-tasks.md`, `dev-setup.md` and `status.md` describe GitHub as
  `origin` and the Forgejo copy as a backup. CI is dispatched with
  `gh workflow run ci.yml --ref <ref>` and read in GitHub's checks.
- The README says that x86jit is written mostly by LLM coding agents, that
  development happens on GitHub, that pull requests are disabled and that issues
  are welcome.
- Everything pushed to `origin`, including bead history, is public at once.
- The x86-64 Linux lane on GitHub takes over the Forgejo runner's role as the
  CI host where the x86-64 host paths and the native oracle run for real; the
  aarch64 Linux lane runs on GitHub-hosted runners too. macOS stays a local
  check.
- Commit ids quoted in documents written before 2026-11-16 name commits of the
  pre-publication history. They resolve on the Forgejo backup under
  `archive/pre-republish`, not on GitHub.
- Consumers re-pin to the public commits (doc-28 § Consumers and their pins).

## Revisit when

- the project accepts contributions;
- crates.io publishing is set up, which brings a release workflow;
- GitHub changes its handling of pull requests or Actions;
- the beads tool changes how it stores its database in git.
