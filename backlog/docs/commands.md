---
id: doc-2
title: 'Commands'
type: guide
created_date: '2026-07-06 11:25'
---

# Commands

Everyday commands for this project. Keep this file **flat and copy-pasteable** — agents and humans both grep it.

All commands assume you're inside the devShell (`nix develop`, or auto-loaded via direnv). See [`dev-setup.md`](dev-setup.md).

## Build / run / test

```sh
cargo build                      # whole workspace
cargo build -p x86jit-core       # single crate
cargo test                       # all tests (unit + integration)
cargo nextest run                # faster test runner (shipped in the devShell)
cargo run -p <bin>               # once a binary target exists
```

The JIT backend is behind a default feature `jit` on `x86jit-cranelift`:

```sh
cargo build -p x86jit-cranelift                       # with JIT (default)
cargo build -p x86jit-cranelift --no-default-features # core-only, no cranelift pulled
```

The Unicorn differential oracle is behind the `unicorn` feature on `x86jit-tests`
(off by default; links nixpkgs `libunicorn` via pkg-config — needs the devShell):

```sh
cargo nextest run                                     # default: no native deps
nix develop -c cargo nextest run -p x86jit-tests --features unicorn   # interp-vs-Unicorn diff
# capture a snippet into a permanent .ron vector (Unicorn is the oracle):
nix develop -c cargo run -p x86jit-tests --features unicorn --bin capture -- \
  --bytes <hex> --init "rax=0x..,rbx=.." --name <id> --tags <a,b> --out x86jit-tests/vectors/<cat>/
```

⚠️ Don't pass `--all-features` outside the devShell — it enables `unicorn`, whose
sys crate needs `libclang` + `pkg-config unicorn` (provided by `nix develop`).

## The real-program ladder (unemulinux)

The corpus here validates *what is lifted*. What real software trips over is the
ladder — busybox, sqlite, CPython, the Go servers, caddy — and since the split it
runs in `unemulinux`, a separate repository. `scripts/ladder.sh`
drives it from here:

```sh
scripts/ladder.sh                     # smoke subset vs the working tree (~30 s)
scripts/ladder.sh --full              # the whole ladder (~10 min)
scripts/ladder.sh --rev HEAD~5        # a specific revision, in a detached worktree
scripts/ladder.sh --full -- -E 'test(caddy)'   # extra nextest args after --
```

It needs a sibling `../unemulinux` checkout (or `UNEMULINUX_DIR`). Without one it
exits **2**, not 0 — "the ladder did not run" must never look like "the ladder
passed". The `--if-present` form, used by the pre-push hook, downgrades that to a
printed skip.

The smoke subset is one static musl program, one dynamic glibc program, one Go binary
and one busybox applet — four different failure classes. It is **not** the ladder:
sqlite, lua, CPython, djpeg, caddy, the OCI rungs and the threading paths are all
outside it, and the script says so every time it finishes.

`--rev` pins the *recompiler*; the userland is always the working tree next door, so
a harness change and the engine change it exercises can be tested together.

## Bounded code-memory campaigns

Run one binary at a time on shared hosts, with one parent test thread and a
180-second command timeout. Isolated children use two libtest threads, keeping
the active parent and child to at most six threads. Normal stress cases have
fixed iteration limits, an 80-second cooperative deadline and a 90-second
subprocess watchdog, allowing margin for loaded CI containers. Each workload
uses at most four child-process threads, including the idle libtest driver and
backend compiler/maintenance threads. Watchdog parents only poll. No native
platform result is implied by a cross-check. The design doc records measured
and missing coverage.

```sh
# A separate target directory keeps these builds apart from other workspace builds.
export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/x86jit-code-memory-target" RUSTC_WRAPPER= CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=2
timeout 180 nice -n 10 cargo --offline test -p x86jit-cranelift --test code_memory_stress -- --test-threads 1 --nocapture
timeout 180 nice -n 10 cargo --offline test -p x86jit-cranelift --test code_memory_fault_stress -- --test-threads 1 --nocapture
timeout 180 nice -n 10 cargo --offline test -p x86jit-cranelift --lib -- --test-threads 1
timeout 180 nice -n 10 cargo --offline test -p x86jit-cranelift --test code_memory_bound -- --test-threads 1
timeout 180 nice -n 10 cargo --offline test -p x86jit-tests --test memory --features unicorn region_fault_state:: -- --test-threads 1
# Optional packed-storage bound (100,000 units; ignored with the soak):
timeout 180 nice -n 10 cargo --offline test -p x86jit-cranelift --test code_memory_bound hundred_thousand_units_have_bounded_mappings_and_residency -- --ignored --exact --test-threads 1 --nocapture
# Optional longer soak: 2048 cycles, at most four child threads, 120s watchdog.
timeout 180 nice -n 10 cargo --offline test -p x86jit-cranelift --test code_memory_stress dense_reclamation_soak -- --ignored --exact --test-threads 1 --nocapture
```

**CI notes (READ):** the workspace nextest gate discovers the stress binaries on
native Linux x86-64 and arm64 runners. `.config/nextest.toml` puts the stress,
fault-stress and bound binaries plus isolated library campaigns in a group with
`max-threads = 1`. Each selected test reserves `num-test-threads` globally, so
other workspace tests cannot run alongside it. The soak and 100,000-unit bound
remain ignored; the heavy tier runs them (§ Test tiers). The latter took 40.88 seconds alone on macOS arm64 in debug;
normal gates retain the 512-unit bound.
The library alias-relocation campaign is enabled only on Linux/FreeBSD and uses
the real executable aliases, finalization and per-allocation retirement. FreeBSD
cross jobs compile the stress and signal paths but cannot exercise mappings,
backing discard, instruction visibility or signal delivery. There is no native
FreeBSD CI runner. macOS runtime validation remains a separate native gate.

## Release-profile runs

`cargo nextest run` builds the **dev** profile, so every test exercises the engine at
`opt-level 0`. A defect that only appears under optimization is invisible to all of
them — task-223 was exactly that, and its regression test passed under `cargo test`
while failing under `cargo test --release` on the same tree.

```sh
cargo nextest run --release --features unicorn -E 'not binary(fuzz_robustness)'
cargo nextest run --release -p x86jit-tests -E '(binary(codegen) & test(/^jit::/)) or (binary(differential) & test(/^differential::/))'
```

CI runs a narrower form (the second line plus the corpus, smc, mt and host-fault
recovery binaries; see `scripts/ci-gate.sh` and `.github/workflows/ci.yml`); the full release run is a local
check worth doing before a release or after touching the softfloat and codegen paths.

## AVX/VEX differential fuzz (`cargo xfuzz`)

```sh
cargo xfuzz --secs 60                  # register operands (the historical campaign)
cargo xfuzz --mem --secs 60            # the same programs with MEMORY source operands
cargo xfuzz --mem --seed 206           # replay exactly one program
cargo xfuzz --list                     # every op, grouped by family
```

`--mem` (task-325) rewrites each memory-capable VEX op to take its last source from
memory, over a two-page region so an operand can be 32-byte aligned, arbitrarily
unaligned, or straddling a page boundary. It is the only leg that can falsify
memory-source decoding, effective-address computation and load width — the register
campaign counts an op covered on the strength of its register form alone.

Expect `UnknownInstruction` findings from it: those are memory forms the lifter does not
have, the same set the coverage map reports as `reg_only`. A **wrong result** with no
trap is the interesting kind.

## Typecheck / lint / format

```sh
cargo check --workspace --target aarch64-unknown-linux-gnu --tests   # cross-target guard
cargo check --locked --workspace --tests --target aarch64-unknown-freebsd
cargo check --locked --workspace --tests --target x86_64-unknown-freebsd
cargo clippy --all-targets -- -D warnings                 # default features (no native deps)
nix develop -c cargo clippy -p x86jit-tests --features unicorn --all-targets -- -D warnings
cargo fmt --all                  # write
cargo fmt --all -- --check       # verify (CI)
```

## Nix

```sh
nix develop                      # enter devShell (toolchain + nextest)
nix flake check                  # evaluate flake outputs
nix build                        # build the package output
nix flake metadata               # inspect resolved inputs / lock
```

## Pre-commit

```sh
pre-commit install --install-hooks                  # one-time, per clone (pre-commit + pre-push)
pre-commit run --all-files                          # run active hooks
pre-commit run --all-files --hook-stage manual      # include staged-as-manual hooks
```

Stages: **pre-commit** = hygiene + `cargo fmt --check`; **pre-push** = `cargo clippy -D warnings`
+ the **workspace** aarch64 cross-target check + the perf gate.

That cross-check is `--workspace` for a reason worth keeping: it ran `-p x86jit-cranelift`
alone until 2026-08-14, so it never saw `x86jit-tests`, where four files imported the
`cfg(target_arch = "x86_64")` native oracle unconditionally. The first execution of the
aarch64 CI lane failed to compile the crate; the narrow guard had been green throughout.
Anything host-specific in a test file is invisible to a guard scoped to one crate.

## Performance (bench + regression gate)

Always `--release` (debug timings are meaningless). See [`design/`] and `bench/README.md`.

```sh
cargo run -p x86jit-bench --release -- record        # measure HEAD; write bench/history/<sha>.json,
                                                     # bench/baseline.json, backlog/docs/performance.md
cargo run -p x86jit-bench --release -- gate          # compare HEAD vs baseline; exit 1 on >10% regression
cargo run -p x86jit-bench --release -- compare A B    # delta table between two records
```

The **pre-push perf gate** (`scripts/perf-gate.sh`) blocks a push whose interpreter or JIT time
regresses more than `X86JIT_PERF_THRESHOLD` percent (default 10) vs `bench/baseline.json`:

```sh
X86JIT_ALLOW_PERF_REGRESSION=1 git push              # override an intended/accepted regression
cargo run -p x86jit-bench --release -- record        # then accept it as the new baseline + commit
git add bench/baseline.json backlog/docs/performance.md bench/history/
```

The gate skips when there's no baseline (fresh clone) or the host differs (timings aren't comparable
across machines). `performance.md` (Backlog.md doc-21) shows each snapshot's Δ vs the prior baseline.

### Performance campaign (decision-23)

`scripts/perf-campaign.sh` measures the decision-23 performance requirement: host instructions
per guest step of `real16_loops` on all 16 workloads × 9 paths, and the counted store probe,
base against candidate, in two build profiles, and gives the two-profile verdict. Run it in
the dev shell at default QoS; it lowers only its builds.

```sh
nix develop --command scripts/perf-campaign.sh origin/main          # candidate = this working tree
nix develop --command scripts/perf-campaign.sh cf32ec2e f14e9d1c    # two commits
scripts/perf-campaign.sh --profile perf BASE CAND                   # one profile only, for exploration
scripts/perf-campaign.sh --profile release --processes 10 BASE CAND # the shipped profile, 10 processes
scripts/perf-campaign.sh --build-only BASE                          # fill the cache only
python3 scripts/perf_campaign.py analyze <out>/perf/raw             # analyse kept raw data again
python3 scripts/perf_campaign.py combine release=<out>/release/analysis.json perf=<out>/perf/analysis.json
```

What it does:

- **Profiles.** By default (`--profiles release,perf`) both sides are built and measured in the
  default release profile and in the `perf` profile, all builds first, then one profile's
  measuring after the other. `--profile NAME` runs one profile alone; its verdict is that
  profile's own.
- **Builds** each side and profile with `cargo build --locked --profile <profile> -p x86jit-bench
  --example real16_loops --example interpreter_store_probe`, one at a time, under `taskpolicy -c background`
  (`nice` elsewhere), `CARGO_BUILD_JOBS=4`, `RUSTC_WRAPPER=` and `CARGO_INCREMENTAL=0`. A commit
  is built from `git archive`; a working tree from a copy of its tracked and untracked, not
  ignored, files. A clean working tree counts as its HEAD commit.
- **Caches** the binaries of every commit under `<cache>/bin/<commit>-<profile>-<toolchain>`, so
  a `main` commit used as the base is built once. The cache is `--cache DIR`, else
  `$X86JIT_PERF_CACHE`, else `$TMPDIR/x86jit-perf-cache`; it also holds the shared target
  directory (dependencies are reused) and a lock that serialises builds. Working-tree binaries
  are not cached; they are kept in the run directory.
- **The `perf` profile** (workspace `Cargo.toml`) is release with `codegen-units = 1` for
  `x86jit-core` and `x86jit-bench`. A commit from before the profile gets the same profile on the
  cargo command line. Base binaries are cached per profile.
- **The store probe** is the bench example `interpreter_store_probe`, which prints host
  instructions per store on macOS as its sixth column; nothing is patched into the tree. For a
  commit from before that column, the script builds this checkout's copy of the example in its
  place and records that in the binaries' `meta.txt`.
- **Measures** with address-space randomization off for every process: on macOS through
  `posix_spawn` with `_POSIX_SPAWN_DISABLE_ASLR`, on Linux through `setarch -R`. Each process runs
  one workload, three rounds over all paths (`real16_loops DIVISOR 3 WORKLOAD`), and reports
  the median of its rounds; `straight-line` runs seven rounds, because its measured run is short
  (about 90000 steps) and one perturbed round moves it by 1–5 %. Three processes
  per cell and side by default (`--processes`, 3–5), four processes at once (`--parallel`), the two
  sides interleaved and alternating order. The store probe runs its 8 cells (width 8 and 32, single
  and crossing, recovery off and on) with 200000 iterations, the same number of processes.
- **Divisors.** Each workload's step count and warm-up are divided by 8, except `straight-line`,
  which runs at 1: its run grows with the divisor (one warm-up and three measured passes per
  unit). Each divisor was validated against divisor 1 on one binary with
  `python3 scripts/perf_campaign.py validate-divisors --bin DIR --out DIR`, which runs every
  workload at divisor 1 and at each candidate divisor and reports, per workload, the largest
  change of a path's median of processes; a divisor is kept when that change stays within the
  process-to-process spread. The replays in `performance.md` § "Short campaign protocol" check
  the deltas, not only the values, against recorded campaigns at divisor 1.
- **Reports**, per profile in `<out>/<profile>/`: `report.txt` (the per-cell change table, the
  path means, the store probe, the spread within a side, cells with two modes and the profile's
  own verdict), `analysis.json`, `cells.csv` and the raw per-process files in `raw/`; and in
  `<out>/` the two-profile verdict, `combined.txt` and `combined.json`. `--out DIR` defaults to
  `<cache>/runs/<time>-<base>-<cand>`.
  A cell's value is the median over processes of each process's in-run value; the path mean is
  the geometric mean of a path's cells on each side.
- **Gate, per profile:** a cell or store-probe cell is over when its change exceeds +0.5 %
  (`--gate`). The step, retired-instruction and warm-up counts must be equal across all
  processes of both sides. The bimodal rule of decision-23 is the fallback, applied within each
  profile before the combination: a cell in which a side's values still form two modes is
  compared on like modes, and its workload is topped up to seven processes per side.
- **Gate, two profiles:** a cell **blocks** only when it is over in both the release and the
  `perf` profile. A cell over in one profile only is **drift**: the report lists it with the
  profile it is over in and both changes, and it does not block. The store probe follows the same
  rule.
- **Exit status:** 0 pass, 1 block, 2 a usage, build or measuring problem (a failed process,
  unequal counts, a zero counter, a cell missing in one profile), 3 drift only. With
  `--profile NAME`: 0 pass, 1 a cell over, 2 a problem. The instruction counter exists on macOS
  only; on Linux the script runs but reports a problem.

A campaign with cached bases takes two candidate builds and a few minutes of measuring; the
timings are in `performance.md` § "Short campaign protocol".


### Test layout

x86jit-tests builds a few test binaries instead of one per file:
- the area groups (`legacy`, `differential`, `codegen`, `exits`, `memory`, `threads`, `isa`) are
  `tests/<group>/main.rs`, with one module per test file;
- a file that defines a `#[global_allocator]`, and `fuzz_robustness`, stay single-file binaries.

To add a test file to a group, put it in `tests/<group>/` and add its `mod` line to that group's
`main.rs`. Cargo does not find it on its own. A test's nextest name is
`x86jit-tests::<group> <file>::<test>`. If the new file holds a test slower than 5 s, list it in
`.config/slow-tests.txt` and run `python3 -B scripts/test-tiers.py sync`.
## Test tiers

The test suite runs in three tiers, each a nextest profile in `.config/nextest.toml`.
The overrides of the default profile (the code-memory group) apply to all three.

| Tier | Profile | Contents | Runs where |
| ---- | ------- | -------- | ---------- |
| fast | `fast` | every test except `fuzz_robustness` and the slow list | locally: the worker before hand-back and the supervisor before a push (`scripts/test-scope.sh`, the Definition of Done) |
| full | `ci` | every test except `fuzz_robustness`, with a JUnit report; plus the release subset | CI, on every push that runs the gate, both Linux lanes, before `main` |
| heavy | `heavy` | `fuzz_robustness` and the ignored soaks (`dense_reclamation_soak`, the 100,000-unit bound) | off the path to `main`: by hand today, a scheduled workflow later |

```sh
bash scripts/ci-gate.sh fast      # the fast tier: cargo nextest run --profile fast ...
bash scripts/ci-gate.sh tests     # the full tier, as CI runs it
bash scripts/ci-gate.sh heavy     # the heavy tier: --profile heavy --run-ignored all
cargo nextest run --profile fast -p x86jit-tests --test codegen   # the profile takes any selection
```

The **slow list** is `.config/slow-tests.txt`: one test per line, `<binary-id> <test-name>`,
with `only=<lane glob>` for a test that exists only on some lanes (`linux-x86_64`,
`linux-aarch64`, `macos-aarch64`). A test enters it when it takes longer than 5 s on any
lane. The fast profile's filter is generated from it; after editing the list, run
`python3 -B scripts/test-tiers.py sync`, and the gate-script tests fail while the two
differ.

The **ratchet** keeps the fast tier honest in both directions. The CI step *Fast-tier
ratchet* (`scripts/ci-gate.sh ratchet`) reads the full tier's JUnit report and fails when a
test that the fast tier runs took more than 10 s, or when a listed test that applies to the
lane is missing from the report. Tests between 5 s and 10 s are printed as candidates
without failing: the gap absorbs the run-to-run variance of shared runners. The report is
kept in the lane's log artifact as `junit-tests.xml`, the input for the next list update.
A worker can check a local fast run the same way, timing only:

```sh
python3 -B scripts/test-tiers.py ratchet --tier fast target/nextest/fast/junit.xml   # also with CARGO_TARGET_DIR set
```

Every test of the full tier still runs before `main`, in CI on Linux x86-64 and aarch64; the
slow list does not run on the developer's machine for every change. Darwin-specific slow
tests are covered only by a local full run (`bash scripts/ci-gate.sh tests`) until a macOS
lane exists.

## Test scope

`scripts/test-scope.sh` chooses the smallest local command set that covers a branch's
changes. It is meant for workers' inner loops and for the local gate before work is handed
back; both scopes that run tests use the fast tier (§ Test tiers). It does not replace CI:
GitHub runs the full gate, slow tests included, on every push except a docs-only one
(§ Git remote and CI), which `--scope` identifies there.

```sh
bash scripts/test-scope.sh origin/main            # classify the changes, print the commands
bash scripts/test-scope.sh --run origin/main      # run them; keeps going, exits 1 on any failure
bash scripts/test-scope.sh --scope origin/main    # print only the scope (`none` for an empty diff)
bash scripts/test-scope.sh --classify <path>...   # the class of each path
bash scripts/test-scope.sh --plan legacy-only     # one scope's commands, without a diff
```

The changed paths are those that differ between the working tree and the merge base with
the base ref (the commits of `git diff <base>...HEAD` plus staged and unstaged edits),
together with untracked files. A rename counts at its old path and at its new one. Each
path is docs, legacy or shared, and the widest class present selects exactly one scope:

| Scope | Selected when | Runs |
| ----- | ------------- | ---- |
| docs-only | every path is docs: under `backlog/` (except `backlog/docs/compat/`), a `*.md` or `*.txt` outside the crate, `scripts`, `vendor` and dot-config directories, or the `oracles` submodule. An empty diff runs nothing. | `cargo fmt --check`, the guest-agnostic guard and pre-commit on the changed files; no build |
| legacy-only | every path is legacy or docs | fmt, the guard, clippy with `-D warnings`, `test_legacy_scope.py`, in the fast profile every test binary that selects Real16 or a CPU model (all `legacy_*` and `real16_*`, `ss286`, `cf16`, `compat` and others), the shared checks `boundary`, `environment_boundary`, `run_steps`, `irq_shadow`, `retired_count`, `smc`, `cross_region_*` and `application_codegen`, the `x86jit-core` and `x86jit-tests` unit tests, and `cargo check --workspace --tests` for both Linux targets |
| shared | any other path | the fast tier: every `scripts/ci-gate.sh` step that CI runs on either lane, except that `fast` replaces the full tier (`tests`, `ratchet`, `release`); both Linux cross-checks, the FreeBSD checks unless `XJ_DEVSHELL=no-freebsd`, and the gate-script tests |

The legacy paths are `interp/legacy.rs`, `interp/i8086.rs`, `lift/legacy.rs`,
`lift/i8086.rs`, `vm/nec.rs`, `vm/i8086.rs` and `model.rs` under `x86jit-core/src/`, with
their submodules; `ir/verify/legacy_probes.rs`; `x86jit-tests/tests/legacy_*`; the two
support files that only legacy tests include; and `backlog/docs/compat/`, the ISA map that
the `compat` test checks. `x86jit-core/src/legacy.rs` and `vm/legacy.rs` are shared: every
vCPU uses the types the first defines, and the second holds `run_dispatch`, which is also
the Long64 and Compat32 dispatch loop. The call-graph evidence is in the script.

`scripts/tests/test_legacy_scope.py` keeps the legacy-only scope sound, and CI runs it in
the gate-script step. It fails when shared engine code gains a reference into a legacy
module, when `alias_context` (the one legacy function that application dispatch calls)
changes, when a legacy file implements a shared type or contains a doctest, and when a test
that selects Real16 or a CPU model is missing from the legacy-only set. If a new reference
is reached only from Real16 or a Machine profile, record it with its reason in the test's
`REVIEWED_ENTRY_POINTS`; otherwise remove that module from `legacy_paths` in the script.
`scripts/tests/test_test_scope.py` checks the classifier itself on throwaway repositories.

```sh
python3 -B -m unittest discover -s scripts/tests -p test_legacy_scope.py -v
python3 -B -m unittest discover -s scripts/tests -p test_test_scope.py -v
```

## Git remote and CI

`origin` is the public GitHub repository, <https://github.com/unemu-org/x86jit>
([decision-26](../decisions/decision-26%20-%20GitHub-as-the-primary-forge.md)).
Push only there, at the maintainer's request; a push is public at once. A Forgejo
copy pulls a read-only mirror of it as a backup and receives no pushes. CI runs
only on GitHub Actions, and the workflow publishes nothing.

- **GitHub**: `.github/workflows/ci.yml` runs on every push, on pull requests and
  on manual dispatch; pull requests are disabled on the repository, so in practice
  pushes and dispatches start it. It has three jobs.
  - **plan** (`ubuntu-24.04`, about a minute) checks out the full history and runs
    `scripts/ci-plan.py`, then the gate-script tests
    (`python3 -B -m unittest discover -s scripts/tests -p 'test_*.py'`). The plan
    decides:
    - a **push to `main` whose commit already has a run, on any branch, in which both
      test jobs succeeded** (a `push` or `workflow_dispatch` run of `ci.yml`, found
      through the Actions API by commit SHA) skips the test jobs. This is the normal
      case after a `ci/*` branch passed and `main` is fast-forwarded to it;
    - otherwise the changed paths are classified with `scripts/test-scope.sh --scope`
      against the previous tip for a push to `main`, the merge base with `origin/main`
      for any other push, and the base commit of a pull request. **Docs-only**
      (§ Test scope: nothing under the crates, `scripts`, `vendor`, `.github`,
      `.cargo`, `.config`, no Cargo or flake file, no test fixture) runs fmt and the
      guest-agnostic guard in the plan job, without Nix — fmt through the runner's
      rustup at the Rust version `flake.nix` pins (`XJ_DEVSHELL=host`) — and stops;
    - **legacy-only and shared** run the full gate.

    Every doubt runs the full gate: a failed API lookup, an event other than push or
    pull request, a missing base commit, an empty diff, a classifier error, or the
    plan script itself failing.
  - **test (x86_64)** on `ubuntu-24.04` and **test (aarch64)** on `ubuntu-24.04-arm`,
    when the plan asks for the gate. Both install pinned Nix, restore the Nix store,
    and use the devShell from `flake.lock`. Each runs fmt, the guest-agnostic guard,
    clippy with all features and `-D warnings`, cargo-deny, the workspace nextest run
    with Unicorn and without `fuzz_robustness` (the full tier, nextest profile `ci`),
    the fast-tier ratchet on its JUnit report (§ Test tiers), the release subset, workspace doctests
    with all features, and a workspace/test type-check for the opposite Linux
    architecture; the x86-64 job adds separate FreeBSD compile-check steps for both
    architectures. The release subset builds only the eight binaries it runs (`jit`,
    `differential`, `corpus`, `smc`, `mt`, `fault_landing_pad`, `region_fault_state`,
    `rust_memory_faults`). There is no separate Compat32 lane: its 22 tests
    (`differential32` and the two `*_matches_interp_32` fuzz tests) run in the
    workspace nextest step.
    The cross-check uses default features because the cross architecture's
    Unicorn library is unavailable; each native lane tests Unicorn. The x86-64 Linux
    job also exercises the cfg-gated native oracle.
    After the devShell succeeds, later gates run even if an earlier gate fails; the
    test jobs also run when only the gate-script tests failed. Any job's failure fails
    the workflow, without cancelling the other architecture.
  - **warm caches** (both architectures), on a push to `main` whose test jobs did not
    run, so the caches other refs restore stay present. It restores the Nix store and
    the Rust dependency cache; on a dependency-cache miss (a changed `Cargo.lock`,
    toolchain or evicted entry) it builds what the test job builds without running
    any test (`ci-gate.sh clippy`, `build-tests`, `build-release` and the cross
    checks), and saves both.

  GitHub supplies the job checks; this workflow posts no commit statuses of its own.
  Combined gate logs are retained as `ci-logs-x86_64` / `ci-logs-aarch64` (and
  `ci-logs-plan` for the docs-only lane) artifacts for seven days; the test lanes'
  artifacts also hold the full tier's JUnit report, `junit-tests.xml`. Installer and cache
  diagnostics remain in the Actions job log.
  All actions are pinned to commit SHAs, checkout does not persist credentials, and
  the repository permissions are `contents: read` and `actions: read` (the plan's run
  lookup, with the default token). No configured secrets are needed, and every runner
  is GitHub-hosted.
  Caches: the Nix store (`nix-community/cache-nix-action`, keyed by OS, architecture,
  dev shell and the flake files), the Rust dependency cache (`Swatinem/rust-cache`)
  and the pinned 80286 corpus, shared between architectures. The Rust dependency cache
  holds the Cargo registry and git checkouts and the compiled dependencies under
  `target/` of every profile and target one job builds (dev, test, the release subset,
  cross checks); workspace crates are pruned, so the project's own crates always build
  fresh. It has one entry per architecture and dev shell (`shared-key`), keyed further
  by the toolchain's `rustc -vV`, the Cargo manifests and lockfile and the `CARGO*` /
  `RUST*` variables, and the test and warm-caches jobs share it. The former cache of the
  whole `target/` (7.7–8.3 GB per architecture) is gone: restoring it took longer than
  the dependency builds it saved. The steady state is two Nix stores, two dependency
  entries and the corpus, which must stay under GitHub's 10 GB per repository; check
  with `gh api repos/unemu-org/x86jit/actions/cache/usage` and `gh cache list`.
  Only pushes to `main` save caches; every other ref restores `main`'s,
  as GitHub scopes caches, and per-branch copies would evict them. A corpus cache miss
  fetches the corpus, and `SS286_REQUIRED=1` makes an absent corpus fail the test gate.
  The `oracles` submodule is deliberately not checked out: these gates use in-code
  witness values, while `vendor/80286/fetch.sh` fetches its pinned corpus separately.
  Manuals remain necessary when deriving new hardware facts.
  A test job took about 30 minutes with the former `target/` cache; the target is 18
  minutes or less with warm caches, and a docs-only push finishes in under 3 minutes.
  Each test job has a 90-minute timeout.
  Every push to `origin` starts a run. To rerun a ref without a new push, dispatch it
  with an authenticated GitHub CLI, `gh workflow run ci.yml --ref <ref>`; read the
  result in the commit's checks or with `gh run list --workflow ci.yml`. A dispatch
  always runs the full gate.
  A newer run on the same ref cancels an older one (concurrency group
  `ci-<workflow>-<ref>`), except on `main`: a `main` run classifies only the paths
  since the previous push, so each one finishes.
- **Local macOS**: `cargo nextest`, clippy and fmt additionally cover
  aarch64-apple-darwin. The Linux hosted lanes do not replace checks of Darwin-specific
  code. The native x86-64 Linux oracle and the real-program ladder do not run on macOS.
- The pre-push hook `ladder-smoke` builds `../unemulinux`, which does not compile on macOS. On that
  host the hook cannot pass; skipping it (`SKIP=ladder-smoke git push`, never `--no-verify`) is the
  maintainer's call for each push, after the other pre-push hooks have passed.

Run an individual CI gate from the repository root (the script enters the Nix
devShell for Cargo commands and writes `ci-logs/<step>.log`, or
`ci-logs/cross-<target>.log` for cross-checks):

```sh
bash scripts/ci-gate.sh fmt
bash scripts/ci-gate.sh clippy
bash scripts/ci-gate.sh deny
bash scripts/ci-gate.sh corpus       # fetch the corpus before the full test gate
bash scripts/ci-gate.sh tests
bash scripts/ci-gate.sh release
bash scripts/ci-gate.sh doctests
bash scripts/ci-gate.sh cross aarch64-unknown-linux-gnu   # on x86-64
bash scripts/ci-gate.sh cross x86_64-unknown-linux-gnu    # on aarch64
bash scripts/ci-gate.sh cross aarch64-unknown-freebsd
bash scripts/ci-gate.sh cross x86_64-unknown-freebsd
python3 -B -m unittest discover -s scripts/tests -p test_ci_gate.py -v  # gates, plan, workflow
XJ_DEVSHELL=host bash scripts/ci-gate.sh fmt   # fmt without Nix, as the docs-only lane runs it
python3 -B -m unittest discover -s scripts/tests -p test_legacy_scope.py -v   # § Test scope
nix shell nixpkgs#actionlint -c actionlint .github/workflows/ci.yml
```

`CI_LOG_DIR` overrides the log directory; the script preserves the failing command's
exit status through `tee`. The script tests stub external commands, checking dispatch,
required corpus settings, all four cross-targets and failure propagation without running
the full gate. `test_ci_gate.py` also runs `scripts/ci-plan.py` against throwaway
repositories and a stubbed `gh`, and checks the workflow's plan, docs-only, cache and
skip steps.

The FreeBSD lane runs `cargo check --locked --workspace --tests --target <target>`
with default features. It type-checks the libraries and test code, including the
FreeBSD code-memory and fork branches, without linking or executing FreeBSD binaries.
Unicorn is disabled because its target library is unavailable. Runtime coverage
remains with embedders until a FreeBSD runner exists.

`flake.nix` explicitly selects stable Rust **1.98.1**, with both FreeBSD targets in
the rust-overlay `targets` list. `flake.lock` pins the overlay to
`89750a2ef1c6a44db9f4251b16c23deffdcfddff`; its release manifest supplies prebuilt
`rust-std` archives dated **2026-09-03** for both targets. No nightly, `-Zbuild-std`,
FreeBSD sysroot or cross-linker is needed. The previous overlay pin selected 1.96.1,
whose manifest omitted ARM64 FreeBSD std. The selected release distributes both targets
([Rust target documentation](https://doc.rust-lang.org/rustc/platform-support/freebsd.html)).
This toolchain update leaves the workspace's declared Rust 1.95 MSRV unchanged.
Successful direct Cargo checks were reproduced with rustup stable 1.98.1 and both
prebuilt target libraries installed, outside the Nix devShell's 1.96.1 compiler.

On a cold machine, `nix develop --no-update-lock-file` fetches the pinned toolchain
and target components through Nix. In an offline environment, provision these first;
the FreeBSD archives are
`https://static.rust-lang.org/dist/2026-09-03/rust-std-1.98.1-aarch64-unknown-freebsd.tar.xz`
and
`https://static.rust-lang.org/dist/2026-09-03/rust-std-1.98.1-x86_64-unknown-freebsd.tar.xz`
(hashes are in the pinned overlay manifest). An installed rustup target does not
provide std to the Nix compiler; always use the gate to select the pinned toolchain.
