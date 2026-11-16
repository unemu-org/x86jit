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
timeout 180 nice -n 10 cargo --offline test -p x86jit-tests --test region_fault_state --features unicorn -- --test-threads 1
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
remain ignored. The latter took 40.88 seconds alone on macOS arm64 in debug;
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
cargo nextest run --release -p x86jit-tests -E 'binary(jit) or binary(differential)'
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

## Test scope

`scripts/test-scope.sh` chooses the smallest local command set that covers a branch's
changes. It is meant for workers' inner loops and for the local gate before work is handed
back. It does not replace CI: GitHub still runs the full gate on every push.

```sh
bash scripts/test-scope.sh origin/main            # classify the changes, print the commands
bash scripts/test-scope.sh --run origin/main      # run them; keeps going, exits 1 on any failure
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
| legacy-only | every path is legacy or docs | fmt, the guard, clippy with `-D warnings`, `test_legacy_scope.py`, every test binary that selects Real16 or a CPU model (all `legacy_*` and `real16_*`, `ss286`, `cf16`, `compat` and others), the shared checks `boundary`, `environment_boundary`, `run_steps`, `irq_shadow`, `retired_count`, `smc`, `cross_region_*` and `application_codegen`, the `x86jit-core` and `x86jit-tests` unit tests, and `cargo check --workspace --tests` for both Linux targets |
| shared | any other path | the full gate: every `scripts/ci-gate.sh` step that CI runs on either lane, both Linux cross-checks, the FreeBSD checks unless `XJ_DEVSHELL=no-freebsd`, and the gate-script tests |

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
  pushes and dispatches start it. Its two independent jobs use GitHub-hosted
  `ubuntu-24.04`
  (x86-64) and `ubuntu-24.04-arm` (aarch64) runners. Both install pinned Nix and use
  the devShell from `flake.lock`. Each runs fmt, the guest-agnostic guard, clippy with
  all features and `-D warnings`, cargo-deny, the workspace nextest run with Unicorn
  and without `fuzz_robustness`, the release subset, the Compat32 lane, workspace
  doctests with all features, and a workspace/test type-check for the opposite Linux
  architecture, plus separate FreeBSD compile-check steps for both architectures.
  The cross-check uses default features because the cross architecture's
  Unicorn library is unavailable; each native lane tests Unicorn. The x86-64 Linux
  job also exercises the cfg-gated native oracle.
  After the devShell succeeds, later gates run even if an earlier gate fails; either
  job's failure fails the workflow, without cancelling the other architecture.
  GitHub supplies the job checks; this workflow posts no commit statuses of its own.
  Combined gate logs are retained as `ci-logs-x86_64` / `ci-logs-aarch64` artifacts for
  seven days. Installer and cache diagnostics remain in the Actions job log.
  All actions are pinned to commit SHAs, checkout does not persist credentials, and
  the only repository permission is `contents: read`. No configured secrets are
  needed, and every runner is GitHub-hosted.
  Cargo caches are separated by OS, architecture, Nix inputs, gate script and Cargo
  lockfile; the pinned 80286 corpus cache is shared between architectures. Only push
  runs save caches; pull requests can restore them. A corpus cache miss fetches the
  corpus, and `SS286_REQUIRED=1` makes an absent corpus fail the test gate.
  The `oracles` submodule is deliberately not checked out: these gates use in-code
  witness values, while `vendor/80286/fetch.sh` fetches its pinned corpus separately.
  Manuals remain necessary when deriving new hardware facts.
  Allow roughly 20–40 minutes per lane with cold caches and 10–25 minutes with warm
  caches. These are planning estimates pending the first hosted run; each job has a
  90-minute timeout. Runner provisioning, Nix binary-cache availability, disk use,
  corpus downloads and native-oracle results must be checked on that first run.
  Every push to `origin` starts a run. To rerun a ref without a new push, dispatch it
  with an authenticated GitHub CLI, `gh workflow run ci.yml --ref <ref>`; read the
  result in the commit's checks or with `gh run list --workflow ci.yml`.
  A newer run on the same ref cancels an older one (concurrency group
  `ci-<workflow>-<ref>`).
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
bash scripts/ci-gate.sh lane32
bash scripts/ci-gate.sh doctests
bash scripts/ci-gate.sh cross aarch64-unknown-linux-gnu   # on x86-64
bash scripts/ci-gate.sh cross x86_64-unknown-linux-gnu    # on aarch64
bash scripts/ci-gate.sh cross aarch64-unknown-freebsd
bash scripts/ci-gate.sh cross x86_64-unknown-freebsd
python3 -B -m unittest discover -s scripts/tests -p test_ci_gate.py -v  # no builds/downloads
python3 -B -m unittest discover -s scripts/tests -p test_legacy_scope.py -v   # § Test scope
nix shell nixpkgs#actionlint -c actionlint .github/workflows/ci.yml
```

`CI_LOG_DIR` overrides the log directory; the script preserves the failing command's
exit status through `tee`. The script tests stub external commands, checking dispatch,
required corpus settings, all four cross-targets and failure propagation without running
the full gate.

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
