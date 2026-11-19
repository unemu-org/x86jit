#!/usr/bin/env bash
# Run one CI gate locally, preserving its exit status and combined log.
# Run from the repository root. CI_LOG_DIR defaults to ci-logs.
set -euo pipefail

usage() {
    echo 'Usage: bash scripts/ci-gate.sh {devshell|corpus|fmt|guard|clippy|deny|tests|ratchet|fast|heavy|release|build-tests|build-release|doctests|cross <target>}' >&2
    exit 2
}

[[ $# -ge 1 ]] || usage
step=$1
if [[ "$step" == cross ]]; then
    [[ $# -eq 2 ]] || usage
    case "$2" in
        x86_64-unknown-linux-gnu|aarch64-unknown-linux-gnu|x86_64-unknown-freebsd|aarch64-unknown-freebsd) ;;
        *) usage ;;
    esac
else
    [[ $# -eq 1 ]] || usage
fi

# The test tiers are nextest profiles in .config/nextest.toml
# (backlog/docs/commands.md § Test tiers): tests is the full tier and writes the
# JUnit report that ratchet checks, fast is the local hand-back gate, and heavy
# runs fuzz_robustness and the ignored soaks.
#
# XJ_DEVSHELL selects a flake dev shell other than the default; the aarch64
# Linux CI lane uses `no-freebsd` (flake.nix). `host` uses no Nix: the runner's
# rustup with the toolchain version flake.nix pins. Only fmt takes it (guard,
# corpus and ratchet never enter the shell); the docs-only CI lane uses it to skip
# the Nix install.
host_toolchain=
if [[ "${XJ_DEVSHELL:-}" == host ]]; then
    case "$step" in
        fmt | guard | corpus | ratchet) ;;
        *) echo "ci-gate: $step needs the Nix dev shell, not XJ_DEVSHELL=host" >&2; exit 2 ;;
    esac
    host_toolchain=$(sed -n 's/.*rust-bin\.stable\."\([0-9][0-9.]*\)".*/\1/p' flake.nix | head -n 1)
    [[ -n "$host_toolchain" ]] || {
        echo 'ci-gate: no rust-bin.stable."<version>" in flake.nix' >&2
        exit 2
    }
    gate_command=(rustup run "$host_toolchain")
elif [[ -n "${XJ_DEVSHELL:-}" && "${XJ_DEVSHELL}" != default ]]; then
    gate_command=(nix develop ".#${XJ_DEVSHELL}" --no-update-lock-file --command)
else
    gate_command=(nix develop --no-update-lock-file --command)
fi
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
# The release subset: the jit, differential, corpus, smc, mt, region_fault_state
# and rust_memory_faults modules of the grouped binaries, and fault_landing_pad.
release_tests=(--test codegen --test differential --test fault_landing_pad --test memory
    --test threads)
release_filter='binary(=fault_landing_pad) | (binary(=codegen) & test(/^jit::/))'
release_filter+=' | (binary(=differential) & test(/^(corpus|differential)::/))'
release_filter+=' | (binary(=memory) & test(/^(region_fault_state|rust_memory_faults)::/))'
release_filter+=' | (binary(=threads) & test(/^(mt|smc)::/))'
case "$step" in
    devshell) gate_command+=(rustc --version) ;;
    corpus) gate_command=(./x86jit-tests/vendor/80286/fetch.sh) ;;
    fmt) gate_command+=(cargo fmt --all --check) ;;
    guard) gate_command=(./scripts/guest-agnostic-guard.sh) ;;
    clippy)
        gate_command+=(cargo clippy --locked --workspace --all-targets --all-features -- -D warnings)
        ;;
    deny) gate_command+=(cargo deny --all-features check licenses bans sources) ;;
    tests)
        export SS286_REQUIRED=1
        # nextest.toml serializes code-memory campaigns and reserves the global
        # test budget while each runs; the soak and 100,000-unit bound are ignored.
        gate_command+=(cargo nextest run --profile ci --locked --workspace
            --features x86jit-tests/unicorn
            --no-fail-fast --success-output final -E 'not binary(fuzz_robustness)')
        ;;
    # Reads the report of the tests step; needs no dev shell. nextest writes it
    # under the workspace's target/nextest, also when CARGO_TARGET_DIR is set.
    ratchet)
        junit=target/nextest/ci/junit.xml
        gate_command=(python3 -B scripts/test-tiers.py ratchet "$junit")
        ;;
    fast)
        export SS286_REQUIRED=1
        gate_command+=(cargo nextest run --profile fast --locked --workspace
            --features x86jit-tests/unicorn --no-fail-fast)
        ;;
    heavy)
        gate_command+=(cargo nextest run --profile heavy --run-ignored all --locked --workspace
            --features x86jit-tests/unicorn --no-fail-fast --success-output final)
        ;;
    release)
        # Build only the binaries this gate runs: a nextest -E filter selects
        # after building every test binary of the package. The filter keeps the
        # release subset to its modules of the grouped binaries.
        gate_command+=(cargo nextest run --locked --release -p x86jit-tests
            --no-fail-fast --success-output final "${release_tests[@]}" -E "$release_filter")
        ;;
    # Compile what tests and release build, with the same profile, packages and
    # features, without running anything: CI fills main's dependency cache with them.
    build-tests)
        gate_command+=(cargo test --locked --workspace --features x86jit-tests/unicorn --no-run)
        ;;
    build-release)
        gate_command+=(cargo test --locked --release -p x86jit-tests --no-run "${release_tests[@]}")
        ;;
    doctests) gate_command+=(cargo test --locked --workspace --doc --all-features) ;;
    # Target Unicorn libraries are not available: type-check default features;
    # native lanes exercise Unicorn on their host architecture and OS.
    cross) gate_command+=(cargo check --locked --workspace --tests --target "$2") ;;
    *) usage ;;
esac

log_dir=${CI_LOG_DIR:-ci-logs}
log_name=$step
[[ "$step" != cross ]] || log_name="cross-$2"
mkdir -p "$log_dir"
if [[ -n "$host_toolchain" && "$step" == fmt ]]; then
    rustup toolchain install "$host_toolchain" --profile minimal --component rustfmt \
        --no-self-update 2>&1 | tee "$log_dir/toolchain.log"
fi
# The report goes with the logs, so a slow-list update can start from it.
if [[ "$step" == ratchet && -f "$junit" ]]; then
    cp "$junit" "$log_dir/junit-tests.xml"
fi
"${gate_command[@]}" 2>&1 | tee "$log_dir/$log_name.log"
