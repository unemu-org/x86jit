#!/usr/bin/env bash
# Run one CI gate locally, preserving its exit status and combined log.
# Run from the repository root. CI_LOG_DIR defaults to ci-logs.
set -euo pipefail

usage() {
    echo 'Usage: bash scripts/ci-gate.sh {devshell|corpus|fmt|guard|clippy|deny|tests|release|lane32|doctests|cross <target>}' >&2
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

gate_command=(nix develop --no-update-lock-file --command)
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
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
        gate_command+=(cargo nextest run --locked --workspace --features x86jit-tests/unicorn
            --no-fail-fast --success-output final -E 'not binary(fuzz_robustness)')
        ;;
    release)
        gate_command+=(cargo nextest run --locked --release -p x86jit-tests
            --no-fail-fast --success-output final
            -E 'binary(jit) or binary(differential) or binary(corpus) or binary(smc) or binary(mt) or binary(fault_landing_pad) or binary(region_fault_state) or binary(rust_memory_faults)')
        ;;
    lane32)
        gate_command+=(cargo nextest run --locked -p x86jit-tests --features unicorn
            --no-fail-fast --success-output final
            -E 'binary(differential32) or (binary(fuzz) and (test(jit_matches_interp_32) or test(unicorn_matches_interp_32)))')
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
"${gate_command[@]}" 2>&1 | tee "$log_dir/$log_name.log"
