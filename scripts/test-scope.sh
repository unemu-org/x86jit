#!/usr/bin/env bash
# Choose the local test scope for this branch's changes, then print it or run it.
#
#   bash scripts/test-scope.sh [--run] <base-ref>
#   bash scripts/test-scope.sh --classify <path>...
#   bash scripts/test-scope.sh --plan {docs-only|legacy-only|shared}
#
# The changed paths are those that differ between the merge base of <base> and
# HEAD and the working tree (the commits of `git diff <base>...HEAD` plus staged
# and unstaged edits), together with untracked files. Rename detection is off, so
# a moved file counts at its old path and at its new one. Each path is classified
# as docs, legacy or shared, and the widest class present selects the scope:
#   docs-only    every path is docs. An empty diff runs nothing.
#   legacy-only  every path is legacy or docs.
#   shared       anything else. This runs the full gate, the gates CI runs.
# --run executes the commands and keeps going after a failure, as CI does; it
# exits 1 if any command failed. --classify prints the class of each path and
# --plan prints a scope's commands; neither runs anything.
# The script shortens local loops. CI on GitHub still runs the full gate on every
# push (backlog/docs/commands.md § Test scope).
set -euo pipefail

usage() {
    echo 'Usage: bash scripts/test-scope.sh [--run] <base-ref>' >&2
    echo '       bash scripts/test-scope.sh --classify <path>...' >&2
    echo '       bash scripts/test-scope.sh --plan {docs-only|legacy-only|shared}' >&2
    exit 2
}

# Legacy paths. A change confined to them cannot alter Long64, Compat32 or JIT
# execution, so the legacy-only scope runs only the tests that reach them.
# Evidence (repoctx rdeps, callers and search at dff162c5, each call site read):
#   Shared code enters these modules only at dispatch points that a Long64 or
#   Compat32 block cannot reach:
#     lift/mod.rs lift_insn, with_segment   legacy_context is set, or mode Real16
#     interp/mod.rs walk_ops,               IrOp::Legacy and the Real16-only IR
#       finish_instruction_fault            ops, which only legacy lowering emits
#     interp/control.rs exec_loop_cx        LoopCx, a Real16-only op
#     interp/integer.rs exec_load_retry     mode Real16
#     vm.rs run_engine_attempt,             DIRECT20 for a twenty-bit CpuModel;
#       set_cpu_model                       a CpuModel the embedder selected
#     vm/legacy.rs run_dispatch and the     KIND >= MACHINE_DISPATCH, or the
#       Machine execution methods           Machine-only legacy_execution state
#   The one exception is pinned: run_dispatch calls lift/legacy.rs alias_context
#   for every KIND, and the application branch ignores the result.
#   The JIT never receives legacy IR: x86jit-cranelift treats IrOp::Legacy as
#   unreachable! (codegen/mod.rs), and vm.rs rejects blocks with a legacy context
#   before any backend sees them. repoctx also lists x87/xstate.rs (`transfer`)
#   and jit_abi/fault.rs (`slot`) as callers; both are local closures that share
#   a name with a legacy function.
# Two candidate modules are shared, not legacy:
#   x86jit-core/src/legacy.rs defines types that every vCPU uses. CpuState::new
#   builds CpuState.legacy (state.rs), which is part of the frozen JIT ABI
#   layout; LegacyOp is a variant payload of every IrOp (ir.rs); every IrBlock
#   carries legacy_context; Vcpu::run compares ExecutionProfile.
#   x86jit-core/src/vm/legacy.rs holds run_dispatch, which is also the Long64 and
#   Compat32 dispatch loop: vm.rs run_application_dispatch calls it with
#   APPLICATION_DISPATCH.
# ir/verify/legacy_probes.rs is compiled only for the core unit tests.
# tests/support/model_api.rs and descriptor_fixture.rs are included only by
# legacy_* tests. backlog/docs/compat/ is the ISA map that the compat test,
# part of the legacy-only test set, checks.
# scripts/tests/test_legacy_scope.py fails when shared code gains a reference
# into these modules, when a pinned function changes, when a legacy file
# implements a shared type or adds a doctest, and when a test that selects
# Real16 or a CPU model is missing from the legacy-only test set below.
legacy_paths=(
    'x86jit-core/src/interp/legacy.rs' 'x86jit-core/src/interp/legacy/*'
    'x86jit-core/src/interp/i8086.rs' 'x86jit-core/src/interp/i8086/*'
    'x86jit-core/src/lift/legacy.rs' 'x86jit-core/src/lift/legacy/*'
    'x86jit-core/src/lift/i8086.rs' 'x86jit-core/src/lift/i8086/*'
    'x86jit-core/src/vm/nec.rs' 'x86jit-core/src/vm/nec/*'
    'x86jit-core/src/vm/i8086.rs' 'x86jit-core/src/vm/i8086/*'
    'x86jit-core/src/model.rs' 'x86jit-core/src/model/*'
    'x86jit-core/src/ir/verify/legacy_probes.rs'
    'x86jit-tests/tests/legacy_*'
    'x86jit-tests/tests/support/model_api.rs'
    'x86jit-tests/tests/support/descriptor_fixture.rs'
    'x86jit-tests/tests/support/port_block.rs'
    'backlog/docs/compat/*'
)
# Markdown and text in these directories can be compiled or compared as fixtures.
code_dirs=(x86jit-* scripts vendor .github .cargo .config)

# Legacy-only test set: every test binary that selects Real16 or a CPU model,
# directly, through a tests/support file, or through an x86jit-tests helper
# module that does so (compat, ss286). The set also keeps the shared checks
# boundary, environment_boundary, run_steps, irq_shadow, retired_count, smc,
# cross_region_* and application_codegen, which checks Long64 code generation.
# Binaries with these prefixes are found by file name.
legacy_test_prefixes=(legacy_ real16_ cross_region_)
legacy_tests=(
    application_codegen boundary cf16 compat coverage_ratchet cpuid
    environment_boundary evex_modifiers exit_request exit_request_signal fault_metadata
    fences flags_at_exit gathers
    irq_shadow machine_empty_fpu mmio_rmw mmx port_string retired_count run_steps
    same_block_smc smc ss286 string_mmio time_stats vcpu_reset x87_pointers
)

path_class() {
    local path=$1 pattern top
    case "$path" in
        backlog/docs/compat/*) echo legacy; return ;;
        backlog/*) echo docs; return ;;
        # CI checks out without submodules; no gate reads the oracle stash.
        oracles | oracles/*) echo docs; return ;;
    esac
    for pattern in "${legacy_paths[@]}"; do
        # shellcheck disable=SC2053 # the right-hand side is a glob on purpose
        if [[ "$path" == $pattern ]]; then
            echo legacy
            return
        fi
    done
    case "$path" in
        *.md | *.txt)
            top=${path%%/*}
            if [[ "$top" == "$path" ]]; then
                echo docs
                return
            fi
            for pattern in "${code_dirs[@]}"; do
                # shellcheck disable=SC2053
                if [[ "$top" == $pattern ]]; then
                    echo shared
                    return
                fi
            done
            echo docs
            return
            ;;
    esac
    echo shared
}

root=$(git rev-parse --show-toplevel 2>/dev/null) || {
    echo 'test-scope: not inside a git repository' >&2
    exit 2
}
cd "$root"

shell_prefix='nix develop'
if [[ -n "${XJ_DEVSHELL:-}" && "${XJ_DEVSHELL}" != default ]]; then
    shell_prefix+=" .#${XJ_DEVSHELL}"
fi
shell_prefix+=' --no-update-lock-file --command'
corpus_metadata=x86jit-tests/vendor/80286/v1_real_mode/metadata.json

commands=()
add_corpus_fetch() {
    # As in CI on a cache miss; ss286 and cf16 run with SS286_REQUIRED=1.
    [[ -f "$corpus_metadata" ]] || commands+=('bash scripts/ci-gate.sh corpus')
}

plan_docs() {
    # $1: the changed paths are known (a diff), or a placeholder is printed (--plan).
    local known=$1 path files=()
    shift
    if [[ "$known" == yes && $# -eq 0 ]]; then
        return
    fi
    commands+=('bash scripts/ci-gate.sh fmt' 'bash scripts/ci-gate.sh guard')
    if [[ "$known" == no ]]; then
        commands+=("$shell_prefix pre-commit run --files <changed files>")
        return
    fi
    for path in "$@"; do
        [[ -f "$path" ]] && files+=("$(printf '%q' "$path")")
    done
    if [[ ${#files[@]} -gt 0 ]]; then
        commands+=("$shell_prefix pre-commit run --files ${files[*]}")
    fi
}

plan_legacy() {
    local name prefix file tests=() test_args=''
    tests=("${legacy_tests[@]}")
    for prefix in "${legacy_test_prefixes[@]}"; do
        for file in x86jit-*/tests/"$prefix"*.rs; do
            [[ -f "$file" ]] || continue
            name=${file##*/}
            tests+=("${name%.rs}")
        done
    done
    while IFS= read -r name; do
        test_args+=" --test $name"
    done < <(printf '%s\n' "${tests[@]}" | LC_ALL=C sort -u)
    commands+=(
        'bash scripts/ci-gate.sh fmt'
        'bash scripts/ci-gate.sh guard'
        'bash scripts/ci-gate.sh clippy'
        'python3 -B -m unittest discover -s scripts/tests -p test_legacy_scope.py'
    )
    add_corpus_fetch
    commands+=(
        "SS286_REQUIRED=1 $shell_prefix cargo nextest run --locked --workspace --features x86jit-tests/unicorn --no-fail-fast$test_args"
        "$shell_prefix cargo nextest run --locked -p x86jit-core -p x86jit-tests --features x86jit-tests/unicorn --no-fail-fast --lib"
        'bash scripts/ci-gate.sh cross aarch64-unknown-linux-gnu'
        'bash scripts/ci-gate.sh cross x86_64-unknown-linux-gnu'
    )
}

plan_shared() {
    # .github/workflows/ci.yml, both lanes. The aarch64 Linux lane has no
    # FreeBSD std (flake.nix), so XJ_DEVSHELL=no-freebsd skips those checks.
    commands+=(
        "python3 -B -m unittest discover -s scripts/tests -p 'test_*.py'"
    )
    add_corpus_fetch
    commands+=(
        'bash scripts/ci-gate.sh fmt'
        'bash scripts/ci-gate.sh guard'
        'bash scripts/ci-gate.sh clippy'
        'bash scripts/ci-gate.sh deny'
        'bash scripts/ci-gate.sh tests'
        'bash scripts/ci-gate.sh release'
        'bash scripts/ci-gate.sh lane32'
        'bash scripts/ci-gate.sh doctests'
        'bash scripts/ci-gate.sh cross aarch64-unknown-linux-gnu'
        'bash scripts/ci-gate.sh cross x86_64-unknown-linux-gnu'
    )
    if [[ "${XJ_DEVSHELL:-}" != no-freebsd ]]; then
        commands+=(
            'bash scripts/ci-gate.sh cross aarch64-unknown-freebsd'
            'bash scripts/ci-gate.sh cross x86_64-unknown-freebsd'
        )
    fi
}

plan() {
    local scope=$1 known=$2
    shift 2
    case "$scope" in
        docs-only) plan_docs "$known" "$@" ;;
        legacy-only) plan_legacy ;;
        shared) plan_shared ;;
    esac
}

print_commands() {
    local command
    echo 'commands:'
    if [[ ${#commands[@]} -eq 0 ]]; then
        echo '  (none: nothing changed)'
    fi
    for command in ${commands[@]+"${commands[@]}"}; do
        echo "  $command"
    done
}

[[ $# -ge 1 ]] || usage
case "$1" in
    --classify)
        shift
        [[ $# -ge 1 ]] || usage
        for path in "$@"; do
            printf '%s\t%s\n' "$(path_class "$path")" "$path"
        done
        exit 0
        ;;
    --plan)
        [[ $# -eq 2 ]] || usage
        case "$2" in docs-only | legacy-only | shared) ;; *) usage ;; esac
        plan "$2" no
        echo "scope: $2"
        print_commands
        exit 0
        ;;
esac

run=no
if [[ "$1" == --run ]]; then
    run=yes
    shift
fi
[[ $# -eq 1 && "$1" != -* ]] || usage
base=$1
merge_base=$(git merge-base "$base" HEAD 2>/dev/null) || {
    echo "test-scope: no merge base between '$base' and HEAD" >&2
    exit 2
}

# The working tree against the merge base covers commits since it, staged and
# unstaged edits. Changes inside a submodule's own checkout are not the branch's.
paths=()
while IFS= read -r -d '' path; do
    paths+=("$path")
done < <({
    git diff --name-only --no-renames --ignore-submodules=dirty -z "$merge_base"
    git ls-files --others --exclude-standard -z
} | LC_ALL=C sort -zu)

scope=docs-only
echo "base: $base (merge base $(git rev-parse --short "$merge_base"))"
echo "changed paths: ${#paths[@]}"
for path in ${paths[@]+"${paths[@]}"}; do
    class=$(path_class "$path")
    printf '  %-6s  %s\n' "$class" "$path"
    case "$class" in
        shared) scope=shared ;;
        legacy) [[ "$scope" == shared ]] || scope=legacy-only ;;
    esac
done
plan "$scope" yes ${paths[@]+"${paths[@]}"}
echo "scope: $scope"
print_commands

[[ "$run" == yes ]] || exit 0
failed=()
for command in ${commands[@]+"${commands[@]}"}; do
    printf '\n==> %s\n' "$command"
    bash -c "$command" || failed+=("$command")
done
echo
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "test-scope: ${#failed[@]} of ${#commands[@]} commands failed:"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
echo "test-scope: ${#commands[@]} commands passed (scope $scope)"
