#!/usr/bin/env bash
# The decision-23 performance campaign: real16_loops on every workload and path, and
# the counted store probe, base against candidate. Builds both sides in the default
# release profile and in the `perf` profile (one codegen unit for the core and the
# bench), keeps base binaries by commit and profile, runs the processes with
# address-space randomization off, and prints each profile's per-cell table and path
# means and the two-profile verdict: a cell blocks only when it is over the gate in both
# profiles; over in one profile only, it is drift.
#
#   scripts/perf-campaign.sh [options] BASE [CANDIDATE]
#
# BASE is a commit. CANDIDATE is a commit or a directory holding a working tree; the
# default is the working tree of the checkout that holds this script. Run it inside the
# dev shell (`nix develop --command scripts/perf-campaign.sh ...`), at default QoS: the
# builds lower their own priority, the measuring must not.
#
# Exit status: 0 pass, 1 block, 2 a usage, build or measuring problem, 3 drift only.
# With --profile NAME (one profile, for exploration): 0 pass, 1 a cell misses, 2 problem. backlog/docs/commands.md § Performance campaign has the
# protocol; scripts/perf_campaign.py has the measuring and the analysis.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=$ROOT/scripts/perf_campaign.py

usage() {
  cat <<'EOF'
usage: scripts/perf-campaign.sh [options] BASE [CANDIDATE]

  BASE        a commit; its binaries are built once and kept in the cache
  CANDIDATE   a commit, or a directory holding a working tree
              (default: this checkout's working tree; a clean tree counts as its HEAD)

options:
  --profiles LIST   profiles to build, measure and combine (default release,perf):
                    a cell blocks only when it is over the gate in every one of them
  --profile NAME    one profile only (perf or release), for exploration; the verdict is
                    that profile's own (exit 0 pass, 1 miss, 2 problem)
  --cache DIR       binary cache and build directory
                    (default: $X86JIT_PERF_CACHE, else $TMPDIR/x86jit-perf-cache)
  --out DIR         where the raw data and the report go
                    (default: <cache>/runs/<time>-<base>-<candidate>)
  --processes N     processes per cell and side (default 3)
  --parallel N      processes at once (default 4)
  --gate PCT        per-cell gate in percent (default 0.5)
  --jobs N          CARGO_BUILD_JOBS for the builds (default 4)
  --build-qos MODE  background (default) or default
  --no-store        skip the store probe
  --build-only      build and cache both sides, do not measure
  -h, --help        this text
EOF
}

profiles=release,perf
single=0
cache=${X86JIT_PERF_CACHE:-${TMPDIR:-/tmp}/x86jit-perf-cache}
cache=${cache%/}
out=
processes=3
parallel=4
gate=0.5
jobs=4
build_qos=background
store=1
build_only=0

while [ $# -gt 0 ]; do
  case $1 in
    --profiles) profiles=$2; single=0; shift 2 ;;
    --profile) profiles=$2; single=1; shift 2 ;;
    --cache) cache=${2%/}; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --processes) processes=$2; shift 2 ;;
    --parallel) parallel=$2; shift 2 ;;
    --gate) gate=$2; shift 2 ;;
    --jobs) jobs=$2; shift 2 ;;
    --build-qos) build_qos=$2; shift 2 ;;
    --no-store) store=0; shift ;;
    --build-only) build_only=1; shift ;;
    -h | --help) usage; exit 0 ;;
    --) shift; break ;;
    -*) echo "perf-campaign: unknown option $1" >&2; usage >&2; exit 2 ;;
    *) break ;;
  esac
done
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  usage >&2
  exit 2
fi
IFS=, read -r -a profile_list <<<"$profiles"
if [ ${#profile_list[@]} -lt 1 ] || { [ "$single" = 1 ] && [ ${#profile_list[@]} != 1 ]; }; then
  echo "perf-campaign: --profile takes one profile; --profiles a comma-separated list" >&2
  exit 2
fi
for profile in "${profile_list[@]}"; do
  case $profile in
    perf | release) ;;
    *) echo "perf-campaign: profiles are perf and release, not '$profile'" >&2; exit 2 ;;
  esac
done
case $build_qos in
  background | default) ;;
  *) echo "perf-campaign: --build-qos must be background or default" >&2; exit 2 ;;
esac

die() {
  echo "perf-campaign: $*" >&2
  exit 2
}

# Resolve a commit in the repository that holds the script.
resolve_commit() {
  git -C "$ROOT" rev-parse --verify --quiet "$1^{commit}"
}

base_commit=$(resolve_commit "$1") || die "base '$1' is not a commit"
cand_arg=${2:-$ROOT}
cand_commit=
cand_tree=
cand_git=$ROOT
if [ -d "$cand_arg" ]; then
  cand_tree=$(cd "$cand_arg" && pwd)
  git -C "$cand_tree" rev-parse --show-toplevel >/dev/null 2>&1 || die "'$cand_arg' is not a git working tree"
  cand_tree=$(git -C "$cand_tree" rev-parse --show-toplevel)
  if [ -z "$(git -C "$cand_tree" status --porcelain --untracked-files=normal -- . ':!oracles')" ]; then
    # A clean tree is its HEAD commit, which can be cached.
    cand_commit=$(git -C "$cand_tree" rev-parse HEAD)
    cand_git=$cand_tree
    cand_tree=
  fi
else
  cand_commit=$(resolve_commit "$cand_arg") || die "candidate '$cand_arg' is neither a directory nor a commit"
fi

if ! command -v rustc >/dev/null || ! command -v cargo >/dev/null; then
  die "cargo not found; run inside the dev shell"
fi
# The cache key names the toolchain: another compiler gives other binaries.
toolchain=$(rustc -vV | shasum -a 256 | cut -c1-12)

mkdir -p "$cache/bin"
qos=()
if [ "$build_qos" = background ]; then
  if command -v taskpolicy >/dev/null 2>&1; then
    qos=(taskpolicy -c background)
  else
    qos=(nice -n 19)
  fi
fi

lock=$cache/lock
# One build at a time per cache: the snapshot directory and the target directory are
# shared. ensure_side runs in a subshell, so the lock lasts for one side's build.
take_lock() {
  local waited=0
  until mkdir "$lock" 2>/dev/null; do
    if [ $((waited % 60)) = 0 ]; then
      echo "perf-campaign: waiting for $lock (another campaign is building)" >&2
    fi
    sleep 5
    waited=$((waited + 5))
  done
  trap 'rmdir "$lock" 2>/dev/null || true' EXIT
}

# build_side LABEL DEST (commit in $side_commit and $side_git, or a tree in $side_tree)
build_side() {
  local label=$1 dest=$2
  local snap=$cache/src log=$cache/build-$label.log
  local cfg=() overlay=no start end
  rm -rf "$snap"
  mkdir -p "$snap"
  if [ -n "$side_tree" ]; then
    python3 -I "$PY" snapshot "$side_tree" "$snap"
  else
    git -C "$side_git" archive "$side_commit" | tar -x -C "$snap"
  fi
  # Extracted files keep old modification times; a shared target directory would then
  # reuse another side's artifacts. Make every source newer than any earlier build.
  find "$snap" -type f -exec touch {} +
  local probe=$snap/x86jit-bench/examples/interpreter_store_probe.rs
  if ! grep -q instructions_per_store "$probe" 2>/dev/null; then
    # A commit from before the counted probe: build this checkout's probe in its place.
    cp "$ROOT/x86jit-bench/examples/interpreter_store_probe.rs" "$probe"
    overlay=$(shasum -a 256 "$probe" | cut -c1-12)
  fi
  if [ "$profile" = perf ] && ! grep -q '^\[profile\.perf\]' "$snap/Cargo.toml"; then
    # A commit from before the profile: define the same profile on the command line.
    cfg=(--config 'profile.perf.inherits="release"'
      --config 'profile.perf.package.x86jit-core.codegen-units=1'
      --config 'profile.perf.package.x86jit-bench.codegen-units=1')
  fi
  echo "perf-campaign: building $label ($profile) ..." >&2
  start=$(date +%s)
  if ! (cd "$snap" && env CARGO_TARGET_DIR="$cache/target" CARGO_BUILD_JOBS="$jobs" \
    CARGO_INCREMENTAL=0 RUSTC_WRAPPER= ${qos[@]+"${qos[@]}"} cargo build --locked --profile "$profile" \
    -p x86jit-bench --example real16_loops --example interpreter_store_probe ${cfg[@]+"${cfg[@]}"}) \
    >"$log" 2>&1; then
    tail -30 "$log" >&2
    die "build of $label failed; log: $log"
  fi
  end=$(date +%s)
  local tmp=$dest.tmp.$$
  rm -rf "$tmp"
  mkdir -p "$tmp"
  cp "$cache/target/$profile/examples/real16_loops" "$cache/target/$profile/examples/interpreter_store_probe" "$tmp/"
  {
    echo "source=${side_commit:-tree $side_tree}"
    echo "profile=$profile"
    echo "profile_from_command_line=$([ ${#cfg[@]} -gt 0 ] && echo yes || echo no)"
    echo "store_probe_overlay=$overlay"
    echo "toolchain=$(rustc -V)"
    echo "build_seconds=$((end - start))"
    echo "built=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    (cd "$tmp" && shasum -a 256 real16_loops interpreter_store_probe | sed 's/^/sha256 /')
  } >"$tmp/meta.txt"
  rm -rf "$dest"
  mv "$tmp" "$dest"
  rm -rf "$snap"
  echo "perf-campaign: built $label in $((end - start)) s" >&2
}

# ensure_side LABEL COMMIT GIT TREE -> prints the binary directory
ensure_side() {
  local label=$1
  side_commit=$2 side_git=$3 side_tree=$4
  local dest
  if [ -n "$side_commit" ]; then
    dest=$cache/bin/$side_commit-$profile-$toolchain
    if [ -x "$dest/real16_loops" ] && [ -x "$dest/interpreter_store_probe" ]; then
      echo "perf-campaign: $label ${side_commit:0:12} ($profile) from the cache" >&2
      echo "$dest"
      return
    fi
  else
    dest=$cache/bin/tree-$$-$profile
  fi
  take_lock
  build_side "$label" "$dest"
  echo "$dest"
}

base_name=${base_commit:0:12}
cand_name=${cand_commit:0:12}
[ -n "$cand_name" ] || cand_name=tree
if [ -z "$out" ]; then
  out=$cache/runs/$(date +%Y%m%d-%H%M%S)-$base_name-$cand_name
fi
mkdir -p "$out"

# Build (or take from the cache) both sides in every profile before measuring any.
# base_bins[i] and cand_bins[i] belong to profile_list[i].
base_bins=()
cand_bins=()
for profile in "${profile_list[@]}"; do
  base_bins+=("$(ensure_side base "$base_commit" "$ROOT" "")")
  cand_bin=$(ensure_side candidate "$cand_commit" "$cand_git" "$cand_tree")
  if [ -n "$cand_tree" ]; then
    # A working tree's binaries are not cached; move them next to the run.
    dir=$out/cand-bin-$profile
    rm -rf "$dir"
    mv "$cand_bin" "$dir"
    cand_bin=$dir
  fi
  cand_bins+=("$cand_bin")
done
if [ "$build_only" = 1 ]; then
  for i in "${!profile_list[@]}"; do
    echo "${profile_list[$i]} base      ${base_bins[$i]}"
    echo "${profile_list[$i]} candidate ${cand_bins[$i]}"
  done
  exit 0
fi

store_args=()
[ "$store" = 1 ] || store_args=(--no-store)
measure() { # INDEX OUTDIR
  python3 -I "$PY" campaign --base "${base_bins[$1]}" --cand "${cand_bins[$1]}" --out "$2" \
    --base-name "$base_name" --cand-name "$cand_name${cand_tree:+ ($cand_tree)}" \
    --profile "${profile_list[$1]}" \
    --processes "$processes" --parallel "$parallel" --gate "$gate" ${store_args[@]+"${store_args[@]}"}
}
if [ "$single" = 1 ]; then
  measure 0 "$out"
  exit $?
fi

combine_args=()
for i in "${!profile_list[@]}"; do
  profile=${profile_list[$i]}
  echo "==================== profile $profile ===================="
  set +e
  measure "$i" "$out/$profile"
  status=$?
  set -e
  # 0 pass and 1 miss are verdicts for the combination; anything else is a problem
  # that combine reports from the analysis.
  [ -f "$out/$profile/analysis.json" ] || die "profile $profile produced no analysis (exit $status)"
  combine_args+=("$profile=$out/$profile/analysis.json")
done
echo "==================== both profiles ===================="
set +e
python3 -I "$PY" combine --gate "$gate" --out "$out" "${combine_args[@]}"
status=$?
set -e
echo "raw data and reports: $out"
exit $status
