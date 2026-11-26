#!/usr/bin/env bash
# The decision-23 performance campaign: real16_loops on every workload and path, and
# the counted store probe, base against candidate. Builds both sides in the default
# release profile and in the `perf` profile (one codegen unit for the core and the
# bench), all builds at once, keeps base binaries by commit and profile, runs the
# processes with address-space randomization off, and prints each profile's per-cell
# table and path means and the two-profile verdict: a cell blocks only when it is over
# the gate in both profiles; over in one profile only, it is drift.
#
#   scripts/perf-campaign.sh [options] BASE [CANDIDATE]
#   scripts/perf-campaign.sh --prebuilt DIR [options] BASE CANDIDATE
#
# BASE is a commit. CANDIDATE is a commit or a directory holding a working tree; the
# default is the working tree of the checkout that holds this script. Run it inside the
# dev shell (`nix develop --command scripts/perf-campaign.sh ...`), at default QoS. The
# measuring starts only after every build has finished, so no build runs beside it.
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
       scripts/perf-campaign.sh --prebuilt DIR [options] BASE CANDIDATE

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
  --jobs N          CARGO_BUILD_JOBS of each build (default: the CPUs divided by the
                    number of builds that run at once, rounded up)
  --build-qos MODE  default (the default) or background: background lowers the builds'
                    priority (taskpolicy -c background on macOS, efficiency cores only;
                    nice elsewhere), for a machine that other work shares
  --no-store        skip the store probe
  --build-only      build and cache both sides, do not measure
  --prebuilt DIR    measure only, no build and no cargo: DIR/base-<profile>/ and
                    DIR/candidate-<profile>/ hold each side's real16_loops and
                    interpreter_store_probe (and meta.txt); BASE and CANDIDATE are the
                    names the reports give them, checked against meta.txt's source
  -h, --help        this text
EOF
}

die() {
  echo "perf-campaign: $*" >&2
  exit 2
}

profiles=release,perf
single=0
cache=${X86JIT_PERF_CACHE:-${TMPDIR:-/tmp}/x86jit-perf-cache}
cache=${cache%/}
out=
processes=3
parallel=4
gate=0.5
jobs=
build_qos=default
store=1
build_only=0
prebuilt=

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
    --prebuilt) prebuilt=${2%/}; shift 2 ;;
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
  die "--profile takes one profile; --profiles a comma-separated list"
fi
for profile in "${profile_list[@]}"; do
  case $profile in
    perf | release) ;;
    *) die "profiles are perf and release, not '$profile'" ;;
  esac
done
case $build_qos in
  background | default) ;;
  *) die "--build-qos must be background or default" ;;
esac
case $jobs in
  '') ;;
  *[!0-9]* | 0*) die "--jobs takes a positive number" ;;
esac
if [ -n "$prebuilt" ]; then
  [ $# = 2 ] || die "--prebuilt takes BASE and CANDIDATE"
  [ "$build_only" = 0 ] || die "--prebuilt measures only; --build-only builds only"
fi

# Resolve a commit in the repository that holds the script.
resolve_commit() {
  git -C "$ROOT" rev-parse --verify --quiet "$1^{commit}"
}

cand_commit=
cand_tree=
cand_git=$ROOT
if [ -n "$prebuilt" ]; then
  # Names only: the binaries were built elsewhere (the CI build jobs), and this
  # checkout need not hold either commit.
  base_commit=$1
  cand_commit=$2
else
  base_commit=$(resolve_commit "$1") || die "base '$1' is not a commit"
  cand_arg=${2:-$ROOT}
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
fi

qos=()
if [ "$build_qos" = background ]; then
  if command -v taskpolicy >/dev/null 2>&1; then
    qos=(taskpolicy -c background)
  else
    qos=(nice -n 19)
  fi
fi

has_bins() {
  [ -x "$1/real16_loops" ] && [ -x "$1/interpreter_store_probe" ]
}

lock=$cache/lock
have_lock=0
# One campaign builds at a time per cache: its builds use the cache's snapshot and
# target directories, and together they take the CPUs. The lock lasts for the build
# phase; measuring and cached sides do not take it.
take_lock() {
  local waited=0
  until mkdir "$lock" 2>/dev/null; do
    if [ $((waited % 60)) = 0 ]; then
      echo "perf-campaign: waiting for $lock (another campaign is building)" >&2
    fi
    sleep 5
    waited=$((waited + 5))
  done
  have_lock=1
}

# build_side LABEL PROFILE COMMIT GIT TREE DEST JOBS: one side in one profile, from
# COMMIT of the repository GIT or from the working tree TREE, into DEST. Every
# LABEL-PROFILE slot has its own snapshot and target directory, so the builds of one
# campaign run side by side without waiting for cargo's lock on a shared target
# directory; a slot's target directory is kept, so its dependencies stay built.
build_side() {
  local label=$1 profile=$2 side_commit=$3 side_git=$4 side_tree=$5 dest=$6 jobs=$7
  local slot=$label-$profile
  local snap=$cache/src-$slot target=$cache/target-$slot log=$cache/build-$slot.log
  local cfg=() overlay=no start end
  rm -rf "$snap"
  mkdir -p "$snap"
  if [ -n "$side_tree" ]; then
    python3 -I "$PY" snapshot "$side_tree" "$snap"
  else
    git -C "$side_git" archive "$side_commit" | tar -x -C "$snap"
  fi
  # Extracted files keep old modification times; a kept target directory would then
  # reuse an earlier build's artifacts. Make every source newer than any earlier build.
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
  echo "perf-campaign: building $label ($profile, $jobs jobs) ..." >&2
  start=$(date +%s)
  if ! (cd "$snap" && env CARGO_TARGET_DIR="$target" CARGO_BUILD_JOBS="$jobs" \
    CARGO_INCREMENTAL=0 RUSTC_WRAPPER= ${qos[@]+"${qos[@]}"} cargo build --locked --profile "$profile" \
    -p x86jit-bench --example real16_loops --example interpreter_store_probe ${cfg[@]+"${cfg[@]}"}) \
    >"$log" 2>&1; then
    tail -30 "$log" | sed "s/^/[$slot] /" >&2
    die "build of $label ($profile) failed; log: $log"
  fi
  end=$(date +%s)
  local tmp=$dest.tmp.$$
  rm -rf "$tmp"
  mkdir -p "$tmp"
  cp "$target/$profile/examples/real16_loops" "$target/$profile/examples/interpreter_store_probe" "$tmp/"
  {
    echo "source=${side_commit:-tree $side_tree}"
    echo "profile=$profile"
    echo "profile_from_command_line=$([ ${#cfg[@]} -gt 0 ] && echo yes || echo no)"
    echo "store_probe_overlay=$overlay"
    echo "toolchain=$(rustc -V)"
    echo "build_jobs=$jobs"
    echo "build_seconds=$((end - start))"
    echo "built=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    (cd "$tmp" && shasum -a 256 real16_loops interpreter_store_probe | sed 's/^/sha256 /')
  } >"$tmp/meta.txt"
  rm -rf "$dest"
  mv "$tmp" "$dest"
  rm -rf "$snap"
  echo "perf-campaign: built $label ($profile) in $((end - start)) s" >&2
}

# The builds the campaign needs, as parallel arrays: one entry per build, each
# destination once (base and candidate may be the same commit).
t_label=()
t_profile=()
t_commit=()
t_git=()
t_tree=()
t_dest=()
add_build() { # LABEL PROFILE COMMIT GIT TREE DEST
  local d
  for d in ${t_dest[@]+"${t_dest[@]}"}; do
    [ "$d" != "$6" ] || return 0
  done
  t_label+=("$1")
  t_profile+=("$2")
  t_commit+=("$3")
  t_git+=("$4")
  t_tree+=("$5")
  t_dest+=("$6")
}

# Kill a process and everything below it (a build: env, cargo, rustc).
# shellcheck disable=SC2329 # called from on_exit
kill_tree() {
  local child
  for child in $(pgrep -P "$1" 2>/dev/null); do
    kill_tree "$child"
  done
  kill "$1" 2>/dev/null || true
}
build_pids=()
# shellcheck disable=SC2329 # the EXIT trap
on_exit() {
  local pid
  for pid in ${build_pids[@]+"${build_pids[@]}"}; do
    kill_tree "$pid"
  done
  [ "$have_lock" = 0 ] || rmdir "$lock" 2>/dev/null || true
}

# Run the builds in t_* side by side and wait for all of them. Each gets --jobs, or
# an equal share of the CPUs.
run_builds() {
  mkdir -p "$cache/bin"
  trap on_exit EXIT
  trap 'exit 130' INT TERM
  take_lock
  # Another campaign may have built a side while this one waited for the lock.
  local i keep=()
  for i in "${!t_dest[@]}"; do
    if has_bins "${t_dest[$i]}"; then
      echo "perf-campaign: ${t_label[$i]} (${t_profile[$i]}) built meanwhile, from the cache" >&2
    else
      keep+=("$i")
    fi
  done
  if [ ${#keep[@]} -gt 0 ]; then
    # The layout before the per-slot directories: one shared target and snapshot
    # directory, which nothing uses any more.
    rm -rf "$cache/target" "$cache/src"
    local build_jobs=$jobs cpus
    if [ -z "$build_jobs" ]; then
      cpus=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
      build_jobs=$(((cpus + ${#keep[@]} - 1) / ${#keep[@]}))
    fi
    echo "perf-campaign: ${#keep[@]} build(s) at once, $build_jobs jobs each" >&2
    local start pids=() failed=0
    start=$(date +%s)
    for i in "${keep[@]}"; do
      build_side "${t_label[$i]}" "${t_profile[$i]}" "${t_commit[$i]}" "${t_git[$i]}" \
        "${t_tree[$i]}" "${t_dest[$i]}" "$build_jobs" &
      pids+=($!)
      build_pids+=($!)
    done
    for i in "${!pids[@]}"; do
      wait "${pids[$i]}" || failed=$((failed + 1))
    done
    build_pids=()
    [ "$failed" = 0 ] || die "$failed build(s) failed"
    echo "perf-campaign: builds done in $(($(date +%s) - start)) s" >&2
  fi
  rmdir "$lock" 2>/dev/null || true
  have_lock=0
}

# check_prebuilt DIR NAME PROFILE: DIR holds binaries built from NAME in PROFILE.
check_prebuilt() {
  local dir=$1 name=$2 profile=$3 src prof
  has_bins "$dir" || die "no executable real16_loops and interpreter_store_probe in $dir"
  [ -f "$dir/meta.txt" ] || return 0
  src=$(sed -n 's/^source=//p' "$dir/meta.txt")
  prof=$(sed -n 's/^profile=//p' "$dir/meta.txt")
  case $src in
    "$name"*) ;;
    *) die "$dir was built from '$src', not '$name'" ;;
  esac
  [ "$prof" = "$profile" ] || die "$dir was built in profile '$prof', not '$profile'"
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
  if [ -n "$prebuilt" ]; then
    check_prebuilt "$prebuilt/base-$profile" "$base_commit" "$profile"
    check_prebuilt "$prebuilt/candidate-$profile" "$cand_commit" "$profile"
    base_bins+=("$prebuilt/base-$profile")
    cand_bins+=("$prebuilt/candidate-$profile")
    continue
  fi
  dest=$cache/bin/$base_commit-$profile-$toolchain
  base_bins+=("$dest")
  if has_bins "$dest"; then
    echo "perf-campaign: base ${base_commit:0:12} ($profile) from the cache" >&2
  else
    add_build base "$profile" "$base_commit" "$ROOT" "" "$dest"
  fi
  if [ -n "$cand_commit" ]; then
    dest=$cache/bin/$cand_commit-$profile-$toolchain
    if has_bins "$dest"; then
      echo "perf-campaign: candidate ${cand_commit:0:12} ($profile) from the cache" >&2
    else
      add_build candidate "$profile" "$cand_commit" "$cand_git" "" "$dest"
    fi
  else
    dest=$cache/bin/tree-$$-$profile
    add_build candidate "$profile" "" "" "$cand_tree" "$dest"
  fi
  cand_bins+=("$dest")
done
[ ${#t_dest[@]} = 0 ] || run_builds
if [ -n "$cand_tree" ]; then
  # A working tree's binaries are not cached; move them next to the run.
  for i in "${!profile_list[@]}"; do
    dir=$out/cand-bin-${profile_list[$i]}
    rm -rf "$dir"
    mv "${cand_bins[$i]}" "$dir"
    cand_bins[i]=$dir
  done
fi
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
