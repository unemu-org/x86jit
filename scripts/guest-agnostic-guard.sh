#!/usr/bin/env bash
# Refuse a tracked file that names a downstream consumer.
#
# x86jit is a guest-agnostic x86-64 engine: it knows nothing about a particular
# guest or about the emulator embedding it. That is the project's central
# claim, and it is the kind of claim that erodes one comment at a time — a lift driven
# by a downstream trap gets a note saying which game hit it, and a year later the
# engine is full of guest-specific references.
#
# It has happened: agents working a lift on a downstream project's behalf have left
# game names, guest runtime names and that project's task ids in engine source. A scan
# today is clean; nothing but this script stops it coming back.
#
#   scripts/guest-agnostic-guard.sh          # scan every tracked file and path
#
# The scan covers the whole tree: engine and test source, the docs and decisions under
# backlog/, top-level markdown, text files, scripts and configuration. The repository is
# public, so a document leaks a guest name as surely as a comment does. Binary files are
# skipped (grep -I) and their paths still checked; submodules are not entered. Describe a
# downstream user generically ("a FreeBSD embedder", "a Real16 embedder"). "Who reported
# this" belongs in the reporting project's own tracker: bead text here is guest-agnostic
# too (`bd recall publication-rule`).
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Downstream consumers, their guests, and their runtimes. Extend it when a new
# consumer appears — the cost of a false positive is renaming one comment; the cost of
# a miss is the engine quietly acquiring a guest.
DENY=(
  celeste "little nightmares" doom quake
  ps4 ps5 playstation orbis liverpool
  unemups4
  mono monogame sgen libfmod xna
  homebrew
)

# NOT on the list, deliberately:
#   unemulinux — the sibling embedder, not a guest. Naming it is correct and in places
#     required: boundary.rs exists to say what belongs there rather than here, and
#     several doc comments point at its design documents. A rule that forbade it would
#     forbid explaining the boundary.
#   fmod — C's floating-point remainder, which f80.rs legitimately cites. The leak
#     shape is `libfmod`, the audio library, so match that instead.
#   pkg — appears in `pkg-config` and is too generic to earn its place.
#
# Matched as whole words because they occur inside ordinary identifiers and English:
# `xna` inside `xnat()`, `mono` inside `monotonic`, `doom`/`quake` in prose.
WORD_ONLY=" mono doom quake homebrew xna "

# Exact phrases that contain a DENY term as a whole word but name no consumer. Each is
# removed from a matching line before the line is tested again, so the same term
# elsewhere on that line still fails the guard. Keep each entry exact and say where it
# occurs.
ALLOW=(
  -Zprint-mono-items # rustc's monomorphization dump flag (backlog/docs/performance.md)
)

# This script names every term it refuses, so it is the one file it cannot scan.
SELF=scripts/guest-agnostic-guard.sh

# Tracked regular files: a submodule is a directory here, and a file deleted from the
# working tree is not there to read.
files=()
while IFS= read -r -d '' path; do
  [[ "$path" != "$SELF" && -f "$path" ]] && files+=("$path")
done < <(git ls-files -z)
(( ${#files[@]} )) || { echo "no tracked files — nothing to scan"; exit 0; }

# Print the stdin lines on which $pattern still matches once every ALLOW phrase is
# removed. The printed line is the original.
drop_allowed() {
  local line rest phrase
  while IFS= read -r line; do
    rest=$line
    for phrase in "${ALLOW[@]}"; do
      rest=${rest//"$phrase"/}
    done
    if grep -qiE -- "$pattern" <<<"$rest"; then
      printf '%s\n' "$line"
    fi
  done
}

rc=0
for term in "${DENY[@]}"; do
  if [[ "$WORD_ONLY" == *" $term "* ]]; then
    pattern="\\b${term}\\b"
  else
    pattern="$term"
  fi
  # -I skips binaries, -H and -n give the file and line, -i because a comment may
  # capitalise it. grep exits 1 when nothing matches, which is the passing case.
  hits=$(grep -HIniE --color=never -- "$pattern" "${files[@]}" 2>/dev/null | drop_allowed || true)
  names=$(printf '%s\n' "${files[@]}" | grep -iE -- "$pattern" | drop_allowed || true)
  if [[ -n "$hits" ]]; then
    echo "guest-agnostic guard: a tracked file names a downstream consumer ('$term'):"
    # shellcheck disable=SC2001  # indenting every line of a multi-line block; the
    # parameter-expansion form does not do per-line replacement.
    echo "$hits" | sed 's/^/  /'
    rc=1
  fi
  if [[ -n "$names" ]]; then
    echo "guest-agnostic guard: a tracked path names a downstream consumer ('$term'):"
    # shellcheck disable=SC2001  # as above.
    echo "$names" | sed 's/^/  /'
    rc=1
  fi
done

if (( rc )); then
  cat >&2 <<'EOF'

x86jit is guest-agnostic: no downstream project, guest, game or runtime may be named
in a tracked file or path — source, docs and decisions alike. "Who reported this"
belongs in the reporting project's own tracker; beads here stay guest-agnostic as well.

If the term is a false positive (a crate name, an English word, a tool flag), add the
exact phrase to ALLOW, add the term to WORD_ONLY, or drop it from DENY in
scripts/guest-agnostic-guard.sh, with a comment saying why.
EOF
fi
exit $rc
