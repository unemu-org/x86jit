#!/usr/bin/env bash

# >>> repoctx (managed — edits here are overwritten) >>>
repoctx prime 2>/dev/null
# <<< repoctx (managed) <<<
# Claude Code SessionStart hook (project-local).
# Prints a quick orientation snapshot so the agent doesn't burn tokens
# rediscovering project state.

set -u

cd "${CLAUDE_PROJECT_DIR:-$(pwd)}" 2>/dev/null || exit 0

print_section() {
  printf '\n--- %s ---\n' "$1"
}

print_section "branch + last 5 commits"
git log --format="%h %s" -5 2>/dev/null || echo "(no git)"

# --- open work lives in beads (`bd`, prefix xj); docs and decisions stay under backlog/ ---
if command -v bd >/dev/null 2>&1 && [ -d .beads ]; then
  # The shared working rules (roles, bead-to-main, worker briefs, publication rule) are beads
  # memories. Printed first and in full: hook output is capped, and these must not be cut.
  # stdout and stderr are kept apart so a bd warning cannot break the JSON parse.
  errf=$(mktemp)
  print_section "beads memories (shared rules — bd memories)"
  if out=$(bd memories --json 2>"$errf"); then
    printf '%s' "$out" | python3 -I -c 'import json,sys
for k,v in sorted(json.load(sys.stdin).items()):
  if k != "schema_version": print(f"[{k}] {v}\n")' 2>/dev/null \
      || echo "  (could not parse bd memories --json — run it by hand)"
  else
    echo "  (bd memories failed: $(head -1 "$errf"))"
  fi
  # --flat: one line per bead with the id; errors are shown, never folded into "(none)".
  print_section "in-progress beads"
  if out=$(bd list --status in_progress --flat 2>"$errf"); then
    out=$(printf '%s\n' "$out" | grep -E 'xj-' | head -8)
    if [ -n "$out" ]; then echo "$out"; else echo "  (none)"; fi
  else
    echo "  (bd list failed: $(head -1 "$errf"))"
  fi
  print_section "ready beads (bd ready --exclude-type epic, top 8)"
  if out=$(bd ready --exclude-type epic 2>"$errf"); then
    out=$(printf '%s\n' "$out" | grep -E 'xj-' | sed 's/ ← .*//' | head -8)
    if [ -n "$out" ]; then echo "$out"; else echo "  (none)"; fi
  else
    echo "  (bd ready failed: $(head -1 "$errf"))"
  fi
  rm -f "$errf"
else
  print_section "beads"
  echo "  (bd not on PATH or .beads/ missing — see backlog/docs/dev-setup.md § beads)"
fi

print_section "load order reminder"
cat <<'EOF'
1. AGENTS.md (root) → conventions + pointer table
2. bd memories → roles, bead-to-main flow, worker rules; backlog/docs/working-on-tasks.md → bd usage
3. bd list --status in_progress, then bd show <id> → recent notes on the active bead
4. Read only the backlog/docs/*.md files relevant to the task
5. backlog/docs/design/spec.md → authoritative design spec + milestones
EOF

exit 0
