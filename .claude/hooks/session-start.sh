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
  # --flat: one line per bead with the id; errors are shown, never folded into "(none)".
  print_section "in-progress beads"
  if out=$(bd list --status in_progress --flat 2>&1); then
    out=$(printf '%s\n' "$out" | grep -E 'xj-' | head -12)
    if [ -n "$out" ]; then echo "$out"; else echo "  (none)"; fi
  else
    echo "  (bd list failed: $(printf '%s' "$out" | head -1))"
  fi
  print_section "ready beads (bd ready --exclude-type epic, top 15)"
  if out=$(bd ready --exclude-type epic 2>&1); then
    out=$(printf '%s\n' "$out" | grep -E 'xj-' | sed 's/ ← .*//' | head -15)
    if [ -n "$out" ]; then echo "$out"; else echo "  (none)"; fi
  else
    echo "  (bd ready failed: $(printf '%s' "$out" | head -1))"
  fi
else
  print_section "beads"
  echo "  (bd not on PATH or .beads/ missing — see backlog/docs/dev-setup.md § beads)"
fi

print_section "load order reminder"
cat <<'EOF'
1. AGENTS.md (root) → conventions + pointer table
2. backlog/docs/agent-workflow.md → roles + bead-to-main flow; working-on-tasks.md → bd usage
3. bd list --status in_progress, then bd show <id> → recent notes on the active bead
4. Read only the backlog/docs/*.md files relevant to the task
5. backlog/docs/design/spec.md → authoritative design spec + milestones
EOF

exit 0
