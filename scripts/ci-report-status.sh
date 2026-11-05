#!/usr/bin/env bash
# Report a CI job's outcome as a commit status, so the result and every failing
# step can be read through the forge API without opening the job log.
#
# Environment (set by .forgejo/workflows/ci.yml):
#   TOKEN       the job's automatic token
#   API         .../api/v1/repos/<owner>/<repo>/statuses/<sha>
#   RUN_URL     link to the run, used as the status target
#   JOB_STATUS  success | failure | cancelled
#   OUTCOMES    space-separated <step>=<outcome> pairs, in step order; a step's
#               output is in ci-logs/<step>.log
#
# This script always exits 0: it reports the job result, it never changes it.
set -uo pipefail

# The forge stores the description as text; keep it bounded anyway.
LIMIT=12000
PER_STEP=40
PANICS=24
MESSAGE=240

# nextest result words for a failed test: FAIL, FAIL + LEAK, LEAK-FAIL, TIMEOUT,
# a named signal (SIGSEGV, SIGKILL, …) or ABORT / ABORT SIG <n>. Lines look like
# "FAIL [   0.042s] (  9/12) binary test". Plain LEAK is a passing test.
FAILED_TEST='^\s*(FAIL( \+ LEAK)?|LEAK-FAIL|TIMEOUT|SIG[A-Z0-9]+|ABORT( SIG [0-9]+)?) \['

case "${JOB_STATUS:-failure}" in
  success)
    state=success
    desc="all gate steps passed"
    ;;
  cancelled)
    state=error
    desc="job cancelled"
    ;;
  *)
    state=failure
    desc="job failed outside the gate steps"
    names=""
    details=""
    for pair in ${OUTCOMES:-}; do
      step=${pair%%=*}
      outcome=${pair#*=}
      [ "$outcome" = failure ] || continue
      names="$names $step"
      log="ci-logs/$step.log"
      if [ ! -f "$log" ]; then
        detail="(no log)"
      else
        lines=$(grep -E "$FAILED_TEST" "$log" | sed -E 's/^\s+//' | sort -u)
        if [ -z "$lines" ]; then
          # Not a test step, or no test result line: the error lines, else the tail.
          lines=$(grep -E 'error|panicked|failed' "$log")
          [ -n "$lines" ] || lines=$(tail -n "$PER_STEP" "$log")
        fi
        count=$(printf '%s\n' "$lines" | grep -c .)
        detail=$(printf '%s\n' "$lines" | head -n "$PER_STEP")
        [ "$count" -le "$PER_STEP" ] || detail="$detail
… $((count - PER_STEP)) more"
        [ -n "$detail" ] || detail="(empty log)"
        # The first panic messages: the line after each "panicked at <file:line:col>:",
        # plus an assertion's "left:" and "right:" lines, which carry its values.
        # One line per distinct message, with the first test that raised it. Each
        # line is cut, so that a long context string cannot hide the values.
        panics=$(grep -A3 -E 'panicked at .*:[0-9]+:[0-9]+:$' "$log" \
          | grep -vE '^--$' | sed -E 's/^\s+//' \
          | awk -v cut="$MESSAGE" '
              function flush() { if (state && !seen[msg]++) print where ": " msg; state = 0 }
              /panicked at/ { flush(); where = $2; state = 1; next }
              state == 1 { msg = substr($0, 1, cut); state = 2; next }
              state == 2 && /^(left|right):/ { msg = msg " | " substr($0, 1, cut); next }
              { flush() }
              END { flush() }' \
          | head -n "$PANICS")
        [ -z "$panics" ] || detail="$detail
panics:
$panics"
      fi
      details="$details$step failed:
$detail
"
    done
    [ -z "$names" ] || desc="failed:$names
$details"
    ;;
esac

# The native-oracle tests compare against this CPU, and x87 and SSE corner cases
# differ by vendor: name the CPU model (never the machine) on the first line.
cpu=$(sed -n 's/^model name[[:space:]]*: *//p' /proc/cpuinfo 2>/dev/null | head -n 1)
desc="host CPU: ${cpu:-unknown}
$desc"

desc=$(printf '%s' "$desc" | tr -d '\r')
if [ "${#desc}" -gt "$LIMIT" ]; then
  desc="$(printf '%s' "$desc" | head -c "$LIMIT")
[truncated]"
fi

body=$(STATE="$state" DESC="$desc" node -e '
  process.stdout.write(JSON.stringify({
    state: process.env.STATE,
    context: "ci / x86_64",
    description: process.env.DESC,
    target_url: process.env.RUN_URL,
  }));
')

if ! resp=$(curl -sS --fail-with-body -X POST -H "Authorization: token $TOKEN" \
  -H "Content-Type: application/json" --data "$body" "$API" 2>&1); then
  echo "status report failed (the job result stands): $(printf '%s' "$resp" | head -c 500)"
fi
printf '%s\n%s\n' "$state" "$desc"
exit 0
