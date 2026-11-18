#!/usr/bin/env python3
"""Decide what a CI run of .github/workflows/ci.yml must do, before any heavy job.

    python3 -B scripts/ci-plan.py

Run from the repository root in a full clone (checkout with fetch-depth 0). It reads
the GitHub Actions environment and writes `gates`, `docs` and `warm` (each "true" or
"false"), `scope` and `reason` to $GITHUB_OUTPUT, and prints them.

  gates  run the test jobs on both architectures (the full gate).
  docs   run the docs-only lane: fmt and the guest-agnostic guard, without Nix.
  warm   refresh main's dependency caches, because no test job runs to do it.

The run is skipped (gates and docs false) only on a push to main whose commit
already has a CI run, on any branch, in which both test jobs succeeded. Otherwise
the changed paths are classified by `scripts/test-scope.sh --scope`: docs-only
runs the docs lane, every other scope the full gate. The base is the previous
tip for a push to main, the merge base with origin/main for any other push, and
the base commit of a pull request.

Every doubt runs the full gate: a failed API call, an unknown event, a missing
base commit, an empty diff or a classifier error. The script itself never fails
the job; if it raises, the workflow step falls back to the full gate as well.
"""

import json
import os
from pathlib import Path
import subprocess
import sys


SCOPE_SCRIPT = Path(__file__).resolve().parent / "test-scope.sh"
MAIN = "refs/heads/main"
# The matrix of the test job in ci.yml; test_ci_gate.py keeps the two in step.
REQUIRED_JOBS = ("test (x86_64)", "test (aarch64)")
# Runs that tested the commit itself. A pull_request run tests a merge commit.
TRUSTED_EVENTS = ("push", "workflow_dispatch")
ZERO_SHA = "0" * 40


def git(*args):
    return subprocess.run(["git", *args], text=True, capture_output=True, check=False)


def gh_json(path):
    """GET a GitHub REST path with the gh CLI; raises on any failure."""
    result = subprocess.run(["gh", "api", "-H", "Accept: application/vnd.github+json", path],
                            text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"gh api {path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def workflow_file(env):
    # GITHUB_WORKFLOW_REF: owner/repo/.github/workflows/ci.yml@refs/heads/main
    ref = env.get("GITHUB_WORKFLOW_REF", "")
    name = ref.split("@", 1)[0].rsplit("/", 1)[-1]
    return name or "ci.yml"


def passed_elsewhere(env):
    """The id of an earlier run of this workflow, for this commit, whose test jobs
    all succeeded; None if there is none. Raises when the lookup fails."""
    repo, sha = env["GITHUB_REPOSITORY"], env["GITHUB_SHA"]
    current = str(env.get("GITHUB_RUN_ID", ""))
    runs = gh_json(f"repos/{repo}/actions/workflows/{workflow_file(env)}/runs"
                   f"?head_sha={sha}&status=success&per_page=100")
    for run in runs.get("workflow_runs", []):
        if (str(run.get("id")) == current or run.get("head_sha") != sha
                or run.get("event") not in TRUSTED_EVENTS
                or run.get("conclusion") != "success"):
            continue
        jobs = gh_json(f"repos/{repo}/actions/runs/{run['id']}/jobs?per_page=100")
        concluded = {job.get("name"): job.get("conclusion") for job in jobs.get("jobs", [])}
        if all(concluded.get(name) == "success" for name in REQUIRED_JOBS):
            return run["id"]
    return None


def commit_exists(rev):
    return git("cat-file", "-e", f"{rev}^{{commit}}").returncode == 0


def diff_base(env):
    """The base to classify against, or None when there is no safe one."""
    event, ref = env.get("GITHUB_EVENT_NAME", ""), env.get("GITHUB_REF", "")
    if event == "push" and ref == MAIN:
        before = env.get("EVENT_BEFORE", "")
        if before and before != ZERO_SHA and commit_exists(before):
            return before
        return None
    if event == "push":
        return "origin/main" if commit_exists("origin/main") else None
    if event == "pull_request":
        base = env.get("PR_BASE_SHA", "")
        return base if base and commit_exists(base) else None
    return None


def classify(base):
    result = subprocess.run(["bash", str(SCOPE_SCRIPT), "--scope", base], text=True,
                            capture_output=True, check=False)
    if result.returncode != 0:
        return None, f"test-scope.sh failed: {result.stderr.strip()}"
    return result.stdout.strip(), None


def plan(env):
    full = {"gates": "true", "docs": "false", "warm": "false", "scope": "shared"}
    is_main_push = env.get("GITHUB_EVENT_NAME") == "push" and env.get("GITHUB_REF") == MAIN
    if is_main_push:
        try:
            run = passed_elsewhere(env)
        except Exception as error:  # noqa: BLE001 - any lookup failure runs everything
            return {**full, "reason": f"run lookup failed ({error}); full gate"}
        if run is not None:
            return {"gates": "false", "docs": "false", "warm": "true", "scope": "skip",
                    "reason": f"commit already passed both test jobs in run {run}"}
    base = diff_base(env)
    if base is None:
        return {**full, "reason": "no safe diff base for this event; full gate"}
    scope, error = classify(base)
    if scope is None:
        return {**full, "reason": f"{error}; full gate"}
    if scope == "docs-only":
        return {"gates": "false", "docs": "true", "warm": "true" if is_main_push else "false",
                "scope": scope, "reason": f"every path changed since {base} is docs"}
    if scope == "none":
        return {**full, "reason": f"no path changed since {base}; full gate"}
    return {**full, "scope": scope, "reason": f"scope {scope} since {base}; full gate"}


def main():
    result = plan(os.environ)
    # One line per output: $GITHUB_OUTPUT is line-oriented.
    lines = [f"{key}={' '.join(str(result[key]).split())}" for key in ("gates", "docs", "warm", "scope", "reason")]
    print("\n".join(lines))
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
