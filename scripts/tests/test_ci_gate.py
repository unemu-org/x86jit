"""Exercise CI dispatch, the CI plan and the workflow's gate steps without compiling
or fetching data."""

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "scripts" / "ci-gate.sh"
PLAN = ROOT / "scripts" / "ci-plan.py"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
STUB = """#!/usr/bin/env python3
import json
import os
import sys
with open(os.environ['CALL_FILE'], 'a') as calls:
    calls.write(json.dumps({
        'program': os.path.basename(sys.argv[0]),
        'args': sys.argv[1:],
        'corpus_required': os.environ.get('SS286_REQUIRED'),
        'backtrace': os.environ.get('RUST_BACKTRACE'),
    }) + '\\n')
print('gate stdout')
print('gate stderr', file=sys.stderr)
sys.exit(int(os.environ.get('STUB_STATUS', '0')))
"""


class CiGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in (
            "bin/nix",
            "bin/rustup",
            "scripts/guest-agnostic-guard.sh",
            "x86jit-tests/vendor/80286/fetch.sh",
        ):
            stub = self.root / name
            stub.parent.mkdir(parents=True, exist_ok=True)
            stub.write_text(STUB)
            stub.chmod(0o755)
        self.call_file = self.root / "call.json"
        self.env = {
            **os.environ,
            "PATH": f"{self.root / 'bin'}{os.pathsep}{os.environ['PATH']}",
            "CALL_FILE": str(self.call_file),
            "CI_LOG_DIR": str(self.root / "logs"),
            "STUB_STATUS": "0",
            "RUST_BACKTRACE": "1",
        }
        self.env.pop("SS286_REQUIRED", None)
        self.env.pop("XJ_DEVSHELL", None)

    def run_gate(self, *args, status=0):
        result = subprocess.run(
            ["bash", str(GATE), *args],
            cwd=self.root,
            env={**self.env, "STUB_STATUS": str(status)},
            text=True,
            capture_output=True,
            check=False,
        )
        return result

    def calls(self):
        return [json.loads(line) for line in self.call_file.read_text().splitlines()]

    def call(self):
        return self.calls()[-1]

    def test_full_gate_requires_corpus_and_excludes_only_robustness(self):
        result = self.run_gate("tests")
        self.assertEqual(result.returncode, 0, result.stderr)
        call = self.call()
        self.assertEqual(call["corpus_required"], "1")
        self.assertEqual(call["backtrace"], "1")
        self.assertEqual(call["args"], [
            "develop", "--no-update-lock-file", "--command", "cargo", "nextest",
            "run", "--profile", "ci", "--locked", "--workspace", "--features",
            "x86jit-tests/unicorn", "--no-fail-fast", "--success-output", "final", "-E",
            "not binary(fuzz_robustness)",
        ])

    def test_fast_tier_requires_corpus_and_uses_the_fast_profile(self):
        result = self.run_gate("fast")
        self.assertEqual(result.returncode, 0, result.stderr)
        call = self.call()
        self.assertEqual(call["corpus_required"], "1")
        self.assertEqual(call["args"], [
            "develop", "--no-update-lock-file", "--command", "cargo", "nextest",
            "run", "--profile", "fast", "--locked", "--workspace", "--features",
            "x86jit-tests/unicorn", "--no-fail-fast",
        ])

    def test_heavy_tier_runs_the_ignored_tests_of_the_heavy_profile(self):
        result = self.run_gate("heavy")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.call()["args"], [
            "develop", "--no-update-lock-file", "--command", "cargo", "nextest",
            "run", "--profile", "heavy", "--run-ignored", "all", "--locked", "--workspace",
            "--features", "x86jit-tests/unicorn", "--no-fail-fast", "--success-output",
            "final",
        ])

    def test_ratchet_reads_the_full_tier_report_without_nix(self):
        tiers = self.root / "scripts" / "test-tiers.py"
        tiers.write_text(STUB)
        tiers.chmod(0o755)
        junit = self.root / "target" / "nextest" / "ci" / "junit.xml"
        junit.parent.mkdir(parents=True)
        junit.write_text("<testsuites/>\n")
        result = self.run_gate("ratchet")
        self.assertEqual(result.returncode, 0, result.stderr)
        call = self.call()
        self.assertEqual(call["program"], "test-tiers.py")
        self.assertEqual(call["args"], ["ratchet", "target/nextest/ci/junit.xml"])
        self.assertEqual((self.root / "logs" / "junit-tests.xml").read_text(),
                         "<testsuites/>\n")
        self.assertIn("run: bash scripts/ci-gate.sh ratchet", WORKFLOW.read_text())

    def test_other_gates_dispatch_without_running_heavy_jobs(self):
        expected = {
            "devshell": ["rustc", "--version"],
            "fmt": ["cargo", "fmt", "--all", "--check"],
            "clippy": ["cargo", "clippy", "--locked", "--workspace", "--all-targets",
                       "--all-features", "--", "-D", "warnings"],
            "deny": ["cargo", "deny", "--all-features", "check", "licenses", "bans", "sources"],
            "build-tests": ["cargo", "test", "--locked", "--workspace", "--features",
                            "x86jit-tests/unicorn", "--no-run"],
            "build-release": ["cargo", "test", "--locked", "--release", "-p", "x86jit-tests",
                              "--no-run", "--test", "jit", "--test", "differential",
                              "--test", "corpus", "--test", "smc", "--test", "mt",
                              "--test", "fault_landing_pad", "--test", "region_fault_state",
                              "--test", "rust_memory_faults"],
            # Only the binaries it runs are built: no -E filter after a full build.
            "release": ["cargo", "nextest", "run", "--locked", "--release", "-p", "x86jit-tests",
                        "--no-fail-fast", "--success-output", "final",
                        "--test", "jit", "--test", "differential", "--test", "corpus",
                        "--test", "smc", "--test", "mt", "--test", "fault_landing_pad",
                        "--test", "region_fault_state", "--test", "rust_memory_faults"],
            "doctests": ["cargo", "test", "--locked", "--workspace", "--doc", "--all-features"],
        }
        for step, command in expected.items():
            with self.subTest(step=step):
                result = self.run_gate(step)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.call()["args"], [
                    "develop", "--no-update-lock-file", "--command", *command,
                ])
        for step in ("corpus", "guard"):
            with self.subTest(step=step):
                result = self.run_gate(step)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.call()["args"], [])

    def test_release_binaries_exist(self):
        tests = ROOT / "x86jit-tests" / "tests"
        for name in ("jit", "differential", "corpus", "smc", "mt", "fault_landing_pad",
                     "region_fault_state", "rust_memory_faults"):
            with self.subTest(name=name):
                self.assertTrue((tests / f"{name}.rs").is_file())

    def test_host_shell_formats_with_the_flake_toolchain_without_nix(self):
        (self.root / "flake.nix").write_text(
            'toolchainFor = targets: pkgs.rust-bin.stable."1.98.1".default.override {\n')
        self.env["XJ_DEVSHELL"] = "host"
        result = self.run_gate("fmt")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual([c["program"] for c in calls], ["rustup", "rustup"])
        self.assertEqual(calls[0]["args"], [
            "toolchain", "install", "1.98.1", "--profile", "minimal", "--component", "rustfmt",
            "--no-self-update",
        ])
        self.assertEqual(calls[1]["args"], [
            "run", "1.98.1", "cargo", "fmt", "--all", "--check",
        ])
        self.call_file.unlink()
        result = self.run_gate("guard")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.call()["program"], "guest-agnostic-guard.sh")
        self.call_file.unlink()
        for step in ("clippy", "tests", "fast", "heavy", "release", "build-tests",
                     "build-release"):
            with self.subTest(step=step):
                result = self.run_gate(step)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("needs the Nix dev shell", result.stderr)
                self.assertFalse(self.call_file.exists())

    def test_host_shell_without_a_pinned_version_fails_before_dispatch(self):
        (self.root / "flake.nix").write_text("rust-bin.stable.latest.default\n")
        self.env["XJ_DEVSHELL"] = "host"
        result = self.run_gate("fmt")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(self.call_file.exists())

    def test_host_toolchain_failure_stops_the_gate(self):
        (self.root / "flake.nix").write_text('rust-bin.stable."1.98.1".default\n')
        self.env["XJ_DEVSHELL"] = "host"
        result = self.run_gate("fmt", status=7)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(len(self.calls()), 1)

    def test_cross_checks_cover_workspace_tests_on_linux_and_freebsd(self):
        for target in ("aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu",
                       "aarch64-unknown-freebsd", "x86_64-unknown-freebsd"):
            with self.subTest(target=target):
                result = self.run_gate("cross", target)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.call()["args"], [
                    "develop", "--no-update-lock-file", "--command", "cargo", "check",
                    "--locked", "--workspace", "--tests", "--target", target,
                ])
                self.assertIn("gate stdout", (self.root / f"logs/cross-{target}.log").read_text())

    def test_named_dev_shell_is_selected(self):
        self.env["XJ_DEVSHELL"] = "no-freebsd"
        result = self.run_gate("fmt")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.call()["args"], [
            "develop", ".#no-freebsd", "--no-update-lock-file", "--command",
            "cargo", "fmt", "--all", "--check",
        ])
        self.env["XJ_DEVSHELL"] = "default"
        result = self.run_gate("fmt")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.call()["args"][:3], ["develop", "--no-update-lock-file", "--command"])

    def test_freebsd_failure_survives_tee(self):
        result = self.run_gate("cross", "aarch64-unknown-freebsd", status=23)
        self.assertEqual(result.returncode, 23, result.stderr)
        log = (self.root / "logs/cross-aarch64-unknown-freebsd.log").read_text()
        self.assertIn("gate stderr", log)

    def test_failure_survives_tee_and_both_streams_are_logged(self):
        result = self.run_gate("fmt", status=23)
        self.assertEqual(result.returncode, 23, result.stderr)
        log = (self.root / "logs/fmt.log").read_text()
        self.assertIn("gate stdout", log)
        self.assertIn("gate stderr", log)
        self.assertEqual(result.stdout, log)

    def test_invalid_invocations_fail_before_dispatch(self):
        for args in ((), ("unknown",), ("lane32",), ("cross",), ("cross", "invalid"),
                     ("fmt", "extra"), ("cross", "x86_64-unknown-linux-gnu", "extra")):
            with self.subTest(args=args):
                result = self.run_gate(*args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("Usage:", result.stderr)
                self.assertFalse(self.call_file.exists())


def workflow_jobs():
    """Job id -> its text in ci.yml (two-space job keys under `jobs:`)."""
    text = WORKFLOW.read_text().split("\njobs:\n", 1)[1]
    parts = re.split(r"(?m)^  ([\w-]+):\n", text)
    return dict(zip(parts[1::2], parts[2::2]))


def steps_of(job_text):
    """Each `- name:` step of a job, as its text."""
    return re.split(r"(?m)^      - (?=name:)", job_text)[1:]


def cache_keys(job_text):
    return sorted(re.findall(r"(?m)^\s+(?:key|primary-key): ((?:nix|cargo)-v\d-.+)$", job_text))


class Workflow(unittest.TestCase):
    """The workflow steps that ci-gate.sh and ci-plan.py depend on."""

    def setUp(self):
        self.text = WORKFLOW.read_text()
        self.jobs = workflow_jobs()

    def test_jobs(self):
        self.assertEqual(set(self.jobs), {"plan", "test", "warm-caches"})

    def test_every_gate_step_is_one_ci_gate_dispatches(self):
        usage = re.search(r"ci-gate\.sh \{([^}]*)\}", GATE.read_text()).group(1).split("|")
        steps = {gate.split()[0] for gate in
                 re.findall(r"(?:run: |run: \|\n\s+)?bash scripts/ci-gate\.sh (\S+)",
                            self.text)}
        self.assertTrue(steps)
        for step in steps:
            self.assertIn(step if step != "cross" else "cross <target>", usage)
        self.assertNotIn("lane32", self.text)

    def test_dependency_cache_is_rust_cache_with_workspace_crates_pruned(self):
        # target/ is cached only through rust-cache, which keeps compiled dependencies
        # and prunes workspace crates; no plain cache step may hold it.
        for block in re.findall(r"(?ms)^\s+path: \|\n(.*?)^\s+key:", self.text):
            paths = [line.strip() for line in block.splitlines()]
            self.assertFalse([p for p in paths if p.startswith("target")])
        self.assertNotIn("path: target", self.text)
        self.assertNotIn("actions/cache/restore@0057852bfaa89a56745cba8c7296529d2fc39830 # v4.3.0\n"
                         "        with:\n          path: |\n            ~/.cargo", self.text)
        blocks = {}
        for job in ("test", "warm-caches"):
            steps = [step for step in steps_of(self.jobs[job])
                     if step.startswith("name: Rust dependency cache")]
            self.assertEqual(len(steps), 1, job)
            step = steps[0]
            self.assertRegex(step, r"uses: Swatinem/rust-cache@[0-9a-f]{40} # v\d+\.\d+\.\d+")
            self.assertNotIn("cache-workspace-crates", step)
            self.assertNotIn("cache-all-crates", step)
            self.assertIn("save-if: ${{ github.event_name == 'push' && "
                          "github.ref == 'refs/heads/main' }}", step)
            # One entry per architecture and dev shell, the same in both jobs.
            self.assertIn("shared-key: ${{ matrix.arch }}-${{ matrix.devshell }}", step)
            self.assertIn("cmd-format: nix develop .#${{ matrix.devshell }} "
                          "--no-update-lock-file --command {0}", step)
            blocks[job] = [line for line in step.split("with:", 1)[1].splitlines()
                           if line.strip() and not line.strip().startswith("#")]
        self.assertEqual(blocks["test"], blocks["warm-caches"])

    def test_jobs_sharing_the_rust_cache_have_the_same_env_and_matrix(self):
        def env(job):
            return re.search(r"(?ms)^    env:\n(.*?)^    steps:", self.jobs[job]).group(1)
        def matrix(job):
            return re.findall(r"(?m)^          - arch: .*(?:\n            .*)*",
                              self.jobs[job])
        def strip_comments(text):
            return "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))
        self.assertEqual(strip_comments(env("test")), strip_comments(env("warm-caches")))
        test_rows = [re.sub(r"\n\s+#.*", "", row) for row in matrix("test")]
        self.assertEqual(test_rows, matrix("warm-caches"))

    def test_warm_job_builds_what_the_test_job_builds(self):
        warm = self.jobs["warm-caches"]
        for gate in ("clippy", "build-tests", "build-release", 'cross "$CROSS_TARGET"',
                     "cross aarch64-unknown-freebsd", "cross x86_64-unknown-freebsd"):
            self.assertIn(f"bash scripts/ci-gate.sh {gate}", warm)
        for step in steps_of(warm):
            if step.startswith("name: Build"):
                self.assertIn("steps.rust_cache.outputs.cache-hit != 'true'", step)

    def test_plan_job_has_full_history_and_reads_runs(self):
        plan = self.jobs["plan"]
        self.assertIn("fetch-depth: 0", plan)
        self.assertIn("python3 -B scripts/ci-plan.py ||", plan)
        self.assertIn("printf '%s\\n' gates=true docs=false warm=false scope=shared", plan)
        self.assertIn("GH_TOKEN: ${{ github.token }}", plan)
        self.assertIn("EVENT_BEFORE: ${{ github.event.before }}", plan)
        self.assertRegex(self.text, r"(?m)^permissions:\n  contents: read\n  actions: read\n")
        self.assertIn("python3 -B -m unittest discover -s scripts/tests -p 'test_*.py'", plan)

    def test_docs_lane_runs_fmt_and_the_guard_without_nix(self):
        plan = self.jobs["plan"]
        docs = [step for step in steps_of(plan)
                if "steps.plan.outputs.docs == 'true'" in step]
        self.assertEqual(len(docs), 2)
        self.assertIn("XJ_DEVSHELL: host", docs[0])
        self.assertIn("run: bash scripts/ci-gate.sh fmt", docs[0])
        self.assertIn("run: bash scripts/ci-gate.sh guard", docs[1])
        self.assertNotIn("nix-quick-install-action", plan)
        self.assertNotIn("nix develop", plan)

    def test_test_job_runs_only_when_the_plan_says_so(self):
        test = self.jobs["test"]
        self.assertIn("needs: plan", test)
        self.assertIn("if: ${{ !cancelled() && needs.plan.outputs.gates == 'true' }}", test)
        warm = self.jobs["warm-caches"]
        self.assertIn("needs.plan.outputs.warm == 'true'", warm)
        self.assertIn("github.ref == 'refs/heads/main'", warm.split("steps:")[0])

    def test_plan_names_the_test_jobs_of_the_matrix(self):
        test = self.jobs["test"]
        self.assertIn("name: test (${{ matrix.arch }})", test)
        arches = re.findall(r"(?m)^\s+- arch: (\S+)$", test)
        names = tuple(f"test ({arch})" for arch in arches)
        self.assertEqual(sorted(names), sorted(load_plan().REQUIRED_JOBS))

    def test_only_main_pushes_save_caches(self):
        for step in steps_of(self.jobs["test"]):
            if re.search(r"uses: \S+/save@", step):
                with self.subTest(step=step.splitlines()[0]):
                    self.assertIn("github.event_name == 'push' && "
                                  "github.ref == 'refs/heads/main'", step)

    def test_warm_job_uses_the_test_job_cache_keys(self):
        self.assertEqual(cache_keys(self.jobs["warm-caches"]), cache_keys(self.jobs["test"]))
        self.assertEqual(len(cache_keys(self.jobs["test"])), 1)  # nix; rust-cache: above
        corpus = "key: ss286-${{ hashFiles('x86jit-tests/vendor/80286/fetch.sh') }}"
        self.assertIn(corpus, self.jobs["warm-caches"])
        self.assertIn(corpus, self.jobs["test"])


def load_plan():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ci_plan", PLAN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STUB_GH = """#!/usr/bin/env python3
import json
import os
import sys
path = sys.argv[-1]
with open(os.environ['GH_LOG'], 'a') as log:
    log.write(path + '\\n')
fixture = json.loads(open(os.environ['GH_FIXTURE']).read())
if path not in fixture:
    print('HTTP 404', file=sys.stderr)
    sys.exit(1)
print(json.dumps(fixture[path]))
"""
SHA_RUNS = ("repos/o/r/actions/workflows/ci.yml/runs?head_sha={sha}&status=success"
            "&per_page=100")


class CiPlan(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name)
        self.repo = self.temp / "repo"
        self.repo.mkdir()
        bin_dir = self.temp / "bin"
        bin_dir.mkdir()
        (bin_dir / "gh").write_text(STUB_GH)
        (bin_dir / "gh").chmod(0o755)
        self.fixture = self.temp / "fixture.json"
        self.fixture.write_text("{}")
        self.output = self.temp / "output"
        self.gh_log = self.temp / "gh.log"
        self.env = {
            **os.environ,
            "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.invalid",
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "GH_FIXTURE": str(self.fixture), "GH_LOG": str(self.gh_log),
            "GITHUB_OUTPUT": str(self.output),
            "GITHUB_REPOSITORY": "o/r", "GITHUB_RUN_ID": "900",
            "GITHUB_WORKFLOW_REF": "o/r/.github/workflows/ci.yml@refs/heads/main",
            "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/ci/work",
            "EVENT_BEFORE": "", "PR_BASE_SHA": "",
        }
        for name in ("XJ_DEVSHELL",):
            self.env.pop(name, None)
        self.git("init", "-q", "-b", "main")
        self.write("README.md")
        self.write("x86jit-core/src/vm.rs")
        self.write("x86jit-core/src/vm/nec.rs")
        self.commit()
        self.git("update-ref", "refs/remotes/origin/main", "HEAD")
        self.base = self.head()

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.repo, env=self.env, text=True,
                              capture_output=True, check=True).stdout.strip()

    def write(self, path, text="original\n"):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def commit(self):
        self.git("add", "-A")
        self.git("commit", "-q", "--no-verify", "-m", "change")

    def head(self):
        return self.git("rev-parse", "HEAD")

    def plan(self, **env):
        environment = {**self.env, "GITHUB_SHA": self.head(), **env}
        result = subprocess.run(["python3", "-B", str(PLAN)], cwd=self.repo, env=environment,
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        outputs = dict(line.split("=", 1) for line in self.output.read_text().splitlines())
        self.output.unlink()
        self.assertEqual(result.stdout.splitlines(),
                         [f"{k}={outputs[k]}" for k in ("gates", "docs", "warm", "scope",
                                                        "reason")])
        return outputs

    def assertPlan(self, outputs, gates, docs, warm, scope):
        self.assertEqual((outputs["gates"], outputs["docs"], outputs["warm"], outputs["scope"]),
                         (gates, docs, warm, scope), outputs["reason"])

    def runs(self, sha, *runs):
        fixture = json.loads(self.fixture.read_text())
        fixture[SHA_RUNS.format(sha=sha)] = {"workflow_runs": [
            {"id": run_id, "head_sha": sha, "event": event, "conclusion": "success"}
            for run_id, event, _ in runs]}
        for run_id, _, jobs in runs:
            fixture[f"repos/o/r/actions/runs/{run_id}/jobs?per_page=100"] = {"jobs": [
                {"name": name, "conclusion": conclusion} for name, conclusion in jobs.items()]}
        self.fixture.write_text(json.dumps(fixture))

    PASSED = {"plan": "success", "test (x86_64)": "success", "test (aarch64)": "success"}
    DOCS_ONLY = {"plan": "success", "test (x86_64)": "skipped", "test (aarch64)": "skipped"}

    def main_push(self, before):
        return {"GITHUB_REF": "refs/heads/main", "EVENT_BEFORE": before}

    def test_branch_docs_only_runs_the_docs_lane(self):
        self.write("backlog/docs/guide.md")
        self.write("README.md", "edited\n")
        self.commit()
        self.assertPlan(self.plan(), "false", "true", "false", "docs-only")
        self.assertFalse(self.gh_log.exists())  # no lookup off main

    def test_branch_code_legacy_and_ci_changes_run_the_full_gate(self):
        for path, scope in (("x86jit-core/src/vm/nec.rs", "legacy-only"),
                            ("x86jit-core/src/vm.rs", "shared"),
                            (".github/workflows/ci.yml", "shared"),
                            ("scripts/notes.md", "shared"),
                            ("Cargo.lock", "shared"),
                            ("x86jit-tests/tests/vectors/a.txt", "shared")):
            with self.subTest(path=path):
                self.git("checkout", "-q", "-B", "work", self.base)
                self.write("backlog/docs/guide.md")
                self.write(path, "edited\n")
                self.commit()
                self.assertPlan(self.plan(), "true", "false", "false", scope)

    def test_branch_classifies_against_origin_main_not_the_previous_push(self):
        self.write("x86jit-core/src/vm.rs", "edited\n")
        self.commit()
        before = self.head()
        self.write("backlog/docs/guide.md")
        self.commit()
        self.assertPlan(self.plan(EVENT_BEFORE=before), "true", "false", "false", "shared")

    def test_without_a_base_the_full_gate_runs(self):
        self.write("backlog/docs/guide.md")
        self.commit()
        self.git("update-ref", "-d", "refs/remotes/origin/main")
        self.assertPlan(self.plan(), "true", "false", "false", "shared")
        self.assertPlan(self.plan(GITHUB_EVENT_NAME="workflow_dispatch"),
                        "true", "false", "false", "shared")
        self.assertPlan(self.plan(GITHUB_EVENT_NAME="pull_request"),
                        "true", "false", "false", "shared")
        self.assertPlan(self.plan(GITHUB_EVENT_NAME="pull_request", PR_BASE_SHA=self.base),
                        "false", "true", "false", "docs-only")

    def test_an_empty_diff_runs_the_full_gate(self):
        outputs = self.plan()
        self.assertPlan(outputs, "true", "false", "false", "shared")
        self.assertIn("no path changed", outputs["reason"])

    def test_main_skips_a_commit_that_passed_both_test_jobs(self):
        self.write("x86jit-core/src/vm.rs", "edited\n")
        self.commit()
        self.runs(self.head(), (1, "push", self.PASSED))
        outputs = self.plan(**self.main_push(self.base))
        self.assertPlan(outputs, "false", "false", "true", "skip")
        self.assertIn("run 1", outputs["reason"])

    def test_main_does_not_count_runs_that_did_not_test_the_commit(self):
        self.write("x86jit-core/src/vm.rs", "edited\n")
        self.commit()
        failed = {**self.PASSED, "test (aarch64)": "failure"}
        missing = {"plan": "success", "test (x86_64)": "success"}
        self.runs(self.head(), (900, "push", self.PASSED),       # this run
                  (2, "pull_request", self.PASSED),              # a merge commit
                  (3, "push", self.DOCS_ONLY), (4, "push", failed), (5, "push", missing))
        self.assertPlan(self.plan(**self.main_push(self.base)),
                        "true", "false", "false", "shared")
        self.runs(self.head(), (6, "workflow_dispatch", self.PASSED))
        self.assertPlan(self.plan(**self.main_push(self.base)),
                        "false", "false", "true", "skip")

    def test_main_lookup_failure_runs_the_full_gate(self):
        self.write("backlog/docs/guide.md")
        self.commit()
        outputs = self.plan(**self.main_push(self.base))  # no fixture: gh fails
        self.assertPlan(outputs, "true", "false", "false", "shared")
        self.assertIn("lookup failed", outputs["reason"])

    def test_main_docs_only_since_the_previous_push(self):
        self.write("x86jit-core/src/vm.rs", "edited\n")
        self.commit()
        before = self.head()
        self.write("backlog/docs/guide.md")
        self.commit()
        self.runs(self.head(), (3, "push", self.DOCS_ONLY))
        self.assertPlan(self.plan(**self.main_push(before)), "false", "true", "true",
                        "docs-only")
        # The whole push counts, not its last commit.
        self.assertPlan(self.plan(**self.main_push(self.base)), "true", "false", "false",
                        "shared")

    def test_main_without_a_previous_tip_runs_the_full_gate(self):
        self.write("backlog/docs/guide.md")
        self.commit()
        self.runs(self.head())
        for before in ("", "0" * 40, "1" * 40):
            with self.subTest(before=before):
                self.assertPlan(self.plan(**self.main_push(before)),
                                "true", "false", "false", "shared")


if __name__ == "__main__":
    unittest.main()
