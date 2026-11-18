"""Exercise scripts/test-scope.sh on throwaway git repositories, without builds."""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "test-scope.sh"
CI = ROOT / ".github" / "workflows" / "ci.yml"

# A minimal tree with a file of each class, committed on `main`.
BASE_FILES = (
    "README.md",
    "backlog/docs/guide.md",
    "backlog/docs/compat/coverage.json",
    "x86jit-core/src/vm.rs",
    "x86jit-core/src/legacy.rs",
    "x86jit-core/src/vm/legacy.rs",
    "x86jit-core/src/vm/nec.rs",
    "x86jit-tests/tests/legacy_real.rs",
    "x86jit-tests/tests/smc.rs",
)
STUB_GATE = """#!/usr/bin/env bash
echo "gate $*" >> "$CALL_LOG"
[[ "$1" != "${FAIL_STEP:-}" ]] || exit 3
"""
STUB_NIX = """#!/usr/bin/env bash
echo "nix $*" >> "$CALL_LOG"
"""


class Repo:
    def __init__(self, case):
        temp = tempfile.TemporaryDirectory()
        case.addCleanup(temp.cleanup)
        self.dir = Path(temp.name) / "repo"
        self.bin = Path(temp.name) / "bin"
        self.log = Path(temp.name) / "calls.log"
        self.bin.mkdir()
        (self.bin / "nix").write_text(STUB_NIX)
        (self.bin / "nix").chmod(0o755)
        self.env = {
            **os.environ,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.invalid",
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "CALL_LOG": str(self.log),
        }
        self.env.pop("XJ_DEVSHELL", None)
        self.dir.mkdir()
        self.git("init", "-q", "-b", "main")
        for path in BASE_FILES:
            self.write(path)
        self.write("scripts/ci-gate.sh", STUB_GATE)
        self.commit("base")
        self.git("checkout", "-q", "-b", "work")

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.dir, env=self.env, text=True,
                              capture_output=True, check=True).stdout

    def write(self, path, text="original\n"):
        target = self.dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def commit(self, message="change"):
        self.git("add", "-A")
        self.git("commit", "-q", "--no-verify", "-m", message)

    def scope(self, *args, env=None):
        return subprocess.run(["bash", str(SCRIPT), *args], cwd=self.dir,
                              env={**self.env, **(env or {})}, text=True,
                              capture_output=True, check=False)


def scope_of(result):
    return re.search(r"(?m)^scope: (\S+)$", result.stdout).group(1)


def commands_of(result):
    return re.search(r"(?ms)^commands:\n(.*?)(?:^\S|\Z)", result.stdout).group(1).splitlines()


class Classification(unittest.TestCase):
    def test_path_classes(self):
        expected = {
            "backlog/docs/doc-28 - Republish handoff.md": "docs",
            "backlog/decisions/decision-1.md": "docs",
            "backlog/config.yml": "docs",
            "backlog/docs/compat/coverage.json": "legacy",
            "backlog/docs/compat/isa-coverage.md": "legacy",
            "README.md": "docs",
            "AGENTS.md": "docs",
            "notes.txt": "docs",
            "bench/README.md": "docs",
            ".agents/skills/backlog/SKILL.md": "docs",
            "oracles": "docs",
            "x86jit-elf/README.md": "shared",
            "x86jit-cranelift/tests/snapshots/application/Long64-call.txt": "shared",
            "scripts/notes.md": "shared",
            "vendor/cranelift-jit/README.md": "shared",
            "x86jit-core/src/interp/legacy.rs": "legacy",
            "x86jit-core/src/interp/legacy/tasks.rs": "legacy",
            "x86jit-core/src/interp/i8086.rs": "legacy",
            "x86jit-core/src/lift/i8086.rs": "legacy",
            "x86jit-core/src/lift/legacy.rs": "legacy",
            "x86jit-core/src/vm/nec.rs": "legacy",
            "x86jit-core/src/vm/i8086.rs": "legacy",
            "x86jit-core/src/model.rs": "legacy",
            "x86jit-core/src/model/decode/i8086.rs": "legacy",
            "x86jit-core/src/ir/verify/legacy_probes.rs": "legacy",
            "x86jit-tests/tests/legacy_real.rs": "legacy",
            "x86jit-tests/tests/support/model_api.rs": "legacy",
            "x86jit-tests/tests/support/descriptor_fixture.rs": "legacy",
            # Reached from shared paths (scripts/test-scope.sh, evidence block).
            "x86jit-core/src/legacy.rs": "shared",
            "x86jit-core/src/vm/legacy.rs": "shared",
            "x86jit-core/src/vm.rs": "shared",
            "x86jit-core/src/interp/mod.rs": "shared",
            "x86jit-core/src/ir/verify.rs": "shared",
            "x86jit-tests/tests/real16_wrap.rs": "shared",
            "x86jit-tests/tests/support/canonical_mmap.rs": "shared",
            "x86jit-tests/src/compat.rs": "shared",
            "Cargo.toml": "shared",
            ".github/workflows/ci.yml": "shared",
            "LICENSE-MIT": "shared",
        }
        result = subprocess.run(["bash", str(SCRIPT), "--classify", *expected],
                                cwd=ROOT, text=True, capture_output=True, check=True)
        actual = {path: cls for cls, path in
                  (line.split("\t", 1) for line in result.stdout.splitlines())}
        self.assertEqual(actual, expected)


class Scopes(unittest.TestCase):
    def setUp(self):
        self.repo = Repo(self)

    def assertScope(self, scope, *args):
        result = self.repo.scope(*(args or ("main",)))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(scope_of(result), scope, result.stdout)
        return result

    def test_empty_diff_is_a_docs_only_no_op(self):
        result = self.assertScope("docs-only")
        self.assertIn("changed paths: 0", result.stdout)
        self.assertEqual(commands_of(result), ["  (none: nothing changed)"])
        result = self.repo.scope("--run", "main")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.repo.log.exists())

    def test_docs_only(self):
        self.repo.write("backlog/docs/guide.md", "edited\n")
        self.repo.write("backlog/docs/doc-2 - With space.md", "new\n")
        self.repo.commit()
        self.repo.write("README.md", "uncommitted\n")
        result = self.assertScope("docs-only")
        commands = commands_of(result)
        self.assertEqual(commands[:2], ["  bash scripts/ci-gate.sh fmt",
                                        "  bash scripts/ci-gate.sh guard"])
        self.assertEqual(commands[2], (
            "  nix develop --no-update-lock-file --command pre-commit run --files "
            "README.md backlog/docs/doc-2\\ -\\ With\\ space.md backlog/docs/guide.md"))
        self.assertEqual(len(commands), 3)

    def test_legacy_only_including_docs(self):
        self.repo.write("x86jit-core/src/vm/nec.rs", "edited\n")
        self.repo.write("x86jit-tests/tests/legacy_real.rs", "edited\n")
        self.repo.write("backlog/docs/guide.md", "edited\n")
        self.repo.commit()
        result = self.assertScope("legacy-only")
        commands = "\n".join(commands_of(result))
        self.assertIn("bash scripts/ci-gate.sh clippy", commands)
        self.assertIn("--test legacy_real ", commands)
        self.assertIn("-p test_legacy_scope.py", commands)
        self.assertNotIn("ci-gate.sh tests", commands)
        self.assertNotIn("ci-gate.sh release", commands)

    def test_shared(self):
        self.repo.write("x86jit-core/src/vm.rs", "edited\n")
        self.repo.commit()
        result = self.assertScope("shared")
        self.assertIn("  bash scripts/ci-gate.sh fast", commands_of(result))
        self.assertNotIn("  bash scripts/ci-gate.sh tests", commands_of(result))

    def test_legacy_and_shared_together_are_shared(self):
        self.repo.write("x86jit-core/src/vm/nec.rs", "edited\n")
        self.repo.write("x86jit-core/src/legacy.rs", "edited\n")
        self.repo.commit()
        self.assertScope("shared")

    def test_a_rename_counts_at_both_paths(self):
        self.repo.git("mv", "x86jit-core/src/vm/nec.rs", "x86jit-core/src/vm/renamed.rs")
        self.repo.commit()
        self.assertScope("shared")
        repo = Repo(self)
        repo.git("mv", "x86jit-core/src/vm.rs", "x86jit-core/src/vm/i8086.rs")
        repo.commit()
        result = repo.scope("main")
        self.assertEqual(scope_of(result), "shared", result.stdout)
        self.assertIn("shared  x86jit-core/src/vm.rs", result.stdout)

    def test_a_rename_between_legacy_paths_stays_legacy(self):
        self.repo.git("mv", "x86jit-tests/tests/legacy_real.rs",
                      "x86jit-tests/tests/legacy_real_mode.rs")
        self.repo.commit()
        self.assertScope("legacy-only")

    def test_deletions(self):
        self.repo.git("rm", "-q", "x86jit-core/src/vm/nec.rs")
        self.repo.commit()
        self.assertScope("legacy-only")
        self.repo.git("rm", "-q", "x86jit-tests/tests/smc.rs")
        self.assertScope("shared")

    def test_uncommitted_and_untracked_changes_count(self):
        self.repo.write("x86jit-tests/tests/legacy_new.rs", "untracked\n")
        result = self.assertScope("legacy-only")
        self.assertIn("--test legacy_new ", "\n".join(commands_of(result)))
        self.repo.write("x86jit-core/src/vm.rs", "unstaged\n")
        self.assertScope("shared")
        self.repo.git("checkout", "--", "x86jit-core/src/vm.rs")
        self.repo.write("x86jit-core/src/legacy.rs", "staged\n")
        self.repo.git("add", "x86jit-core/src/legacy.rs")
        self.assertScope("shared")

    def test_a_change_reverted_in_the_working_tree_does_not_count(self):
        self.repo.write("x86jit-core/src/vm.rs", "edited\n")
        self.repo.commit()
        self.repo.write("x86jit-core/src/vm.rs", "original\n")
        self.assertScope("docs-only")

    def test_changes_on_the_base_after_the_fork_do_not_count(self):
        self.repo.write("backlog/docs/guide.md", "edited\n")
        self.repo.commit()
        self.repo.git("checkout", "-q", "main")
        self.repo.write("x86jit-core/src/vm.rs", "main moved on\n")
        self.repo.commit()
        self.repo.git("checkout", "-q", "work")
        self.assertScope("docs-only")

    def test_ignored_files_do_not_count(self):
        self.repo.write(".git/info/exclude", "target/\n")
        self.repo.write("target/debug/build.rs", "artifact\n")
        result = self.assertScope("docs-only")
        self.assertNotIn("target/", result.stdout)

    def test_run_keeps_going_after_a_failure_and_reports_it(self):
        self.repo.write("backlog/docs/guide.md", "edited\n")
        result = self.repo.scope("--run", "main", env={"FAIL_STEP": "fmt"})
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(self.repo.log.read_text().splitlines(), [
            "gate fmt",
            "gate guard",
            "nix develop --no-update-lock-file --command pre-commit run --files "
            "backlog/docs/guide.md",
        ])
        self.assertIn("1 of 3 commands failed:\n  bash scripts/ci-gate.sh fmt", result.stdout)

    def test_run_passes(self):
        self.repo.write("backlog/docs/guide.md", "edited\n")
        result = self.repo.scope("--run", "main")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("3 commands passed (scope docs-only)", result.stdout)

    def test_named_dev_shell(self):
        self.repo.write("backlog/docs/guide.md", "edited\n")
        result = self.repo.scope("main", env={"XJ_DEVSHELL": "no-freebsd"})
        self.assertIn("nix develop .#no-freebsd --no-update-lock-file --command pre-commit",
                      result.stdout)

    def test_scope_prints_only_the_scope(self):
        result = self.repo.scope("--scope", "main")
        self.assertEqual((result.returncode, result.stdout), (0, "none\n"), result.stderr)
        for path, scope in (("backlog/docs/guide.md", "docs-only"),
                            ("x86jit-core/src/vm/nec.rs", "legacy-only"),
                            ("x86jit-core/src/vm.rs", "shared")):
            with self.subTest(path=path):
                self.repo.write(path, "edited\n")
                result = self.repo.scope("--scope", "main")
                self.assertEqual((result.returncode, result.stdout), (0, f"{scope}\n"),
                                 result.stderr)
        self.assertFalse(self.repo.log.exists())

    def test_invalid_invocations(self):
        for args in ((), ("--run",), ("main", "extra"), ("--frobnicate",),
                     ("--scope",), ("--scope", "main", "extra"), ("--scope", "no-such-ref"),
                     ("--plan",), ("--plan", "everything"), ("--classify",),
                     ("no-such-ref",)):
            with self.subTest(args=args):
                result = self.repo.scope(*args)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertFalse(self.repo.log.exists())


class Plans(unittest.TestCase):
    def plan(self, scope, env=None):
        environment = {**os.environ, **(env or {})}
        if env is None:
            environment.pop("XJ_DEVSHELL", None)
        result = subprocess.run(["bash", str(SCRIPT), "--plan", scope], cwd=ROOT,
                                env=environment, text=True, capture_output=True,
                                check=True)
        return [line.strip() for line in commands_of(result)]

    def test_shared_plan_runs_every_gate_ci_runs_with_the_fast_tier(self):
        workflow = CI.read_text()
        plan = self.plan("shared")
        gates = re.findall(r"run: (bash scripts/ci-gate\.sh [^\n]+)", workflow)
        expected = {"bash scripts/ci-gate.sh fast"}
        full_tier = {"bash scripts/ci-gate.sh tests", "bash scripts/ci-gate.sh ratchet",
                     "bash scripts/ci-gate.sh release"}
        self.assertLessEqual(full_tier, set(gates))
        self.assertFalse(full_tier & set(plan), "the full tier runs only in CI")
        for gate in gates:
            if gate.endswith(("devshell", "corpus", "build-tests", "build-release")):
                continue  # CI setup and cache warming; the plan fetches the corpus only when absent
            if gate in full_tier:
                continue
            if '"$CROSS_TARGET"' in gate:
                expected |= {"bash scripts/ci-gate.sh cross aarch64-unknown-linux-gnu",
                             "bash scripts/ci-gate.sh cross x86_64-unknown-linux-gnu"}
            else:
                expected.add(gate)
        self.assertTrue(expected)
        self.assertLessEqual(expected, set(plan))
        self.assertNotIn("bash scripts/ci-gate.sh lane32", plan)
        script_tests = re.findall(r"python3 -B -m unittest discover -s scripts/tests -p \S+",
                                  workflow)
        self.assertTrue(script_tests)
        for command in script_tests:
            self.assertIn(command, plan)

    def test_shared_plan_without_freebsd_shell(self):
        plan = self.plan("shared", env={"XJ_DEVSHELL": "no-freebsd"})
        self.assertFalse([c for c in plan if "freebsd" in c])
        self.assertIn("bash scripts/ci-gate.sh cross x86_64-unknown-linux-gnu", plan)

    def test_legacy_plan_selects_the_legacy_tests_and_type_checks_both_linux_targets(self):
        plan = self.plan("legacy-only")
        tests = next(c for c in plan if "--test " in c)
        self.assertTrue(tests.startswith(
            "SS286_REQUIRED=1 nix develop --no-update-lock-file --command cargo nextest run "
            "--profile fast --locked --workspace --features x86jit-tests/unicorn --no-fail-fast --test "))
        names = re.findall(r"--test (\w+)", tests)
        self.assertEqual(names, sorted(set(names)))
        legacy = {p.stem for p in (ROOT / "x86jit-tests" / "tests").glob("legacy_*.rs")}
        for name in legacy | {"boundary", "environment_boundary", "run_steps", "irq_shadow",
                              "retired_count", "smc", "cross_region_ram",
                              "cross_region_host_fault", "application_codegen", "ss286",
                              "cf16"}:
            self.assertIn(name, names)
        for name in ("jit", "differential", "differential32", "fuzz", "mt", "corpus"):
            self.assertNotIn(name, names)
        self.assertIn("nix develop --no-update-lock-file --command cargo nextest run "
                      "--profile fast --locked -p x86jit-core -p x86jit-tests --features x86jit-tests/unicorn "
                      "--no-fail-fast --lib", plan)
        for target in ("aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu"):
            self.assertIn(f"bash scripts/ci-gate.sh cross {target}", plan)
        for gate in ("fmt", "guard", "clippy"):
            self.assertIn(f"bash scripts/ci-gate.sh {gate}", plan)

    def test_docs_plan(self):
        self.assertEqual(self.plan("docs-only"), [
            "bash scripts/ci-gate.sh fmt",
            "bash scripts/ci-gate.sh guard",
            "nix develop --no-update-lock-file --command pre-commit run --files "
            "<changed files>",
        ])


if __name__ == "__main__":
    unittest.main()
