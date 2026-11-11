"""Exercise CI dispatch and failure handling without compiling or fetching data."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


GATE = Path(__file__).resolve().parents[1] / "ci-gate.sh"
STUB = """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
Path(os.environ['CALL_FILE']).write_text(json.dumps({
    'args': sys.argv[1:],
    'corpus_required': os.environ.get('SS286_REQUIRED'),
    'backtrace': os.environ.get('RUST_BACKTRACE'),
}))
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

    def call(self):
        return json.loads(self.call_file.read_text())

    def test_full_gate_requires_corpus_and_excludes_only_robustness(self):
        result = self.run_gate("tests")
        self.assertEqual(result.returncode, 0, result.stderr)
        call = self.call()
        self.assertEqual(call["corpus_required"], "1")
        self.assertEqual(call["backtrace"], "1")
        self.assertEqual(call["args"], [
            "develop", "--no-update-lock-file", "--command", "cargo", "nextest",
            "run", "--locked", "--workspace", "--features", "x86jit-tests/unicorn",
            "--no-fail-fast", "--success-output", "final", "-E",
            "not binary(fuzz_robustness)",
        ])

    def test_other_gates_dispatch_without_running_heavy_jobs(self):
        expected = {
            "devshell": ["rustc", "--version"],
            "fmt": ["cargo", "fmt", "--all", "--check"],
            "clippy": ["cargo", "clippy", "--locked", "--workspace", "--all-targets",
                       "--all-features", "--", "-D", "warnings"],
            "deny": ["cargo", "deny", "--all-features", "check", "licenses", "bans", "sources"],
            "release": ["cargo", "nextest", "run", "--locked", "--release", "-p", "x86jit-tests",
                        "--no-fail-fast", "--success-output", "final", "-E",
                        "binary(jit) or binary(differential) or binary(corpus) or binary(smc) or "
                        "binary(mt) or binary(fault_landing_pad) or binary(region_fault_state) or "
                        "binary(rust_memory_faults)"],
            "lane32": ["cargo", "nextest", "run", "--locked", "-p", "x86jit-tests", "--features",
                       "unicorn", "--no-fail-fast", "--success-output", "final", "-E",
                       "binary(differential32) or (binary(fuzz) and "
                       "(test(jit_matches_interp_32) or test(unicorn_matches_interp_32)))"],
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
        for args in ((), ("unknown",), ("cross",), ("cross", "invalid"),
                     ("fmt", "extra"), ("cross", "x86_64-unknown-linux-gnu", "extra")):
            with self.subTest(args=args):
                result = self.run_gate(*args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("Usage:", result.stderr)
                self.assertFalse(self.call_file.exists())


if __name__ == "__main__":
    unittest.main()
