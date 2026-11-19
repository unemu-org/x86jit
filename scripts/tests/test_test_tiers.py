"""Keep the test tiers honest: the slow list, the fast profile and the ratchet of
scripts/test-tiers.py (backlog/docs/commands.md § Test tiers)."""

import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "test-tiers.py"
SPEC = importlib.util.spec_from_file_location("test_tiers", SCRIPT)
tiers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tiers)


def junit(cases):
    """A nextest-shaped JUnit report: {(binary-id, test): seconds}."""
    suites = {}
    for (binary, test), seconds in cases.items():
        suites.setdefault(binary, []).append(
            f'<testcase name="{test}" classname="{binary}" time="{seconds}"/>')
    body = "".join(f'<testsuite name="{binary}" tests="{len(rows)}">{"".join(rows)}</testsuite>'
                   for binary, rows in suites.items())
    return f'<?xml version="1.0"?><testsuites name="nextest-run">{body}</testsuites>\n'


class Ratchet(unittest.TestCase):
    """The ratchet against the committed slow list, on synthetic reports."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.entries = tiers.read_list()
        self.lane = "linux-x86_64"
        # A full-tier report: every listed test on this lane, slow, and fast ones.
        self.cases = {entry.key: 60.0 for entry in self.entries
                      if entry.applies_to(self.lane)}
        self.cases[("x86jit-tests::jit", "a_fast_test")] = 0.2
        self.cases[("x86jit-core", "lift::tests::another")] = 4.9

    def run_ratchet(self, cases, tier="full", lane=None):
        path = Path(self.temp.name) / "junit.xml"
        path.write_text(junit(cases))
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "ratchet", "--tier", tier,
             "--lane", lane or self.lane, str(path)],
            text=True, capture_output=True, check=False)
        return result

    def test_a_full_report_within_the_bound_passes(self):
        result = self.run_ratchet(self.cases)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ratchet passed", result.stdout)

    def test_a_planted_slow_test_fails(self):
        self.cases[("x86jit-tests::jit", "planted_slow_test")] = 12.5
        result = self.run_ratchet(self.cases)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("x86jit-tests::jit planted_slow_test", result.stdout)
        self.assertIn("over the 10 s bound", result.stdout)

    def test_a_planted_slow_test_fails_on_the_fast_tier_too(self):
        cases = {("x86jit-tests::jit", "a_fast_test"): 0.2,
                 ("x86jit-tests::jit", "planted_slow_test"): 12.5}
        result = self.run_ratchet(cases, tier="fast")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("planted_slow_test", result.stdout)
        self.assertNotIn("gone", result.stdout)

    def test_a_test_between_admission_and_bound_is_a_candidate(self):
        self.cases[("x86jit-tests::jit", "getting_slower")] = 7.0
        result = self.run_ratchet(self.cases)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("candidate", result.stdout)
        self.assertIn("getting_slower", result.stdout)

    def test_a_listed_test_that_is_gone_fails(self):
        gone = next(entry for entry in self.entries if entry.only is None)
        del self.cases[gone.key]
        result = self.run_ratchet(self.cases)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn(f"gone   {gone.binary} {gone.test}", result.stdout)

    def test_a_lane_restricted_entry_is_not_required_elsewhere(self):
        restricted = [entry for entry in self.entries if entry.only]
        self.assertTrue(restricted)
        for lane in ("linux-x86_64", "linux-aarch64", "macos-aarch64"):
            with self.subTest(lane=lane):
                cases = {entry.key: 60.0 for entry in self.entries if entry.applies_to(lane)}
                result = self.run_ratchet(cases, lane=lane)
                self.assertEqual(result.returncode, 0, result.stdout)

    def test_the_heavy_binary_is_never_a_fast_tier_test(self):
        self.cases[("x86jit-tests::fuzz_robustness", "long_campaign")] = 400.0
        result = self.run_ratchet(self.cases)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_an_unreadable_report_is_an_input_error(self):
        path = Path(self.temp.name) / "junit.xml"
        path.write_text("<testsuites>")
        result = subprocess.run([sys.executable, "-B", str(SCRIPT), "ratchet", str(path)],
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 2)


class SlowList(unittest.TestCase):
    def test_the_fast_profile_matches_the_slow_list(self):
        result = subprocess.run([sys.executable, "-B", str(SCRIPT), "sync", "--check"],
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_every_listed_test_is_excluded_by_the_fast_filter(self):
        expression = tiers.fast_filter(tiers.read_list())
        for entry in tiers.read_list():
            with self.subTest(test=entry.test):
                self.assertRegex(expression, re.escape(f"binary_id(={entry.binary}) & (")
                                 + r"[^\n]*" + re.escape(f"test(={entry.test})"))
        self.assertTrue(expression.startswith("not (binary(fuzz_robustness))"))

    def test_lane_globs_name_known_lanes(self):
        lanes = ("linux-x86_64", "linux-aarch64", "macos-aarch64")
        for entry in tiers.read_list():
            if entry.only:
                with self.subTest(test=entry.test):
                    self.assertTrue(any(entry.applies_to(lane) for lane in lanes))

    def test_listed_test_files_exist(self):
        # An integration-test binary is a file or a directory with main.rs, whose
        # modules are files beside it; a renamed file shows up here before a CI
        # run reports the entries as gone.
        for entry in tiers.read_list():
            crate, _, binary = entry.binary.partition("::")
            self.assertTrue((ROOT / crate).is_dir(), entry.binary)
            if binary and not binary.startswith("bin/"):
                tests = ROOT / crate / "tests"
                with self.subTest(binary=entry.binary, test=entry.test):
                    if (tests / binary / "main.rs").is_file():
                        module = entry.test.split("::", 1)[0]
                        self.assertTrue((tests / binary / f"{module}.rs").is_file())
                    else:
                        self.assertTrue((tests / f"{binary}.rs").is_file())

    def test_malformed_and_duplicate_entries_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "slow-tests.txt"
            for text in ("x86jit-core\n", "x86jit-core a b c\n", "x86jit-core a only=\n",
                         "x86jit-core a\nx86jit-core a\n", "x86jit-core a-b\n"):
                with self.subTest(text=text):
                    path.write_text(text)
                    with self.assertRaises(tiers.ListError):
                        tiers.read_list(path)
            path.write_text("# comment\n\nx86jit-core a::b only=*-aarch64  # why\n")
            (entry,) = tiers.read_list(path)
            self.assertEqual((entry.binary, entry.test, entry.only),
                             ("x86jit-core", "a::b", "*-aarch64"))

    def test_heavy_profile_selects_the_robustness_binary_and_the_soaks(self):
        text = tiers.NEXTEST.read_text()
        heavy = text.split("[profile.heavy]", 1)[1].split("'''", 2)[1]
        self.assertIn("binary(fuzz_robustness)", heavy)
        self.assertIn("test(=dense_reclamation_soak)", heavy)
        self.assertIn("test(=hundred_thousand_units_have_bounded_mappings_and_residency)",
                      heavy)


if __name__ == "__main__":
    unittest.main()
