"""Exercise the perf campaign's parsing, statistics and verdict without running a benchmark."""

import contextlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import perf_campaign as pc  # noqa: E402

HEADER = ("name,path,steps,retired_instructions,warmup_steps,rounds,instructions_per_step,"
          "instructions_min,instructions_max,ns_per_step,ns_min,ns_max\n")


def loops_csv(workload, values, counts=("1000", "2000", "100")):
    rows = [HEADER]
    for path, value in values.items():
        rows.append(f"{workload},{path},{counts[0]},{counts[1]},{counts[2]},3,{value:.3f},"
                    f"{value:.3f},{value:.3f},1.0,1.0,1.0\n")
    return "".join(rows)


class RawDir:
    def __init__(self, root):
        self.root = root

    def loops(self, side, workload, index, values, counts=("1000", "2000", "100")):
        Path(self.root, f"{side}-{workload}-{index}.csv").write_text(
            loops_csv(workload, values, counts))

    def store(self, side, name, line):
        Path(self.root, f"store-{side}{name}.csv").write_text(line)


class ParsingTests(unittest.TestCase):
    def test_loops_csv(self):
        rows = pc.parse_loops_csv(loops_csv("alu", {"application-real16": 355.5,
                                                    "machine-i8088": 401.25}))
        self.assertEqual([(r["workload"], r["path"], r["value"]) for r in rows],
                         [("alu", "application-real16", 355.5), ("alu", "machine-i8088", 401.25)])
        self.assertEqual(rows[0]["counts"], ("1000", "2000", "100"))

    def test_store_csv_takes_the_counted_column_only(self):
        text = "8,single,off,3200000,1.234,298.053\n32,crossing,on,3200000,2.0\nnoise\n"
        self.assertEqual(pc.parse_store_csv(text), [((8, "single", "off"), 298.053)])

    def test_load_side_keeps_sides_apart(self):
        with tempfile.TemporaryDirectory() as d:
            raw = RawDir(d)
            raw.loops("base", "data-next-to-code", 1, {"legacy-i286": 10.0})
            raw.loops("base", "data-next-to-code", 2, {"legacy-i286": 11.0})
            raw.loops("base0", "data-next-to-code", 1, {"legacy-i286": 99.0})
            raw.loops("cand", "data-next-to-code", 1, {"legacy-i286": 12.0})
            raw.store("base", "", "8,single,off,1,1.0,5.0\n8,single,off,1,1.0,6.0\n")
            raw.store("base", "-8-single-off-3", "8,single,off,1,1.0,7.0\n")
            raw.store("base0", "", "8,single,off,1,1.0,99.0\n")
            cells, counts, store = pc.load_side(d, "base")
        self.assertEqual(cells, {("data-next-to-code", "legacy-i286"): [10.0, 11.0]})
        self.assertEqual(counts[("data-next-to-code", "legacy-i286")], {("1000", "2000", "100")})
        self.assertEqual(sorted(store["8 single off"]), [5.0, 6.0, 7.0])


class StatisticsTests(unittest.TestCase):
    def test_one_mode(self):
        self.assertEqual(pc.modes([100.0, 100.2, 100.1]), [[100.0, 100.1, 100.2]])

    def test_two_modes_split_at_the_largest_gap(self):
        self.assertEqual(pc.modes([100.0, 100.05, 100.6, 100.62, 100.01]),
                         [[100.0, 100.01, 100.05], [100.6, 100.62]])

    def test_close_clusters_are_one_mode(self):
        self.assertEqual(len(pc.modes([100.0, 100.0, 100.4, 100.4])), 1)

    def test_unimodal_cell_gates_on_the_median(self):
        d = pc.compare_cell([100.0, 100.1, 100.2], [100.5, 100.7, 100.6])
        self.assertFalse(d["bimodal"])
        self.assertAlmostEqual(d["gate_delta"], (100.6 / 100.1 - 1) * 100)
        self.assertAlmostEqual(d["spread_base"], 0.2 / 100.1 * 100)

    def test_bimodal_cell_compares_like_modes(self):
        base = [100.0] * 4 + [100.6] * 3
        cand = [100.1] * 3 + [100.7] * 4
        d = pc.compare_cell(base, cand)
        self.assertTrue(d["bimodal"])
        # The median of processes crosses modes; the like-mode pairs do not.
        self.assertGreater(d["median_delta"], 0.5)
        self.assertEqual([p[:4] for p in d["pairs"]], [[100.0, 100.1, 4, 3], [100.6, 100.7, 3, 4]])
        self.assertAlmostEqual(d["gate_delta"], max((100.1 / 100 - 1), (100.7 / 100.6 - 1)) * 100)
        self.assertEqual(d["unmatched"], [])

    def test_unmatched_mode_does_not_gate(self):
        base = [100.0] * 7
        cand = [100.05] * 5 + [100.8] * 2
        d = pc.compare_cell(base, cand)
        self.assertTrue(d["bimodal"])
        self.assertEqual(len(d["pairs"]), 1)
        self.assertAlmostEqual(d["gate_delta"], 0.05)
        self.assertEqual(d["unmatched"][0][:3], ["cand", 100.8, 2])

    def test_path_means_are_geometric(self):
        cells = {
            "alu|legacy-i286": pc.compare_cell([100.0], [101.0]),
            "store|legacy-i286": pc.compare_cell([400.0], [396.0]),
            "alu|legacy-i386": pc.compare_cell([10.0], [10.0]),
        }
        m = pc.path_means(cells)
        self.assertEqual(m["legacy-i286"]["cells"], 2)
        self.assertAlmostEqual(m["legacy-i286"]["base"], 200.0)
        self.assertAlmostEqual(m["legacy-i286"]["delta"], (math.sqrt(1.01 * 0.99) - 1) * 100)
        self.assertAlmostEqual(m["legacy-i386"]["delta"], 0.0)

    def test_percentile(self):
        self.assertEqual(pc.percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertEqual(pc.percentile([0.0, 10.0], 0.9), 9.0)


class VerdictTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.raw = RawDir(self.temp.name)

    def fill(self, cand_alu=100.0, base_paths=("legacy-i286",), cand_paths=("legacy-i286",)):
        for i in (1, 2, 3):
            self.raw.loops("base", "alu", i, {p: 100.0 for p in base_paths})
            self.raw.loops("cand", "alu", i, {p: cand_alu for p in cand_paths})
            self.raw.loops("base", "store", i, {p: 200.0 for p in base_paths})
            self.raw.loops("cand", "store", i, {p: 200.0 for p in cand_paths})
        self.raw.store("base", "-8-single-off-1", "8,single,off,1,1.0,300.0\n")
        self.raw.store("cand", "-8-single-off-1", "8,single,off,1,1.0,300.0\n")

    def test_pass(self):
        self.fill(cand_alu=100.4)
        r = pc.analyze(self.temp.name)
        self.assertEqual((r["verdict"], r["misses"]), ("PASS", []))
        self.assertEqual(pc.exit_status(r), 0)
        self.assertIn("VERDICT: PASS", pc.report(r))

    def test_a_cell_over_the_gate_misses(self):
        self.fill(cand_alu=100.6)
        r = pc.analyze(self.temp.name)
        self.assertEqual((r["verdict"], r["misses"]), ("MISS", ["alu|legacy-i286"]))
        self.assertEqual(pc.exit_status(r), 1)
        self.assertIn("MISS alu|legacy-i286 +0.600%", pc.report(r))

    def test_the_gate_is_a_parameter(self):
        self.fill(cand_alu=100.6)
        self.assertEqual(pc.analyze(self.temp.name, gate=1.0)["verdict"], "PASS")

    def test_store_probe_gates_too(self):
        self.fill()
        self.raw.store("cand", "-8-single-off-1", "8,single,off,1,1.0,302.0\n")
        r = pc.analyze(self.temp.name)
        self.assertEqual((r["verdict"], r["store_misses"]), ("MISS", ["8 single off"]))

    def test_different_counts_are_a_problem(self):
        self.fill()
        self.raw.loops("cand", "alu", 4, {"legacy-i286": 100.0}, counts=("999", "2000", "100"))
        r = pc.analyze(self.temp.name)
        self.assertEqual(r["verdict"], "PROBLEM")
        self.assertEqual(pc.exit_status(r), 2)

    def test_a_zero_counter_is_a_problem(self):
        self.fill(cand_alu=0.0)
        self.assertEqual(pc.analyze(self.temp.name)["verdict"], "PROBLEM")

    def test_a_missing_cell_is_a_problem(self):
        self.fill()
        self.raw.loops("cand", "push-pop", 1, {"legacy-i286": 1.0})
        r = pc.analyze(self.temp.name)
        self.assertEqual(r["verdict"], "PROBLEM")
        self.assertIn("cell push-pop legacy-i286 missing on one side", r["problems"])

    def test_a_path_only_on_one_side_is_a_note(self):
        self.fill(cand_paths=("legacy-i286", "machine-nec-v20"))
        r = pc.analyze(self.temp.name)
        self.assertEqual(r["verdict"], "PASS")
        self.assertEqual(r["notes"], ["path machine-nec-v20 only on the candidate"])
        self.assertEqual(set(r["paths"]), {"legacy-i286"})

    def test_bimodal_cell_with_few_processes_is_flagged_for_top_up(self):
        for i, (b, c) in enumerate([(100.0, 100.0), (100.0, 100.0), (100.0, 100.8)], 1):
            self.raw.loops("base", "smc-next-64", i, {"machine-i8088": b})
            self.raw.loops("cand", "smc-next-64", i, {"machine-i8088": c})
        r = pc.analyze(self.temp.name)
        self.assertEqual(r["bimodal"], ["smc-next-64|machine-i8088"])
        self.assertEqual(r["bimodal_short"], ["smc-next-64|machine-i8088"])
        self.assertEqual(r["verdict"], "PASS")  # the unmatched upper mode does not gate
        self.assertIn("FEWER THAN 7 PROCESSES PER SIDE", pc.report(r))

    def test_cells_csv_and_json_round_trip(self):
        self.fill(cand_alu=100.2)
        r = pc.analyze(self.temp.name)
        self.assertEqual(json.loads(json.dumps(r))["verdict"], "PASS")
        lines = pc.cells_csv(r).splitlines()
        self.assertEqual(lines[0].split(",")[:3], ["workload", "path", "base_median"])
        self.assertEqual(len(lines), 3)


class PlanTests(unittest.TestCase):
    def test_sides_alternate_and_divisors_apply(self):
        tasks = pc.plan([("base", "B"), ("cand", "C")], [1, 2], workloads=["alu", "straight-line"],
                        store=False)
        order = [(t["index"], t["workload"], t["side"], t["divisor"]) for t in tasks]
        self.assertEqual(order, [
            (1, "alu", "base", 8), (1, "alu", "cand", 8),
            (1, "straight-line", "base", 1), (1, "straight-line", "cand", 1),
            (2, "alu", "cand", 8), (2, "alu", "base", 8),
            (2, "straight-line", "cand", 1), (2, "straight-line", "base", 1),
        ])

    def test_run_task_passes_divisor_and_rounds(self):
        calls = []
        real = pc.run_process
        pc.run_process = lambda argv, out, err, aslr=False: calls.append((argv, out, aslr)) or 0
        try:
            for w, d in (("alu", 8), ("straight-line", 1)):
                pc.run_task({"kind": "loops", "side": "base", "bin": "/b", "workload": w,
                             "index": 2, "divisor": d}, "/raw", aslr=False)
            pc.run_task({"kind": "store", "side": "cand", "bin": "/c", "cell": (8, "single", "on"),
                         "index": 1}, "/raw", aslr=False)
        finally:
            pc.run_process = real
        self.assertEqual(calls, [
            (["/b/real16_loops", "8", "3", "alu"], "/raw/base-alu-2.csv", False),
            (["/b/real16_loops", "1", "7", "straight-line"], "/raw/base-straight-line-2.csv", False),
            (["/c/interpreter_store_probe", "8", "single", "on", "200000"],
             "/raw/store-cand-8-single-on-1.csv", False),
        ])

    def test_store_cells_are_planned_per_index(self):
        tasks = pc.plan([("base", "B"), ("cand", "C")], [1], workloads=[], store=True)
        self.assertEqual(len(tasks), 2 * len(pc.STORE_CELLS))

    def test_linux_turns_randomization_off_with_setarch(self):
        self.assertEqual(pc.aslr_prefix("Linux", "aarch64"), ["setarch", "aarch64", "-R"])
        self.assertEqual(pc.aslr_prefix("Darwin"), [])


class RecordTests(unittest.TestCase):
    def test_record_delta_uses_the_lower_mode(self):
        rec = pc.compare_cell([100.0] * 4 + [100.6] * 3, [100.2] * 4 + [100.9] * 3)
        self.assertAlmostEqual(pc.record_delta(rec), 0.2)
        self.assertAlmostEqual(pc.record_delta(pc.compare_cell([100.0], [100.3])), 0.3)

    def test_compare_record(self):
        def result(deltas):
            return {"cells": {k: pc.compare_cell([100.0], [100.0 + x]) for k, x in deltas.items()},
                    "bimodal": []}
        replay = result({"a|p": 0.6, "b|p": 0.1, "c|p": 0.45})
        record = result({"a|p": 0.7, "b|p": 0.5, "c|p": 0.55})
        c = pc.compare_record(replay, record, gate=0.5, tolerance=0.25)
        self.assertEqual(c["over_both"], ["a|p"])
        self.assertEqual(c["over_replay_only"], [])
        self.assertEqual(c["over_record_only"], ["c|p"])
        self.assertEqual([r[0] for r in c["outside"]], ["b|p"])
        self.assertAlmostEqual(c["diff_max"], 0.4)


def profile_result(cells, store=None, problems=()):
    """An analyze()-shaped result: {cell: change %}, {store cell: change %}."""
    return {
        "cells": {k: pc.compare_cell([100.0], [100.0 + x]) for k, x in cells.items()},
        "store": {k: {"delta": x} for k, x in (store or {}).items()},
        "problems": list(problems),
    }


class CombineTests(unittest.TestCase):
    def combine(self, release, perf, rstore=None, pstore=None, **kw):
        return pc.combine({"release": profile_result(release, rstore, **kw),
                           "perf": profile_result(perf, pstore)})

    def test_over_in_both_profiles_blocks(self):
        c = self.combine({"alu|p": 0.8, "load|p": 0.1}, {"alu|p": 0.6, "load|p": 0.2})
        self.assertEqual(c["verdict"], "BLOCK")
        self.assertEqual(pc.combined_status(c), 1)
        self.assertEqual([e["cell"] for e in c["block"]], ["alu|p"])
        self.assertEqual(c["block"][0]["over"], ["release", "perf"])
        self.assertEqual(c["drift"], [])

    def test_over_in_one_profile_is_drift(self):
        c = self.combine({"alu|p": 0.7, "load|p": 0.1}, {"alu|p": 0.2, "load|p": 2.7})
        self.assertEqual((c["verdict"], pc.combined_status(c)), ("DRIFT", 3))
        self.assertEqual([(e["cell"], e["over"]) for e in c["drift"]],
                         [("load|p", ["perf"]), ("alu|p", ["release"])])
        self.assertAlmostEqual(c["drift"][0]["deltas"]["perf"], 2.7)
        text = pc.combined_report(c)
        self.assertIn("DRIFT (cells): 2", text)
        self.assertIn("load|p: over in perf (release +0.100%, perf +2.700%)", text)
        self.assertIn("COMBINED VERDICT: DRIFT (exit 3)", text)

    def test_a_block_outranks_drift(self):
        c = self.combine({"a|p": 0.6, "b|p": 0.6}, {"a|p": 0.6, "b|p": 0.4})
        self.assertEqual(c["verdict"], "BLOCK")
        self.assertEqual(([e["cell"] for e in c["block"]], [e["cell"] for e in c["drift"]]),
                         (["a|p"], ["b|p"]))

    def test_over_in_neither_passes(self):
        c = self.combine({"alu|p": 0.5, "load|p": -1.0}, {"alu|p": 0.49, "load|p": 0.3})
        self.assertEqual((c["verdict"], pc.combined_status(c)), ("PASS", 0))

    def test_store_probe_follows_the_same_rule(self):
        base = {"alu|p": 0.0}
        c = self.combine(base, base, {"8 single off": 0.6, "32 single on": 0.7},
                         {"8 single off": 0.9, "32 single on": 0.1})
        self.assertEqual(c["verdict"], "BLOCK")
        self.assertEqual([e["cell"] for e in c["store_block"]], ["8 single off"])
        self.assertEqual([e["cell"] for e in c["store_drift"]], ["32 single on"])
        self.assertIn("store 8 single off: over in release+perf", pc.combined_report(c))
        c = self.combine(base, base, {"8 single off": 0.6}, {"8 single off": 0.2})
        self.assertEqual(c["verdict"], "DRIFT")
        c = self.combine(base, base, {"8 single off": 0.2}, {"8 single off": 0.2})
        self.assertEqual(c["verdict"], "PASS")

    def test_a_problem_in_either_profile_wins(self):
        c = self.combine({"alu|p": 0.9}, {"alu|p": 0.9}, problems=["counts differ in alu p"])
        self.assertEqual((c["verdict"], pc.combined_status(c)), ("PROBLEM", 2))
        self.assertEqual(c["problems"], ["release: counts differ in alu p"])

    def test_a_cell_missing_in_one_profile_is_a_problem(self):
        c = self.combine({"alu|p": 0.0, "load|p": 0.0}, {"alu|p": 0.0})
        self.assertEqual(c["verdict"], "PROBLEM")

    def test_bimodal_fallback_is_applied_per_profile_first(self):
        # Two modes in release: like modes pass even though the median of processes is over.
        bimodal = pc.compare_cell([100.0] * 4 + [100.6] * 3, [100.1] * 3 + [100.7] * 4)
        self.assertGreater(bimodal["median_delta"], 0.5)
        release = {"cells": {"smc|p": bimodal}, "store": {}, "problems": []}
        c = pc.combine({"release": release, "perf": profile_result({"smc|p": 0.8})})
        self.assertEqual(c["verdict"], "DRIFT")
        self.assertEqual(c["drift"][0]["over"], ["perf"])

    def test_combine_command_reads_analyses_and_exits_with_the_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name, x in (("release", 0.7), ("perf", 0.8)):
                path = Path(d, f"{name}.json")
                path.write_text(json.dumps(profile_result({"alu|p": x})))
                paths.append(f"{name}={path}")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pc.main(["combine", "--out", d] + paths), 1)
            self.assertEqual(json.loads(Path(d, "combined.json").read_text())["verdict"], "BLOCK")
            self.assertIn("COMBINED VERDICT: BLOCK", Path(d, "combined.txt").read_text())


class SnapshotTests(unittest.TestCase):
    def test_snapshot_copies_tracked_and_untracked_files_only(self):
        with tempfile.TemporaryDirectory() as d:
            tree, dest = Path(d, "tree"), Path(d, "dest")
            tree.mkdir()
            env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
            git = ["git", "-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
            subprocess.run(git + ["init", "-q"], check=True, env=env)
            Path(tree, ".gitignore").write_text("target/\n")
            Path(tree, "a.rs").write_text("tracked")
            Path(tree, "gone.rs").write_text("deleted later")
            subprocess.run(git + ["add", "."], check=True, env=env)
            subprocess.run(git + ["commit", "-q", "-m", "x"], check=True, env=env)
            Path(tree, "a.rs").write_text("edited")
            Path(tree, "gone.rs").unlink()
            Path(tree, "new.rs").write_text("untracked")
            Path(tree, "target").mkdir()
            Path(tree, "target", "big").write_text("ignored")
            self.assertEqual(pc.main(["snapshot", str(tree), str(dest)]), 0)
            self.assertEqual(sorted(p.name for p in dest.iterdir()), [".gitignore", "a.rs", "new.rs"])
            self.assertEqual(Path(dest, "a.rs").read_text(), "edited")


SCRIPTS = Path(__file__).resolve().parents[1]

# Stand-ins for cargo and rustc: the fake cargo records its target directory, its jobs
# and its start and end times, waits a second so that builds run at once can be seen to
# overlap, and writes the two examples where cargo would.
FAKE_CARGO = """#!/bin/sh
profile=
while [ $# -gt 0 ]; do
  [ "$1" = --profile ] && profile=$2
  shift
done
start=$(python3 -c 'import time; print(time.time())')
sleep 1
mkdir -p "$CARGO_TARGET_DIR/$profile/examples"
for n in real16_loops interpreter_store_probe; do
  printf '#!/bin/sh\\n' > "$CARGO_TARGET_DIR/$profile/examples/$n"
  chmod +x "$CARGO_TARGET_DIR/$profile/examples/$n"
done
end=$(python3 -c 'import time; print(time.time())')
echo "$profile $CARGO_TARGET_DIR $CARGO_BUILD_JOBS $start $end" >> "$FAKE_CARGO_LOG"
"""
FAKE_RUSTC = """#!/bin/sh
echo "rustc 0.0.0-fake"
"""
# Benchmarks that print the same counts and values every run.
FAKE_LOOPS = """#!/bin/sh
echo "%s"
echo "$3,legacy-i286,1000,2000,100,3,100.000,100.000,100.000,1.0,1.0,1.0"
""" % HEADER.strip()
FAKE_STORE = """#!/bin/sh
echo "$1,$2,$3,1,1.0,300.0"
"""


def write_exe(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


class ShellTests(unittest.TestCase):
    """scripts/perf-campaign.sh's builds and its measure-only mode, with fake tools."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        d = Path(self.temp.name)
        self.repo, self.cache, self.tools = d / "repo", d / "cache", d / "tools"
        self.log = d / "cargo.log"
        self.env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
                        FAKE_CARGO_LOG=str(self.log),
                        PATH=f"{self.tools}{os.pathsep}{os.environ['PATH']}")
        self.env.pop("X86JIT_PERF_CACHE", None)
        write_exe(self.tools / "cargo", FAKE_CARGO)
        write_exe(self.tools / "rustc", FAKE_RUSTC)

    def make_repo(self, commits=2):
        """A repository holding the campaign scripts; returns its commits."""
        git = ["git", "-C", str(self.repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
               "-c", "commit.gpgsign=false"]
        (self.repo / "scripts").mkdir(parents=True)
        for name in ("perf-campaign.sh", "perf_campaign.py"):
            (self.repo / "scripts" / name).write_bytes((SCRIPTS / name).read_bytes())
        probe = self.repo / "x86jit-bench" / "examples" / "interpreter_store_probe.rs"
        probe.parent.mkdir(parents=True)
        probe.write_text("// instructions_per_store\n")
        subprocess.run(git + ["init", "-q"], check=True, env=self.env)
        shas = []
        for i in range(commits):
            (self.repo / "Cargo.toml").write_text(f"[profile.perf]\n# {i}\n")
            subprocess.run(git + ["add", "."], check=True, env=self.env)
            subprocess.run(git + ["commit", "-q", "-m", str(i)], check=True, env=self.env)
            shas.append(subprocess.run(git + ["rev-parse", "HEAD"], check=True, env=self.env,
                                       capture_output=True, text=True).stdout.strip())
        return shas

    def campaign(self, *args):
        return subprocess.run(["bash", str(self.repo / "scripts" / "perf-campaign.sh"), *args],
                              env=self.env, capture_output=True, text=True, timeout=120)

    def builds(self):
        if not self.log.exists():
            return []
        return [line.split() for line in self.log.read_text().splitlines()]

    def test_builds_run_at_once_each_in_its_own_target_directory(self):
        base, cand = self.make_repo()
        (self.cache / "target").mkdir(parents=True)  # the layout before the slots
        r = self.campaign("--build-only", "--cache", str(self.cache), base, cand)
        self.assertEqual(r.returncode, 0, r.stderr)
        builds = self.builds()
        self.assertEqual(sorted(b[0] for b in builds), ["perf", "perf", "release", "release"])
        self.assertEqual(sorted(Path(b[1]).name for b in builds),
                         ["target-base-perf", "target-base-release",
                          "target-candidate-perf", "target-candidate-release"])
        cpus = int(subprocess.run(["getconf", "_NPROCESSORS_ONLN"], capture_output=True,
                                  text=True).stdout)
        self.assertEqual({b[2] for b in builds}, {str((cpus + 3) // 4)})
        # Every build started before the first one ended.
        self.assertLess(max(float(b[3]) for b in builds), min(float(b[4]) for b in builds))
        self.assertFalse((self.cache / "target").exists())
        for sha in (base, cand):
            for profile in ("release", "perf"):
                [d] = self.cache.glob(f"bin/{sha}-{profile}-*")
                meta = (d / "meta.txt").read_text()
                self.assertIn(f"source={sha}\nprofile={profile}\n", meta)
        self.assertFalse((self.cache / "lock").exists())

        # A second campaign of the same pair takes all four from the cache.
        r = self.campaign("--build-only", "--cache", str(self.cache), base, cand)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(self.builds()), 4)
        self.assertEqual(r.stderr.count("from the cache"), 4)

    def test_the_same_commit_on_both_sides_builds_once_per_profile(self):
        [sha] = self.make_repo(commits=1)
        r = self.campaign("--build-only", "--cache", str(self.cache), "--jobs", "3", sha, sha)
        self.assertEqual(r.returncode, 0, r.stderr)
        builds = self.builds()
        self.assertEqual(sorted(b[0] for b in builds), ["perf", "release"])
        self.assertEqual({b[2] for b in builds}, {"3"})
        lines = r.stdout.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0].split()[-1], lines[1].split()[-1])

    def test_a_failed_build_is_a_problem_and_releases_the_lock(self):
        base, cand = self.make_repo()
        write_exe(self.tools / "cargo", "#!/bin/sh\necho broken >&2\nexit 101\n")
        r = self.campaign("--build-only", "--cache", str(self.cache), base, cand)
        self.assertEqual(r.returncode, 2)
        self.assertIn("4 build(s) failed", r.stderr)
        self.assertFalse((self.cache / "lock").exists())

    def prebuilt(self, base="aaaa1111", cand="bbbb2222"):
        root = Path(self.temp.name, "prebuilt")
        for side, sha in (("base", base), ("candidate", cand)):
            for profile in ("release", "perf"):
                d = root / f"{side}-{profile}"
                write_exe(d / "real16_loops", FAKE_LOOPS)
                write_exe(d / "interpreter_store_probe", FAKE_STORE)
                (d / "meta.txt").write_text(f"source={sha}\nprofile={profile}\n")
        return root

    def test_prebuilt_measures_without_building(self):
        self.make_repo(commits=1)
        write_exe(self.tools / "cargo", "#!/bin/sh\nexit 99\n")
        out = Path(self.temp.name, "out")
        r = self.campaign("--prebuilt", str(self.prebuilt()), "--out", str(out),
                          "--processes", "1", "--parallel", "8", "aaaa1111", "bbbb2222")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(json.loads(Path(out, "combined.json").read_text())["verdict"], "PASS")
        self.assertIn("aaaa1111", Path(out, "release", "report.txt").read_text())

    def test_prebuilt_binaries_of_another_commit_are_a_problem(self):
        self.make_repo(commits=1)
        r = self.campaign("--prebuilt", str(self.prebuilt(cand="cccc3333")),
                          "--out", str(Path(self.temp.name, "out")), "aaaa1111", "bbbb2222")
        self.assertEqual(r.returncode, 2)
        self.assertIn("built from 'cccc3333', not 'bbbb2222'", r.stderr)


class SummaryTests(unittest.TestCase):
    def write(self, d, release, perf, rstore=None, pstore=None):
        results = {}
        for name, cells, store in (("release", release, rstore), ("perf", perf, pstore)):
            r = profile_result(cells, store)
            r["paths"] = pc.path_means(r["cells"])
            r["meta"] = {"base": "aaaa1111 (/tmp/bin/a)", "candidate": "bbbb2222 (/tmp/bin/b)",
                         "processes": 3, "aslr": "off"}
            os.makedirs(Path(d, name))
            Path(d, name, "analysis.json").write_text(json.dumps(r))
            Path(d, name, "report.txt").write_text(f"VERDICT: {name}\n")
            results[name] = r
        Path(d, "combined.json").write_text(json.dumps(pc.combine(results)))

    def test_summary_lists_blocking_and_drift_cells_and_path_means(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(d, {"alu|legacy-i286": 0.8, "load|legacy-i286": 0.7},
                       {"alu|legacy-i286": 0.6, "load|legacy-i286": 0.1},
                       {"8 single off": 0.9}, {"8 single off": 0.1})
            text = pc.summary_markdown(d)
        self.assertIn("## Perf gate: BLOCK (exit 1)", text)
        self.assertIn("Base `aaaa1111`, candidate `bbbb2222`", text)
        self.assertIn("| BLOCK | `alu` I286 | release+perf | +0.800% | +0.600% |", text)
        self.assertIn("| DRIFT | `load` I286 | release | +0.700% | +0.100% |", text)
        self.assertIn("| DRIFT | store 8 single off | release | +0.900% | +0.100% |", text)
        self.assertIn("| I286 | +0.750% | +0.350% |", text)
        self.assertIn("<details><summary>perf profile report</summary>", text)
        self.assertNotIn("/tmp/bin", text)

    def test_summary_of_a_pass_and_of_a_run_without_a_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(d, {"alu|legacy-i286": 0.1}, {"alu|legacy-i286": 0.2})
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(pc.main(["summary", d]), 0)
            self.assertIn("## Perf gate: PASS (exit 0)", out.getvalue())
            self.assertIn("No cell is over the gate in any profile.", out.getvalue())
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("## Perf gate: no verdict", pc.summary_markdown(d))


class CounterCheckTests(unittest.TestCase):
    def check(self, system, **patches):
        saved = {"system": pc.platform.system}
        pc.platform.system = lambda: system
        for name, value in patches.items():
            saved[name] = getattr(pc, name)
            setattr(pc, name, value)
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out, \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                status = pc.main(["counter-check"])
        finally:
            pc.platform.system = saved.pop("system")
            for name, value in saved.items():
                setattr(pc, name, value)
        return status, out.getvalue(), err.getvalue()

    def test_no_counter_outside_macos_and_linux(self):
        status, _, err = self.check("FreeBSD")
        self.assertEqual(status, 2)
        self.assertIn("no host instruction counter on FreeBSD", err)

    def test_linux_without_perf_events_is_a_problem(self):
        def refuse(machine=None):
            raise OSError(2, "perf_event_open(PERF_COUNT_HW_INSTRUCTIONS): No such file")
        status, out, err = self.check("Linux", perf_instruction_counter=refuse,
                                      perf_event_paranoid=lambda: "4")
        self.assertEqual(status, 2)
        self.assertIn("kernel.perf_event_paranoid 4", out)
        self.assertIn("No such file", err)

    def test_linux_counter_that_does_not_move_is_a_problem(self):
        r, w = os.pipe()
        os.close(w)
        reads = iter([5, 5])
        status, _, err = self.check("Linux", perf_instruction_counter=lambda: r,
                                    read_counter=lambda fd: next(reads),
                                    perf_event_paranoid=lambda: "2")
        self.assertEqual(status, 2)
        self.assertIn("does not count", err)

    def test_linux_counter_that_counts_passes(self):
        r, w = os.pipe()
        os.close(w)
        reads = iter([5, 9_000_000])
        status, out, _ = self.check("Linux", perf_instruction_counter=lambda: r,
                                    read_counter=lambda fd: next(reads),
                                    perf_event_paranoid=lambda: "2")
        self.assertEqual(status, 0)
        self.assertIn("perf_event_open instructions 5 -> 9000000", out)

    def test_the_counter_is_read_from_the_v4_layout(self):
        # Apple SDK sys/resource.h, rusage_info_v4: 16-byte uuid, 29 u64s, ri_instructions.
        self.assertEqual(pc.RUSAGE_V4_INSTRUCTIONS * 8, 16 + 29 * 8)

    def test_the_perf_attr_is_the_ver0_layout(self):
        # Linux uapi linux/perf_event.h: PERF_ATTR_SIZE_VER0 is 64.
        self.assertEqual(pc.struct.calcsize(pc.PERF_ATTR_VER0), 64)
        self.assertEqual(pc.PERF_EVENT_OPEN_NR["x86_64"], 298)


if __name__ == "__main__":
    unittest.main()
