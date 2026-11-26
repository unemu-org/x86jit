#!/usr/bin/env python3
"""Measuring and analysis for scripts/perf-campaign.sh (decision-23 performance requirement).

The measure is host instructions per guest step from real16_loops: each process runs one
workload for three rounds over every path and prints, per path, the median of its rounds.
A cell is one (workload, path); its value on a side is the median over that side's
processes. Every process starts with address-space randomization off.

The gate: every cell's change must be at most +GATE % (default 0.5), and so must every
store-probe cell. When a side's values in a cell still form two modes (two clusters
whose medians differ by more than 0.5 %), the decision-23 bimodal rule applies: like
modes are compared, and the campaign runs at least seven processes per side in that
cell. The path means, the geometric mean of a path's cells on each side, are reported
beside the cells.

Two profiles (the gate of record): the campaign runs in the default release profile and in
the `perf` profile (one codegen unit for the core and the bench). Each profile is analysed on
its own, bimodal fallback included; then a cell, or a store-probe cell, BLOCKS only when it is
over the gate in both profiles. A cell over the gate in one profile only is DRIFT: it is
listed with its profile and change, and does not block.

Subcommands:
  campaign           measure two binary directories and analyse (perf-campaign.sh runs it)
  analyze            analyse a raw directory again
  validate-divisors  compare each workload's divisors against divisor 1 on one binary
  compare-record     compare a campaign's analysis with a recorded campaign
  combine            combine a release and a perf analysis into the two-profile verdict
  summary            Markdown of a two-profile campaign's verdict, for a CI job summary
  counter-check      check that the host instruction counter counts on this machine
  snapshot           copy a working tree's tracked and untracked, not ignored, files

Exit status of campaign and analyze (one profile): 0 pass, 1 a cell misses the gate, 2 a
problem. Exit status of combine (two profiles): 0 pass, 1 block, 2 a problem, 3 drift only.
"""

import argparse
import concurrent.futures
import csv
import ctypes
import ctypes.util
import io
import json
import math
import os
import platform
import re
import shutil
import statistics
import struct
import subprocess
import sys
import threading
import time

# The workloads of real16_loops, in its order.
WORKLOADS = [
    "alu", "load", "load-odd", "store", "push-pop", "data-next-to-code",
    "data-next-to-dense-code", "call-ret", "dos-mixed", "rep-movsw", "straight-line",
    "ip-boundary-1", "ip-boundary-64", "smc-next-1", "smc-next-64", "smc-rmw-64",
]

# Step-count divisor per workload. real16_loops divides each workload's step count and
# warm-up by it, except straight-line, whose run grows with the divisor (one warm pass
# and three measured passes per unit), so straight-line runs at 1. Each value was
# validated against divisor 1 with `validate-divisors` (backlog/docs/commands.md).
DEFAULT_DIVISOR = 8
DIVISORS = {"straight-line": 1}

# real16_loops' rounds per process (three is its minimum); a process reports the median
# of its rounds. straight-line's measured run is short (about 90000 steps at divisor 1), so
# one perturbed round moves it by 1-5 %; seven rounds keep such a round out of the median.
DEFAULT_ROUNDS = 3
ROUNDS = {"straight-line": 7}

# The store probe: WIDTH PLACEMENT RECOVERY, run for STORE_ITERATIONS x 16 stores.
STORE_CELLS = [
    (w, p, r) for w in (8, 32) for p in ("single", "crossing") for r in ("off", "on")
]
STORE_ITERATIONS = 200000

PATH_LABELS = {
    "application-real16": "Alias",
    "machine-i8088": "I8088",
    "machine-i8086": "I8086",
    "machine-nec-v20": "V20",
    "machine-nec-v30": "V30",
    "legacy-i286": "I286",
    "legacy-i386": "I386",
    "legacy-i286-a20-off": "I286 A20 off",
    "legacy-i386-a20-off": "I386 A20 off",
}

GATE = 0.5
# Two clusters whose medians differ by more than this many percent are two modes.
MODE_SPLIT = 0.5
# decision-23, gate on bimodal cells: at least seven processes per side.
BIMODAL_MIN_PROCESSES = 7
PROCESS_TIMEOUT = 1200

POSIX_SPAWN_DISABLE_ASLR = 0x0100  # macOS <spawn.h>, _POSIX_SPAWN_DISABLE_ASLR


def divisor(workload, divisors=None):
    table = DIVISORS if divisors is None else divisors
    return table.get(workload, DEFAULT_DIVISOR)


def rounds(workload):
    return ROUNDS.get(workload, DEFAULT_ROUNDS)


def pct(base, cand):
    return (cand / base - 1.0) * 100.0


# --------------------------------------------------------------------------- parsing


def parse_loops_csv(text):
    """Rows of real16_loops' summary CSV: dicts with workload, path, counts, value."""
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        if not row.get("name") or row.get("path") is None:
            continue
        rows.append({
            "workload": row["name"],
            "path": row["path"],
            "counts": (row["steps"], row["retired_instructions"], row["warmup_steps"]),
            "value": float(row["instructions_per_step"]),
        })
    return rows


def parse_store_csv(text):
    """Rows of the counted store probe: ((width, placement, recovery), value)."""
    rows = []
    for line in text.splitlines():
        fields = line.strip().split(",")
        if len(fields) < 6 or not fields[0].isdigit():
            continue
        rows.append(((int(fields[0]), fields[1], fields[2]), float(fields[5])))
    return rows


def store_key(cell):
    return f"{cell[0]} {cell[1]} {cell[2]}"


def load_side(raw, side):
    """Read one side of a raw directory.

    Loop files are `<side>-<workload>-<index>.csv`; store files are `store-<side>.csv`
    (several rows) or `store-<side>-<anything>.csv`. Returns values per cell, the count
    tuples per cell, and values per store cell.
    """
    loop_re = re.compile(rf"{re.escape(side)}-(.+)-(\d+)\.csv")
    store_re = re.compile(rf"store-{re.escape(side)}(-.+)?\.csv")
    cells, counts, store = {}, {}, {}
    for name in sorted(os.listdir(raw)):
        path = os.path.join(raw, name)
        m = loop_re.fullmatch(name)
        if m and not name.startswith("store-"):
            with open(path) as f:
                for row in parse_loops_csv(f.read()):
                    key = (row["workload"], row["path"])
                    cells.setdefault(key, []).append(row["value"])
                    counts.setdefault(key, set()).add(row["counts"])
            continue
        if store_re.fullmatch(name):
            with open(path) as f:
                for cell, value in parse_store_csv(f.read()):
                    store.setdefault(store_key(cell), []).append(value)
    return cells, counts, store


# ------------------------------------------------------------------------ statistics


def modes(values, split=MODE_SPLIT):
    """One or two clusters, split at the largest gap when their medians differ by > split %."""
    v = sorted(values)
    if len(v) < 2:
        return [v]
    _, i = max((v[i + 1] - v[i], i) for i in range(len(v) - 1))
    lo, hi = v[: i + 1], v[i + 1 :]
    if pct(statistics.median(lo), statistics.median(hi)) > split:
        return [lo, hi]
    return [v]


def spread(values):
    """(max - min) / median, in percent."""
    if not values:
        return 0.0
    return (max(values) - min(values)) / statistics.median(values) * 100.0


def compare_cell(base, cand):
    """Compare one cell's per-process values. gate_delta is what the gate judges."""
    b, c = statistics.median(base), statistics.median(cand)
    d = {
        "base": list(base), "cand": list(cand), "base_med": b, "cand_med": c,
        "median_delta": pct(b, c), "spread_base": spread(base), "spread_cand": spread(cand),
        "bimodal": False, "pairs": [], "unmatched": [], "gate_delta": pct(b, c),
    }
    bm, cm = modes(base), modes(cand)
    if len(bm) == 1 and len(cm) == 1:
        return d
    d["bimodal"] = True
    bmed = [statistics.median(m) for m in bm]
    cmed = [statistics.median(m) for m in cm]
    if len(bm) == len(cm):
        pairs = list(zip(range(len(bm)), range(len(cm))))
    elif len(bm) == 1:
        j = min(range(len(cm)), key=lambda j: abs(cmed[j] - bmed[0]))
        pairs = [(0, j)]
        d["unmatched"] = [
            ["cand", cmed[k], len(cm[k]), pct(bmed[0], cmed[k])] for k in range(len(cm)) if k != j
        ]
    else:
        k = min(range(len(bm)), key=lambda k: abs(bmed[k] - cmed[0]))
        pairs = [(k, 0)]
        d["unmatched"] = [
            ["base", bmed[j], len(bm[j]), pct(bmed[j], cmed[0])] for j in range(len(bm)) if j != k
        ]
    d["pairs"] = [
        [bmed[i], cmed[j], len(bm[i]), len(cm[j]), pct(bmed[i], cmed[j])] for i, j in pairs
    ]
    d["gate_delta"] = max(p[4] for p in d["pairs"])
    return d


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def path_means(cells):
    """Per path: geometric means of the base and candidate cell values, and their change."""
    by_path = {}
    for key, d in cells.items():
        _, path = key.split("|", 1)
        by_path.setdefault(path, []).append(d)
    out = {}
    for path, ds in by_path.items():
        b = geomean([d["base_med"] for d in ds])
        c = geomean([d["cand_med"] for d in ds])
        out[path] = {"cells": len(ds), "base": b, "cand": c, "delta": pct(b, c)}
    return out


def ordered_paths(keys):
    seen = []
    for key in keys:
        path = key.split("|", 1)[1]
        if path not in seen:
            seen.append(path)
    known = [p for p in PATH_LABELS if p in seen]
    return known + [p for p in seen if p not in known]


def percentile(values, q):
    v = sorted(values)
    if not v:
        return 0.0
    k = (len(v) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def analyze(raw, base_side="base", cand_side="cand", gate=GATE):
    """Analyse a raw directory. Returns a dict that `report` formats and JSON can hold."""
    bcells, bcounts, bstore = load_side(raw, base_side)
    ccells, ccounts, cstore = load_side(raw, cand_side)
    problems = []
    notes = []
    cells = {}
    bpaths, cpaths = {k[1] for k in bcells}, {k[1] for k in ccells}
    for path in sorted(bpaths ^ cpaths):
        # A path that one side's benchmark does not have (a model added in between).
        notes.append(f"path {path} only on the {'base' if path in bpaths else 'candidate'}")
    workloads = [w for w in WORKLOADS if any(k[0] == w for k in bcells)]
    workloads += sorted({k[0] for k in bcells} - set(workloads))
    for key in sorted(set(bcells) | set(ccells), key=lambda k: (
            workloads.index(k[0]) if k[0] in workloads else len(workloads), k[1])):
        if key[1] not in bpaths & cpaths:
            continue
        if key not in bcells or key not in ccells:
            problems.append(f"cell {key[0]} {key[1]} missing on one side")
            continue
        allc = bcounts[key] | ccounts[key]
        if len(allc) != 1:
            problems.append(f"counts differ in {key[0]} {key[1]}: {sorted(allc)}")
        values = bcells[key] + ccells[key]
        if any(v <= 0 for v in values):
            problems.append(f"no host instruction count in {key[0]} {key[1]}")
            continue
        cells[f"{key[0]}|{key[1]}"] = compare_cell(bcells[key], ccells[key])
    store = {}
    for key in sorted(set(bstore) | set(cstore), key=lambda k: (int(k.split()[0]), k)):
        if key not in bstore or key not in cstore:
            problems.append(f"store cell {key} missing on one side")
            continue
        if any(v <= 0 for v in bstore[key] + cstore[key]):
            problems.append(f"no host instruction count in store cell {key}")
            continue
        b, c = statistics.median(bstore[key]), statistics.median(cstore[key])
        store[key] = {"base": bstore[key], "cand": cstore[key], "base_med": b, "cand_med": c,
                      "delta": pct(b, c)}
    if not cells:
        problems.append("no cells")
    misses = sorted((k for k, d in cells.items() if d["gate_delta"] > gate),
                    key=lambda k: -cells[k]["gate_delta"])
    store_misses = sorted(k for k, d in store.items() if d["delta"] > gate)
    short = [k for k, d in cells.items() if d["bimodal"]
             and min(len(d["base"]), len(d["cand"])) < BIMODAL_MIN_PROCESSES]
    spreads = [d[s] for d in cells.values() for s in ("spread_base", "spread_cand")]
    if problems:
        verdict = "PROBLEM"
    elif misses or store_misses:
        verdict = "MISS"
    else:
        verdict = "PASS"
    return {
        "gate": gate, "base_side": base_side, "cand_side": cand_side,
        "cells": cells, "paths": path_means(cells), "store": store,
        "misses": misses, "store_misses": store_misses, "bimodal": sorted(
            k for k, d in cells.items() if d["bimodal"]),
        "bimodal_short": sorted(short),
        "spread": {"p50": percentile(spreads, 0.5), "p90": percentile(spreads, 0.9),
                   "max": max(spreads) if spreads else 0.0},
        "processes": sorted({(len(d["base"]), len(d["cand"])) for d in cells.values()}),
        "problems": problems, "notes": notes, "verdict": verdict,
    }


def exit_status(result):
    return {"PASS": 0, "MISS": 1}.get(result["verdict"], 2)


def report(result, meta=None):
    meta = meta or {}
    gate = result["gate"]
    cells = result["cells"]
    out = []
    title = "Host instructions per guest step (real16_loops), change from base to candidate"
    out.append(title)
    for k in ("base", "candidate", "profile", "divisors", "rounds", "aslr", "processes", "parallel",
              "measuring_seconds"):
        if k in meta:
            out.append(f"  {k}: {meta[k]}")
    paths = ordered_paths(cells)
    workloads = []
    for key in cells:
        w = key.split("|", 1)[0]
        if w not in workloads:
            workloads.append(w)
    out.append("")
    out.append("Per cell, % (b = two modes, like modes compared; ! = over the gate):")
    out.append("| Workload | " + " | ".join(PATH_LABELS.get(p, p) for p in paths) + " |")
    out.append("|---|" + "---:|" * len(paths))
    for w in workloads:
        row = []
        for p in paths:
            d = cells.get(f"{w}|{p}")
            if d is None:
                row.append("-")
                continue
            mark = ("b" if d["bimodal"] else "") + ("!" if d["gate_delta"] > gate else "")
            row.append(f"{d['gate_delta']:+.3f}{mark}")
        out.append(f"| `{w}` | " + " | ".join(row) + " |")
    out.append("")
    out.append("Path means (geometric mean of the path's cells, host instructions per step):")
    out.append("| Path | Cells | Base | Candidate | Change |")
    out.append("|---|---:|---:|---:|---:|")
    for p in paths:
        m = result["paths"][p]
        out.append(f"| {PATH_LABELS.get(p, p)} | {m['cells']} | {m['base']:.3f} | "
                   f"{m['cand']:.3f} | {m['delta']:+.3f}% |")
    if result["store"]:
        out.append("")
        out.append(f"Store probe, host instructions per store ({STORE_ITERATIONS} x 16 stores; "
                   "median of processes):")
        out.append("| Width placement recovery | Base | Candidate | Change |")
        out.append("|---|---:|---:|---:|")
        for k, d in result["store"].items():
            mark = " !" if d["delta"] > gate else ""
            out.append(f"| {k} | {d['base_med']:.3f} | {d['cand_med']:.3f} | "
                       f"{d['delta']:+.3f}%{mark} |")
    out.append("")
    s = result["spread"]
    out.append(f"Processes per cell (base/candidate): {result['processes']}")
    out.append(f"Spread within a side and cell, (max - min) / median: p50 {s['p50']:.3f} %, "
               f"p90 {s['p90']:.3f} %, max {s['max']:.3f} %")
    deltas = [d["median_delta"] for d in cells.values()]
    if deltas:
        out.append(f"Median deltas: {min(deltas):+.3f} % .. {max(deltas):+.3f} %")
    out.append(f"Cells with two modes: {len(result['bimodal'])}")
    for k in result["bimodal"]:
        d = cells[k]
        pairs = "; ".join(f"{b:.3f}->{c:.3f} ({x:+.3f}%, n {nb}/{nc})" for b, c, nb, nc, x in d["pairs"])
        un = ", ".join(f"{side} {v:.3f} (n {n}, {x:+.3f}%)" for side, v, n, x in d["unmatched"])
        note = "  FEWER THAN 7 PROCESSES PER SIDE" if k in result["bimodal_short"] else ""
        out.append(f"  {k}: median {d['median_delta']:+.3f}%; like modes {pairs}"
                   + (f"; unmatched {un}" if un else "") + note)
    for note in result.get("notes", []):
        out.append(f"Note: {note}")
    if result["problems"]:
        out.append("PROBLEMS:")
        out.extend(f"  {p}" for p in result["problems"])
    out.append("")
    out.append(f"GATE +{gate}% per cell (median of processes; two modes: worst like-mode pair): "
               + ("PASS" if not result["misses"] else f"{len(result['misses'])} MISS"))
    for k in result["misses"]:
        out.append(f"  MISS {k} {cells[k]['gate_delta']:+.3f}%")
    if result["store"]:
        out.append(f"STORE GATE +{gate}%: " + ("PASS" if not result["store_misses"]
                                               else f"{len(result['store_misses'])} MISS"))
        for k in result["store_misses"]:
            out.append(f"  MISS store {k} {result['store'][k]['delta']:+.3f}%")
    out.append(f"VERDICT: {result['verdict']}")
    return "\n".join(out)


def cells_csv(result):
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["workload", "path", "base_median", "cand_median", "median_delta", "gate_delta",
                "bimodal", "base_processes", "cand_processes", "spread_base", "spread_cand"])
    for key, d in result["cells"].items():
        wl, p = key.split("|", 1)
        w.writerow([wl, p, f"{d['base_med']:.3f}", f"{d['cand_med']:.3f}",
                    f"{d['median_delta']:.4f}", f"{d['gate_delta']:.4f}", int(d["bimodal"]),
                    len(d["base"]), len(d["cand"]), f"{d['spread_base']:.4f}",
                    f"{d['spread_cand']:.4f}"])
    return out.getvalue()


# ---------------------------------------------------------------------- two profiles

COMBINED_STATUS = {"PASS": 0, "BLOCK": 1, "PROBLEM": 2, "DRIFT": 3}


def combine(results, gate=GATE):
    """Combine per-profile analyses ({profile: analyze() result}) into the two-profile verdict.

    A cell or store cell blocks when it is over the gate in every profile; over in some but
    not all, it is drift. A problem in any profile, or a cell missing from one, is a problem.
    """
    profiles = list(results)
    problems = [f"{p}: {x}" for p in profiles for x in results[p]["problems"]]
    out = {"profiles": profiles, "gate": gate, "block": [], "drift": [],
           "store_block": [], "store_drift": [], "problems": problems}
    for kind, field, delta in (("cells", "", "gate_delta"), ("store", "store_", "delta")):
        keys = [set(results[p][kind]) for p in profiles]
        common = set.intersection(*keys) if keys else set()
        for k in sorted(set.union(*keys) - common) if keys else []:
            problems.append(f"{kind[:-1] if kind == 'cells' else 'store cell'} {k} missing in a profile")
        for k in sorted(common):
            deltas = {p: results[p][kind][k][delta] for p in profiles}
            over = [p for p in profiles if deltas[p] > gate]
            if not over:
                continue
            entry = {"cell": k, "over": over, "deltas": deltas}
            out[f"{field}block" if len(over) == len(profiles) else f"{field}drift"].append(entry)
    for name in ("block", "drift", "store_block", "store_drift"):
        out[name].sort(key=lambda e: -max(e["deltas"].values()))
    if problems:
        out["verdict"] = "PROBLEM"
    elif out["block"] or out["store_block"]:
        out["verdict"] = "BLOCK"
    elif out["drift"] or out["store_drift"]:
        out["verdict"] = "DRIFT"
    else:
        out["verdict"] = "PASS"
    return out


def combined_status(combined):
    return COMBINED_STATUS[combined["verdict"]]


def combined_report(combined):
    profiles = combined["profiles"]
    gate = combined["gate"]
    lines = [f"TWO-PROFILE GATE ({' and '.join(profiles)}): a cell blocks when it is over "
             f"+{gate}% in every profile; over in one profile only, it is drift."]

    def entry(e, store=False):
        name = f"store {e['cell']}" if store else e["cell"]
        vals = ", ".join(f"{p} {e['deltas'][p]:+.3f}%" for p in profiles)
        return f"  {name}: over in {'+'.join(e['over'])} ({vals})"

    for title, key, store in (("BLOCK", "block", False), ("BLOCK", "store_block", True),
                              ("DRIFT", "drift", False), ("DRIFT", "store_drift", True)):
        if combined[key]:
            lines.append(f"{title} ({'store probe' if store else 'cells'}): {len(combined[key])}")
            lines.extend(entry(e, store) for e in combined[key])
    if combined["problems"]:
        lines.append("PROBLEMS:")
        lines.extend(f"  {p}" for p in combined["problems"])
    lines.append(f"COMBINED VERDICT: {combined['verdict']} (exit {combined_status(combined)})")
    return "\n".join(lines)


def cmd_combine(args):
    results = {}
    for spec in args.analysis:
        profile, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"combine: expected PROFILE=analysis.json, got {spec}")
        with open(path) as f:
            results[profile] = json.load(f)
    c = combine(results, args.gate)
    text = combined_report(c)
    print(text)
    if args.out:
        with open(os.path.join(args.out, "combined.txt"), "w") as f:
            f.write(text + "\n")
        with open(os.path.join(args.out, "combined.json"), "w") as f:
            json.dump(c, f, indent=1)
    return combined_status(c)


def cell_label(key, store=False):
    """`workload` Path for a cell key, `store` W P R for a store-probe cell."""
    if store:
        return f"store {key}"
    workload, path = key.split("|", 1)
    return f"`{workload}` {PATH_LABELS.get(path, path)}"


def summary_markdown(out):
    """Markdown for a CI job summary from a campaign's --out directory: the combined verdict,
    the blocking and drift cells, the problems, the path means and each profile's report."""
    combined_path = os.path.join(out, "combined.json")
    if not os.path.exists(combined_path):
        return ("## Perf gate: no verdict\n\nThe campaign stopped before the two-profile verdict "
                "(a usage, build or measuring problem); see the job log and the artifact.\n")
    with open(combined_path) as f:
        combined = json.load(f)
    profiles = combined["profiles"]
    results = {}
    for p in profiles:
        path = os.path.join(out, p, "analysis.json")
        if os.path.exists(path):
            with open(path) as f:
                results[p] = json.load(f)
    meta = next((r.get("meta", {}) for r in results.values()), {})

    def name(value):  # "abc123 (/path/to/bin)" -> "abc123"
        return str(value).split(" (", 1)[0]

    lines = [f"## Perf gate: {combined['verdict']} (exit {combined_status(combined)})", ""]
    if meta:
        lines.append(f"Base `{name(meta.get('base', '?'))}`, candidate "
                     f"`{name(meta.get('candidate', '?'))}`; {meta.get('processes', '?')} "
                     f"processes per cell and side, randomization {meta.get('aslr', '?')}.")
    lines.append(f"Profiles {' and '.join(profiles)}, gate +{combined['gate']}% per cell: a cell "
                 "blocks when it is over in every profile; over in some only, it is drift.")
    lines.append("")
    rows = [(title, e, store) for title, key, store in (
        ("BLOCK", "block", False), ("BLOCK", "store_block", True),
        ("DRIFT", "drift", False), ("DRIFT", "store_drift", True)) for e in combined[key]]
    if rows:
        lines.append("| Verdict | Cell | Over in | " + " | ".join(profiles) + " |")
        lines.append("|---|---|---|" + "---:|" * len(profiles))
        for title, e, store in rows:
            lines.append(f"| {title} | {cell_label(e['cell'], store)} | {'+'.join(e['over'])} | "
                         + " | ".join(f"{e['deltas'][p]:+.3f}%" for p in profiles) + " |")
    else:
        lines.append("No cell is over the gate in any profile.")
    if combined["problems"]:
        lines += ["", "Problems:", ""] + [f"- {p}" for p in combined["problems"]]
    if results and all(r.get("paths") for r in results.values()):
        paths = ordered_paths(next(iter(results.values()))["cells"])
        lines += ["", "Path means, change in host instructions per guest step:", ""]
        lines.append("| Path | " + " | ".join(results) + " |")
        lines.append("|---|" + "---:|" * len(results))
        for path in paths:
            vals = [f"{r['paths'][path]['delta']:+.3f}%" if path in r["paths"] else "-"
                    for r in results.values()]
            lines.append(f"| {PATH_LABELS.get(path, path)} | " + " | ".join(vals) + " |")
    for p in profiles:
        path = os.path.join(out, p, "report.txt")
        if os.path.exists(path):
            with open(path) as f:
                text = f.read().rstrip("\n")
            lines += ["", f"<details><summary>{p} profile report</summary>", "", "```text",
                      text, "```", "", "</details>"]
    return "\n".join(lines) + "\n"


def cmd_summary(args):
    sys.stdout.write(summary_markdown(args.out))
    return 0


# --------------------------------------------------------------------------- running

# Apple SDK sys/resource.h, rusage_info_v4: 16 bytes of uuid, then 35 u64s, of which
# ri_instructions follows 29 (x86jit-bench/examples/dispatch_support/usage.rs reads the same).
RUSAGE_INFO_V4 = 4
RUSAGE_V4_WORDS = 2 + 35
RUSAGE_V4_INSTRUCTIONS = 2 + 29


def host_instructions():
    """This process's retired host instructions (proc_pid_rusage ri_instructions), macOS only."""
    libc = ctypes.CDLL(ctypes.util.find_library("c") or None, use_errno=True)
    buf = (ctypes.c_uint64 * RUSAGE_V4_WORDS)()
    if libc.proc_pid_rusage(os.getpid(), RUSAGE_INFO_V4, ctypes.byref(buf)) != 0:
        raise OSError(ctypes.get_errno(), "proc_pid_rusage")
    return buf[RUSAGE_V4_INSTRUCTIONS]


# Linux uapi include/uapi/linux/perf_event.h: struct perf_event_attr up to
# PERF_ATTR_SIZE_VER0 (type, size, config, sample_period, sample_type, read_format, the
# flag bitfield, wakeup_events, bp_type, config1), the event the benchmark examples open.
PERF_ATTR_VER0 = "<IIQQQQQIIQ"
PERF_TYPE_HARDWARE = 0
PERF_COUNT_HW_INSTRUCTIONS = 1
PERF_ATTR_PINNED = 1 << 2
PERF_ATTR_EXCLUDE_KERNEL = 1 << 5
PERF_ATTR_EXCLUDE_HV = 1 << 6
PERF_FLAG_FD_CLOEXEC = 1 << 3
# __NR_perf_event_open: oracles/linux-x86/syscall_64.tbl (298, x86-64) and
# include/uapi/asm-generic/unistd.h (241, the generic table arm64 uses).
PERF_EVENT_OPEN_NR = {"x86_64": 298, "aarch64": 241, "arm64": 241}
PERF_EVENT_PARANOID = "/proc/sys/kernel/perf_event_paranoid"


def perf_event_paranoid():
    try:
        with open(PERF_EVENT_PARANOID) as f:
            return f.read().strip()
    except OSError:
        return "unreadable"


def perf_instruction_counter(machine=None):
    """A perf_event_open file descriptor counting this thread's user-mode instructions."""
    machine = machine or platform.machine()
    nr = PERF_EVENT_OPEN_NR.get(machine)
    if nr is None:
        raise OSError(f"no perf_event_open syscall number known for {machine}")
    attr = ctypes.create_string_buffer(struct.pack(
        PERF_ATTR_VER0, PERF_TYPE_HARDWARE, struct.calcsize(PERF_ATTR_VER0),
        PERF_COUNT_HW_INSTRUCTIONS, 0, 0, 0,
        PERF_ATTR_PINNED | PERF_ATTR_EXCLUDE_KERNEL | PERF_ATTR_EXCLUDE_HV, 0, 0, 0))
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    fd = libc.syscall(ctypes.c_long(nr), attr, ctypes.c_long(0), ctypes.c_long(-1),
                      ctypes.c_long(-1), ctypes.c_long(PERF_FLAG_FD_CLOEXEC))
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, f"perf_event_open(PERF_COUNT_HW_INSTRUCTIONS): {os.strerror(err)}")
    return fd


def read_counter(fd):
    data = os.read(fd, 8)
    if len(data) != 8:
        # A pinned event that lost its counter reads as end of file.
        raise OSError("the perf counter is in error state")
    return struct.unpack("<Q", data)[0]


def _check_counts(name, before, after):
    print(f"counter-check: {name} {before} -> {after} (+{after - before})")
    if after <= before:
        print("counter-check: the host instruction counter does not count on this machine",
              file=sys.stderr)
        return 2
    return 0


def cmd_counter_check(args):
    """Exit 0 when the host instruction counter the benchmarks read counts; 2 when it does not
    (another OS, or a virtual machine without the counter), so CI can stop before building."""
    system = platform.system()
    if system == "Linux":
        paranoid = perf_event_paranoid()
        print(f"counter-check: kernel.perf_event_paranoid {paranoid}")
        try:
            fd = perf_instruction_counter()
            try:
                before = read_counter(fd)
                sum(range(2_000_000))
                after = read_counter(fd)
            finally:
                os.close(fd)
        except OSError as e:
            print(f"counter-check: {e} (kernel.perf_event_paranoid {paranoid}; a virtual "
                  "machine without a PMU fails here too)", file=sys.stderr)
            return 2
        return _check_counts("perf_event_open instructions", before, after)
    if system != "Darwin":
        print(f"counter-check: no host instruction counter on {system}; "
              "the campaign measures on macOS and Linux", file=sys.stderr)
        return 2
    try:
        before = host_instructions()
        sum(range(2_000_000))
        after = host_instructions()
    except (OSError, AttributeError) as e:
        print(f"counter-check: proc_pid_rusage failed: {e}", file=sys.stderr)
        return 2
    if before == 0:
        print(f"counter-check: ri_instructions {before} -> {after}")
        print("counter-check: the host instruction counter does not count on this machine",
              file=sys.stderr)
        return 2
    return _check_counts("ri_instructions", before, after)

_libc = None


def _spawn_macos(argv, stdout_path, stderr_path, disable_aslr):
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(ctypes.util.find_library("c") or None, use_errno=True)
    libc = _libc
    attr = ctypes.c_void_p()
    actions = ctypes.c_void_p()
    if libc.posix_spawnattr_init(ctypes.byref(attr)) != 0:
        raise OSError("posix_spawnattr_init")
    if libc.posix_spawn_file_actions_init(ctypes.byref(actions)) != 0:
        raise OSError("posix_spawn_file_actions_init")
    try:
        if disable_aslr:
            rc = libc.posix_spawnattr_setflags(ctypes.byref(attr),
                                               ctypes.c_short(POSIX_SPAWN_DISABLE_ASLR))
            if rc != 0:
                raise OSError(rc, "posix_spawnattr_setflags")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        for fd, path in ((1, stdout_path), (2, stderr_path)):
            rc = libc.posix_spawn_file_actions_addopen(ctypes.byref(actions), fd,
                                                       path.encode(), flags, 0o644)
            if rc != 0:
                raise OSError(rc, "posix_spawn_file_actions_addopen")
        args = (ctypes.c_char_p * (len(argv) + 1))(*[a.encode() for a in argv], None)
        env = [f"{k}={v}".encode() for k, v in os.environ.items()]
        envp = (ctypes.c_char_p * (len(env) + 1))(*env, None)
        pid = ctypes.c_int()
        rc = libc.posix_spawn(ctypes.byref(pid), argv[0].encode(), ctypes.byref(actions),
                              ctypes.byref(attr), args, envp)
        if rc != 0:
            raise OSError(rc, f"posix_spawn {argv[0]}: {os.strerror(rc)}")
        return pid.value
    finally:
        libc.posix_spawn_file_actions_destroy(ctypes.byref(actions))
        libc.posix_spawnattr_destroy(ctypes.byref(attr))


def _wait(pid, timeout):
    deadline = time.monotonic() + timeout
    while True:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return os.waitstatus_to_exitcode(status)
        if time.monotonic() > deadline:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
            return "timeout"
        time.sleep(0.05)


def aslr_prefix(system=None, machine=None):
    """The command prefix that turns randomization off where a spawn flag cannot."""
    system = system or platform.system()
    if system == "Linux":
        return ["setarch", machine or platform.machine(), "-R"]
    return []


def run_process(argv, stdout_path, stderr_path, aslr=False, timeout=PROCESS_TIMEOUT):
    """Run one benchmark process; returns its exit code or "timeout"."""
    if platform.system() == "Darwin":
        pid = _spawn_macos(argv, stdout_path, stderr_path, disable_aslr=not aslr)
        return _wait(pid, timeout)
    prefix = [] if aslr else aslr_prefix()
    with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
        try:
            return subprocess.run(prefix + argv, stdout=out, stderr=err, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            return "timeout"


def plan(sides, indices, workloads=WORKLOADS, store=True, divisors=None):
    """The process list. Sides alternate their order by index, so they share the load."""
    tasks = []
    for i in indices:
        order = sides if i % 2 == 1 else list(reversed(sides))
        for w in workloads:
            for side, bindir in order:
                tasks.append({"kind": "loops", "side": side, "bin": bindir, "workload": w,
                              "index": i, "divisor": divisor(w, divisors)})
        if store:
            for cell in STORE_CELLS:
                for side, bindir in order:
                    tasks.append({"kind": "store", "side": side, "bin": bindir, "cell": cell,
                                  "index": i})
    return tasks


def run_task(task, raw, aslr):
    if task["kind"] == "loops":
        stem = f"{task['side']}-{task['workload']}-{task['index']}"
        argv = [os.path.join(task["bin"], "real16_loops"), str(task["divisor"]),
                str(rounds(task["workload"])), task["workload"]]
    else:
        w, p, r = task["cell"]
        stem = f"store-{task['side']}-{w}-{p}-{r}-{task['index']}"
        argv = [os.path.join(task["bin"], "interpreter_store_probe"), str(w), p, r,
                str(STORE_ITERATIONS)]
    rc = run_process(argv, os.path.join(raw, stem + ".csv"), os.path.join(raw, stem + ".log"),
                     aslr=aslr)
    return stem, rc


def execute(tasks, raw, parallel, aslr=False, log=sys.stderr):
    """Run the tasks, `parallel` at once; returns the failures."""
    os.makedirs(raw, exist_ok=True)
    failures = []
    lock = threading.Lock()
    done = [0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(run_task, t, raw, aslr) for t in tasks]
        for fut in concurrent.futures.as_completed(futures):
            stem, rc = fut.result()
            with lock:
                done[0] += 1
                if rc != 0:
                    failures.append(f"{stem}: exit {rc}")
                if done[0] % 50 == 0 or done[0] == len(tasks):
                    print(f"perf-campaign: {done[0]}/{len(tasks)} processes", file=log, flush=True)
    with open(os.path.join(raw, "status"), "a") as f:
        for line in failures:
            f.write(f"FAIL {line}\n")
    return failures


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_campaign(args):
    raw = os.path.join(args.out, "raw")
    os.makedirs(raw, exist_ok=True)
    sides = [("base", args.base), ("cand", args.cand)]
    aslr = "on" if args.aslr else "off"
    meta = {
        "base": f"{args.base_name} ({args.base})", "candidate": f"{args.cand_name} ({args.cand})",
        "profile": args.profile,
        "divisors": f"{DEFAULT_DIVISOR}, " + ", ".join(f"{k} {v}" for k, v in DIVISORS.items()),
        "rounds": f"{DEFAULT_ROUNDS}, " + ", ".join(f"{k} {v}" for k, v in ROUNDS.items()),
        "aslr": aslr, "processes": args.processes, "parallel": args.parallel,
    }
    binaries = {f"{s}/{n}": sha256(os.path.join(b, n)) for s, b in sides
                for n in ("real16_loops", "interpreter_store_probe")}
    start = time.time()
    failures = execute(plan(sides, range(1, args.processes + 1), store=not args.no_store),
                       raw, args.parallel, aslr=args.aslr)
    result = analyze(raw, gate=args.gate)
    # Bimodal fallback: top a two-mode cell's workload up to seven processes per side.
    topped = sorted({k.split("|", 1)[0] for k in result["bimodal_short"]})
    if topped and not failures:
        print(f"perf-campaign: two modes in {', '.join(result['bimodal_short'])}; "
              f"running {', '.join(topped)} to {BIMODAL_MIN_PROCESSES} processes per side",
              file=sys.stderr)
        failures += execute(plan(sides, range(args.processes + 1, BIMODAL_MIN_PROCESSES + 1),
                                 workloads=topped, store=False), raw, args.parallel,
                            aslr=args.aslr)
        result = analyze(raw, gate=args.gate)
    meta["measuring_seconds"] = round(time.time() - start)
    result["problems"] += [f"process failed: {f}" for f in failures]
    if result["problems"]:
        result["verdict"] = "PROBLEM"
    result["meta"] = meta
    result["binaries"] = binaries
    text = report(result, meta)
    with open(os.path.join(args.out, "report.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, "analysis.json"), "w") as f:
        json.dump(result, f, indent=1)
    with open(os.path.join(args.out, "cells.csv"), "w") as f:
        f.write(cells_csv(result))
    print(text)
    print(f"\nraw data and report: {args.out}")
    return exit_status(result)


def cmd_analyze(args):
    result = analyze(args.raw, args.base_side, args.cand_side, args.gate)
    print(report(result))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    return exit_status(result)


# ---------------------------------------------------------------------- validation


def cmd_validate_divisors(args):
    """Run each workload at divisor 1 and at each candidate divisor on one binary and
    report, per workload and divisor, the largest change of a path's median."""
    raw = os.path.join(args.out, "raw")
    divs = sorted(set([1] + args.divisors))
    workloads = args.workloads or WORKLOADS
    tasks = []
    for i in range(1, args.processes + 1):
        for w in workloads:
            for d in divs:
                tasks.append({"kind": "loops", "side": f"d{d}", "bin": args.bin, "workload": w,
                              "index": i, "divisor": d})
    failures = execute(tasks, raw, args.parallel)
    table = {}
    lines = [f"Divisor validation against divisor 1, {args.processes} processes each, "
             f"median of processes; largest |change| over the paths, %:",
             "| Workload | " + " | ".join(f"/{d}" for d in divs if d != 1) + " |",
             "|---|" + "---:|" * (len(divs) - 1)]
    ones, _, _ = load_side(raw, "d1")
    for w in workloads:
        row = []
        for d in divs:
            if d == 1:
                continue
            cells, _, _ = load_side(raw, f"d{d}")
            worst = 0.0
            for (wl, p), v in cells.items():
                if wl != w or (wl, p) not in ones:
                    continue
                worst = max(worst, abs(pct(statistics.median(ones[(wl, p)]), statistics.median(v))))
            table.setdefault(w, {})[d] = worst
            row.append(f"{worst:.3f}" + (" !" if worst > args.tolerance else ""))
        lines.append(f"| `{w}` | " + " | ".join(row) + " |")
    lines.append(f"! = over the tolerance of {args.tolerance} %")
    if failures:
        lines.append("FAILURES: " + "; ".join(failures))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "divisors.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, "divisors.json"), "w") as f:
        json.dump(table, f, indent=1)
    return 2 if failures else 0


def record_delta(d):
    """The delta of a recorded cell that a randomization-off replay should reproduce:
    the median delta of a one-mode cell, else the like-mode pair of the lowest base mode."""
    if not d.get("bimodal") or not d.get("pairs"):
        return d["median_delta"]
    return min(d["pairs"], key=lambda p: p[0])[4]


def compare_record(replay, record, gate=GATE, tolerance=0.25):
    """Compare two analyses: which cells are over the gate in each, and how far apart
    the deltas of common cells are."""
    common = [k for k in replay["cells"] if k in record["cells"]]
    rows = []
    for k in common:
        a = replay["cells"][k]["gate_delta"]
        b = record_delta(record["cells"][k])
        rows.append((k, a, b, a - b))
    over_replay = {k for k in common if replay["cells"][k]["gate_delta"] > gate}
    over_record = {k for k in common if record["cells"][k]["gate_delta"] > gate}
    diffs = [abs(r[3]) for r in rows]
    return {
        "cells": len(common),
        "over_both": sorted(over_replay & over_record),
        "over_replay_only": sorted(over_replay - over_record),
        "over_record_only": sorted(over_record - over_replay),
        "diff_p50": percentile(diffs, 0.5), "diff_p90": percentile(diffs, 0.9),
        "diff_max": max(diffs) if diffs else 0.0,
        "outside": [r for r in rows if abs(r[3]) > tolerance],
        "rows": rows,
        "replay_bimodal": replay.get("bimodal", []),
    }


def cmd_compare_record(args):
    with open(args.replay) as f:
        replay = json.load(f)
    if args.record_json:
        with open(args.record_json) as f:
            record = json.load(f)
    else:
        record = analyze(args.record_raw, args.record_base, args.record_cand, args.gate)
    c = compare_record(replay, record, args.gate, args.tolerance)
    lines = [
        f"cells compared: {c['cells']}",
        f"over +{args.gate}% in both: {len(c['over_both'])} {c['over_both']}",
        f"over in the replay only: {[(k, round(replay['cells'][k]['gate_delta'], 3), round(record_delta(record['cells'][k]), 3)) for k in c['over_replay_only']]}",
        f"over in the record only: {[(k, round(replay['cells'][k]['gate_delta'], 3), round(record['cells'][k]['gate_delta'], 3)) for k in c['over_record_only']]}",
        f"|replay - record| in percentage points: p50 {c['diff_p50']:.3f}, p90 {c['diff_p90']:.3f}, "
        f"max {c['diff_max']:.3f}",
        f"cells further apart than {args.tolerance} pp: {len(c['outside'])}",
    ]
    for k, a, b, d in c["outside"]:
        lines.append(f"  {k}: replay {a:+.3f}%, record {b:+.3f}% ({d:+.3f} pp)")
    lines.append(f"cells with two modes in the replay: {len(c['replay_bimodal'])} {c['replay_bimodal']}")
    print("\n".join(lines))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(c, f, indent=1)
    return 0


# ------------------------------------------------------------------------- snapshot


def cmd_snapshot(args):
    """Copy a working tree's tracked and untracked, not ignored, files to DEST."""
    listing = subprocess.run(
        ["git", "-C", args.tree, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        check=True, capture_output=True).stdout
    for rel in sorted(set(listing.decode().split("\0"))):
        if not rel:
            continue
        src = os.path.join(args.tree, rel)
        if os.path.isdir(src) and not os.path.islink(src):
            continue  # a submodule
        if not os.path.lexists(src):
            continue  # deleted in the working tree
        dst = os.path.join(args.dest, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.islink(src):
            os.symlink(os.readlink(src), dst)
        else:
            shutil.copy2(src, dst)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("campaign", help="measure two binary directories and analyse")
    p.add_argument("--base", required=True, help="directory with the base binaries")
    p.add_argument("--cand", required=True, help="directory with the candidate binaries")
    p.add_argument("--out", required=True)
    p.add_argument("--base-name", default="base")
    p.add_argument("--cand-name", default="candidate")
    p.add_argument("--profile", default="?")
    p.add_argument("--processes", type=int, default=3)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--gate", type=float, default=GATE)
    p.add_argument("--no-store", action="store_true")
    p.add_argument("--aslr", action="store_true", help="leave randomization on (diagnostics)")
    p.set_defaults(func=cmd_campaign)

    p = sub.add_parser("analyze", help="analyse a raw directory")
    p.add_argument("raw")
    p.add_argument("--base-side", default="base")
    p.add_argument("--cand-side", default="cand")
    p.add_argument("--gate", type=float, default=GATE)
    p.add_argument("--json")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("validate-divisors", help="compare divisors against divisor 1")
    p.add_argument("--bin", required=True, help="directory with real16_loops")
    p.add_argument("--out", required=True)
    p.add_argument("--divisors", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--workloads", nargs="*")
    p.add_argument("--processes", type=int, default=3)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--tolerance", type=float, default=0.15)
    p.set_defaults(func=cmd_validate_divisors)

    p = sub.add_parser("compare-record", help="compare a replay with a recorded campaign")
    p.add_argument("--replay", required=True, help="analysis.json of the replay")
    p.add_argument("--record-json", help="analysis JSON of the record")
    p.add_argument("--record-raw", help="raw directory of the record")
    p.add_argument("--record-base", default="base")
    p.add_argument("--record-cand", default="cand")
    p.add_argument("--gate", type=float, default=GATE)
    p.add_argument("--tolerance", type=float, default=0.25)
    p.add_argument("--json")
    p.set_defaults(func=cmd_compare_record)

    p = sub.add_parser("combine", help="two-profile verdict from per-profile analyses")
    p.add_argument("analysis", nargs="+", help="PROFILE=path/to/analysis.json, one per profile")
    p.add_argument("--gate", type=float, default=GATE)
    p.add_argument("--out", help="directory for combined.txt and combined.json")
    p.set_defaults(func=cmd_combine)

    p = sub.add_parser("summary", help="Markdown job summary of a two-profile campaign")
    p.add_argument("out", help="the campaign's --out directory")
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("counter-check", help="check that the host instruction counter counts")
    p.set_defaults(func=cmd_counter_check)

    p = sub.add_parser("snapshot", help="copy a working tree's files")
    p.add_argument("tree")
    p.add_argument("dest")
    p.set_defaults(func=cmd_snapshot)

    args = parser.parse_args(argv)
    if args.command == "compare-record" and not (args.record_json or args.record_raw):
        parser.error("compare-record needs --record-json or --record-raw")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
