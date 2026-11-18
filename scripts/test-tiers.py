#!/usr/bin/env python3
"""Keep the fast test tier honest: the slow list, the nextest filter and the ratchet.

    python3 -B scripts/test-tiers.py filter
    python3 -B scripts/test-tiers.py sync [--check]
    python3 -B scripts/test-tiers.py ratchet [--tier full|fast] [--lane LANE] [--bound S] <junit.xml>

Run from anywhere in the repository. The tiers are nextest profiles in
.config/nextest.toml (backlog/docs/commands.md § Test tiers):

  fast   every test except fuzz_robustness and the tests in .config/slow-tests.txt.
         The local hand-back gate (scripts/test-scope.sh).
  ci     the full tier: everything except fuzz_robustness, with a JUnit report.
         CI runs it on every lane (scripts/ci-gate.sh tests).
  heavy  fuzz_robustness and the ignored soaks, run with --run-ignored all.

.config/slow-tests.txt is the source of the slow list. `filter` prints the fast
profile's filterset; `sync` writes it into the generated block of nextest.toml,
and `sync --check` exits 1 when the block differs (the gate-script tests run it).

`ratchet` reads the JUnit report of a nextest run. On the full tier it fails when
a test that the fast tier runs took longer than the bound, and when a listed test
that applies to this lane is missing from the report: either the tier grew a slow
test, or the list names a test that no longer exists. On the fast tier only the
timing is checked, since the listed tests are not run. It prints the fast-tier
tests over the admission threshold as candidates for the list without failing.
Exit status: 0 pass, 1 a check failed, 2 a usage or input error.
"""

import argparse
import fnmatch
from pathlib import Path
import platform
import re
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SLOW_LIST = ROOT / ".config" / "slow-tests.txt"
NEXTEST = ROOT / ".config" / "nextest.toml"
# A test enters the slow list when it takes longer than this on any lane.
ADMISSION_SECONDS = 5.0
# The ratchet fails above this. The gap to the admission threshold absorbs the
# run-to-run variance of shared runners, so a test listed at the threshold's
# edge does not fail every other run.
DEFAULT_BOUND_SECONDS = 10.0
# Never in the fast or full tier: the heavy profile runs it.
HEAVY_BINARIES = ("fuzz_robustness",)
BEGIN = "# BEGIN generated fast-tier filter"
END = "# END generated fast-tier filter"
TEST_NAME = re.compile(r"^[A-Za-z0-9_:]+$")
BINARY_ID = re.compile(r"^[A-Za-z0-9_-]+(::[A-Za-z0-9_/-]+)?$")


class ListError(Exception):
    pass


class Entry:
    def __init__(self, binary, test, only, line):
        self.binary = binary
        self.test = test
        self.only = only
        self.line = line

    @property
    def key(self):
        return (self.binary, self.test)

    def applies_to(self, lane):
        return self.only is None or fnmatch.fnmatchcase(lane, self.only)


def read_list(path=SLOW_LIST):
    """The slow list: `<binary-id> <test-name> [only=<lane glob>]` per line."""
    entries = []
    seen = set()
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        text = raw.split("#", 1)[0].strip()
        if not text:
            continue
        fields = text.split()
        only = None
        if len(fields) == 3 and fields[2].startswith("only="):
            only = fields[2][len("only="):]
        elif len(fields) != 2:
            raise ListError(f"{path.name}:{number}: expected `<binary-id> <test> "
                            f"[only=<lane glob>]`, got {raw!r}")
        binary, test = fields[:2]
        if not BINARY_ID.match(binary) or not TEST_NAME.match(test) or only == "":
            raise ListError(f"{path.name}:{number}: malformed entry {raw!r}")
        if (binary, test) in seen:
            raise ListError(f"{path.name}:{number}: duplicate entry {binary} {test}")
        seen.add((binary, test))
        entries.append(Entry(binary, test, only, number))
    return entries


def fast_filter(entries):
    """The fast profile's filterset: one exclusion line per binary, sorted."""
    by_binary = {}
    for entry in entries:
        by_binary.setdefault(entry.binary, []).append(entry.test)
    lines = [" | ".join(f"binary({name})" for name in HEAVY_BINARIES)]
    lines[0] = f"not ({lines[0]})"
    for binary in sorted(by_binary):
        tests = " | ".join(f"test(={name})" for name in sorted(by_binary[binary]))
        lines.append(f"& not (binary_id(={binary}) & ({tests}))")
    return "\n".join(lines)


def generated_block(entries):
    return "\n".join([
        BEGIN,
        "# Written by `python3 -B scripts/test-tiers.py sync` from .config/slow-tests.txt;",
        "# edit the list, not this block.",
        "[profile.fast]",
        "default-filter = '''",
        fast_filter(entries),
        "'''",
        END,
    ])


def sync(check):
    text = NEXTEST.read_text()
    start = text.find(BEGIN)
    end = text.find(END)
    if start < 0 or end < start:
        print(f"test-tiers: {NEXTEST.name} has no generated block", file=sys.stderr)
        return 2
    end += len(END)
    updated = text[:start] + generated_block(read_list()) + text[end:]
    if updated == text:
        return 0
    if check:
        print(f"test-tiers: the fast-tier filter in {NEXTEST.relative_to(ROOT)} is out of "
              "date; run `python3 -B scripts/test-tiers.py sync`", file=sys.stderr)
        return 1
    NEXTEST.write_text(updated)
    print(f"test-tiers: updated {NEXTEST.relative_to(ROOT)}")
    return 0


def host_lane():
    system = platform.system().lower()
    system = {"darwin": "macos"}.get(system, system)
    machine = platform.machine().lower()
    machine = {"arm64": "aarch64", "amd64": "x86_64"}.get(machine, machine)
    return f"{system}-{machine}"


def read_junit(path):
    """{(binary-id, test): seconds} from a nextest JUnit report."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise ListError(f"cannot read the JUnit report {path}: {error}")
    times = {}
    for case in root.iter("testcase"):
        binary = case.get("classname")
        name = case.get("name")
        if not binary or not name:
            raise ListError(f"{path}: a testcase without classname or name")
        times[(binary, name)] = float(case.get("time") or 0.0)
    if not times:
        raise ListError(f"{path}: the report has no test cases")
    return times


def heavy(key):
    binary = key[0].split("::", 1)[-1]
    return binary in HEAVY_BINARIES


def ratchet(junit, tier, lane, bound):
    entries = read_list()
    listed = {entry.key for entry in entries}
    times = read_junit(junit)
    failures = []
    over = sorted((seconds, key) for key, seconds in times.items()
                  if key not in listed and not heavy(key) and seconds > bound)
    for seconds, (binary, test) in reversed(over):
        failures.append(f"{seconds:8.3f}s  {binary} {test}  (fast tier, over the "
                        f"{bound:g} s bound: add it to .config/slow-tests.txt)")
    if tier == "full":
        for entry in entries:
            if entry.applies_to(lane) and entry.key not in times:
                failures.append(f"   gone   {entry.binary} {entry.test}  (.config/slow-tests.txt:"
                                f"{entry.line} names a test this {lane} run did not report: "
                                "remove it, or mark the lanes it exists on with only=)")
    candidates = sorted((seconds, key) for key, seconds in times.items()
                        if key not in listed and not heavy(key)
                        and ADMISSION_SECONDS < seconds <= bound)
    ran = sum(1 for key in times if key not in listed and not heavy(key))
    print(f"test-tiers: {tier} tier on {lane}: {ran} fast-tier tests checked against "
          f"{bound:g} s, {len(entries)} listed")
    for seconds, (binary, test) in reversed(candidates):
        print(f"  candidate {seconds:8.3f}s  {binary} {test}  (over the "
              f"{ADMISSION_SECONDS:g} s admission threshold)")
    if failures:
        print(f"test-tiers: {len(failures)} ratchet failures:")
        for line in failures:
            print(f"  {line}")
        return 1
    print("test-tiers: ratchet passed")
    return 0


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("filter", help="print the fast profile's filterset")
    sync_parser = commands.add_parser("sync", help="write the filter into nextest.toml")
    sync_parser.add_argument("--check", action="store_true",
                             help="exit 1 instead of writing when the block is stale")
    ratchet_parser = commands.add_parser("ratchet", help="check a nextest JUnit report")
    ratchet_parser.add_argument("junit", type=Path)
    ratchet_parser.add_argument("--tier", choices=("full", "fast"), default="full")
    ratchet_parser.add_argument("--lane", default=host_lane(),
                                help="<os>-<arch>, matched by only= (default: this host, "
                                     "%(default)s)")
    ratchet_parser.add_argument("--bound", type=float, default=DEFAULT_BOUND_SECONDS,
                                help="seconds (default %(default)s)")
    args = parser.parse_args(argv)
    try:
        if args.command == "filter":
            print(fast_filter(read_list()))
            return 0
        if args.command == "sync":
            return sync(args.check)
        return ratchet(args.junit, args.tier, args.lane, args.bound)
    except ListError as error:
        print(f"test-tiers: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
