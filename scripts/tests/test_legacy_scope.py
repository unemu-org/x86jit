"""Keep the legacy-only scope of scripts/test-scope.sh sound.

That scope skips the Long64, Compat32 and JIT suites when every changed path is
legacy (the list and its call-graph evidence are in scripts/test-scope.sh). The
scope is sound only while shared engine code enters legacy modules through the
dispatch points reviewed below, and while its test set includes every test that
selects Real16 or a CPU model.

The checks are textual: comments are removed and nothing is type-resolved. They
find path references (`crate::lift::legacy::lower`, a bare `legacy::` in the
parent module's directory, `mod` declarations, `#[path]` includes, re-exported
names) and calls of the methods that legacy files add to a shared type. They
cannot follow a trait method called through a generic parameter, so a legacy
file may implement traits only for types that legacy files define.
"""

from collections import defaultdict
import os
from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "test-scope.sh"
CRATES = ("x86jit-core", "x86jit-cranelift", "x86jit-elf", "x86jit-tests", "x86jit-bench")
ENGINE_SRC = ("x86jit-core/src/", "x86jit-cranelift/src/", "x86jit-elf/src/")
CORE_SRC = "x86jit-core/src/"

# Shared engine code that enters a legacy module, and what it uses there. An
# entry is `<legacy module>::<item>`, `<legacy module>::mod` for the declaration
# or `<Type>::<method>` for a method that a legacy file adds to a shared type.
# Each use is reached only from Real16 or a Machine profile, except the one
# pinned in PINNED_BODIES:
#   lift/mod.rs: lift_insn when ops.legacy_context() is set or the mode is
#     Real16; with_segment when legacy_context is set. LiftBuffer::model_profile
#     is None for application lifting.
#   interp/mod.rs: the walk_ops arms for IrOp::Legacy and the Real16-only IR
#     ops, the MACHINE, DIRECT20 and WRAPPED instantiations, and the
#     instruction adapter, which `()` disables for application walks;
#     finish_instruction_fault only for a block that contains IrOp::Legacy.
#     Only legacy lowering emits these ops (lift/mod.rs above).
#   interp/control.rs: exec_loop_cx executes LoopCx, a Real16-only op.
#   interp/integer.rs: exec_load_retry when ir.mode is Real16.
#   legacy.rs: restore_flags, called only from interp/legacy.rs and
#     interp/legacy/tasks.rs; bus_model and physical_mask read the catalogue
#     in const blocks, at compile time.
#   vm.rs: the run entries select DIRECT20 (CONTEXT) for a twenty-bit
#     CpuModel; set_cpu_model and the CpuModel accessors serve an
#     embedder-selected model; has_pending_nmi reads NmiNesting only after its
#     ExecutionProfile::Machine check.
#   vm/legacy/nmi.rs: the NMI helpers of the Machine dispatcher. They are
#     called only in the KIND >= MACHINE_DISPATCH branch or from Vcpu methods
#     that return early outside ExecutionProfile::Machine.
#   vm/legacy.rs: run_dispatch is also the application dispatch loop, so this
#     file is shared. Its legacy calls sit in the KIND >= MACHINE_DISPATCH
#     branch or in methods that need legacy_execution, which only a Machine
#     profile creates (Execution, resolve_direct*, deliver_legacy_event,
#     deliver_event20, legacy_context_for_run). Its prologue calls
#     alias_context for every KIND; the application branch never reads the
#     value.
#   lib.rs, ir/verify.rs: declarations and public re-exports (legacy_probes is
#     compiled for unit tests only).
REVIEWED_ENTRY_POINTS = {
    "x86jit-core/src/interp/control.rs": {
        "interp::legacy::execute_loop", "lift::legacy::alias_context",
    },
    "x86jit-core/src/interp/integer.rs": {
        "interp::legacy::unified_read_retry",
    },
    "x86jit-core/src/interp/mod.rs": {
        "interp::i8086::Nec", "interp::i8086::bus_evidence", "interp::i8086::interrupt",
        "interp::i8086::mod", "interp::i8086::nec", "interp::i8086::physical",
        "interp::i8086::port_string_fault", "interp::i8086::primitive",
        "interp::i8086::string", "interp::i8086::string_evidence",
        "interp::i8086::string_load", "interp::i8086::string_store",
        "interp::legacy::direct_stack_address", "interp::legacy::execute_loop_op",
        "interp::legacy::execute_shared_op", "interp::legacy::execute_shared_string",
        "interp::legacy::machine_specific", "interp::legacy::mmio_evidence",
        "interp::legacy::mod", "interp::legacy::scalar_limit_access",
        "interp::legacy::selector", "interp::legacy::stack_fault_access",
        "lift::legacy::shared_helper",
    },
    "x86jit-core/src/ir/verify.rs": {
        "ir::verify::legacy_probes::mod",
    },
    "x86jit-core/src/legacy.rs": {
        "model::CpuModel", "model::EARLY_FLAGS_FIXED_ZERO",
    },
    "x86jit-core/src/lib.rs": {
        "model::CpuModel", "model::ModelIsa", "model::ModelProfile",
        "model::ModelQuirks", "model::ModelSystem", "model::ModelTiming",
        "model::NmiNesting", "model::mod",
    },
    "x86jit-core/src/lift/mod.rs": {
        "lift::i8086::mod", "lift::legacy::ContextOps", "lift::legacy::alias_context",
        "lift::legacy::alias_supported", "lift::legacy::direct_address",
        "lift::legacy::lower", "lift::legacy::mod", "lift::legacy::segment",
        "lift::legacy::shared_family", "model::ModelProfile",
    },
    "x86jit-core/src/vm.rs": {
        "lift::i8086::CONTEXT", "model::CpuModel", "model::NmiNesting",
        "vm::i8086::mod", "vm::nec::mod",
    },
    "x86jit-core/src/vm/legacy.rs": {
        "Vcpu::nec_boundary", "interp::i8086::finish_port_input",
        "interp::i8086::interrupt", "interp::legacy::BlockWalk",
        "interp::legacy::Progress", "interp::legacy::deliver_interrupt",
        "interp::legacy::execute_block", "interp::legacy::execute_checked",
        "interp::legacy::execute_shared20", "interp::legacy::finish_port_input",
        "interp::legacy::tasks", "lift::i8086::CONTEXT", "lift::i8086::word_bus",
        "lift::legacy::alias_context",
        "lift::legacy::instruction", "lift::legacy::shared_helper", "model::CpuModel",
        "model::decode", "model::profile_for_generation", "vm::i8086::resolve",
    },
    "x86jit-core/src/vm/legacy/nmi.rs": {
        "interp::legacy::Progress", "model::CpuModel", "model::NmiNesting",
        "model::profile_for_generation",
    },
}

# Legacy functions that Long64 and Compat32 execution calls, pinned by their
# whitespace-normalised source. run_dispatch (vm/legacy.rs) calls alias_context
# on every dispatch entry. Changing a pinned function is a shared change: review
# the application callers, then update the pin or move the call.
PINNED_BODIES = {
    ("x86jit-core/src/lift/legacy.rs", "alias_context"):
        "pub(crate) fn alias_context() -> MachineContext { "
        "crate::legacy::REAL16_ALIAS.context }",
}

# Shared types that legacy files may extend with inherent methods. Their callers
# in shared code are recorded in REVIEWED_ENTRY_POINTS.
REVIEWED_SHARED_SELF_TYPES = {"Vcpu"}

# A test that names one of these selects legacy execution: Real16, a CPU model,
# a Machine context or the NEC 8080 emulation interface.
LEGACY_MARKERS = re.compile(
    r"\b(?:Real16|CpuModel|LegacyCpuModel|CpuGeneration|ExecutionProfile"
    r"|set_cpu_model|set_legacy_cpu_model|legacy_real_entry|derive_legacy_context"
    r"|I8080\w*)\b")


def classify(paths):
    """{path: class} according to scripts/test-scope.sh."""
    result = subprocess.run(
        ["bash", str(SCRIPT), "--classify", *paths],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    return {path: cls for cls, path in
            (line.split("\t", 1) for line in result.stdout.splitlines())}


def legacy_plan():
    result = subprocess.run(
        ["bash", str(SCRIPT), "--plan", "legacy-only"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    return result.stdout


def rust_files():
    for top in CRATES:
        for directory, subdirs, files in os.walk(ROOT / top):
            subdirs[:] = sorted(d for d in subdirs if d != "target")
            for name in sorted(files):
                if name.endswith(".rs"):
                    yield (Path(directory) / name).relative_to(ROOT).as_posix()


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def module_of(path):
    """Module path of a core source file: `x86jit-core/src/a/b.rs` -> `a::b`."""
    parts = path[len(CORE_SRC):-len(".rs")].split("/")
    if parts[-1] in ("mod", "lib"):
        parts.pop()
    return "::".join(parts)


def items(group):
    """Leading item names of a `use` tail: `{a, b::c, self}` or `name`."""
    if group.startswith("{"):
        return [re.split(r"[\s:]", part.strip())[0]
                for part in group[1:-1].split(",") if part.strip()]
    return [group]


def impl_blocks(text):
    """(header, self type name, body, indent) of every impl block, at any depth."""
    blocks = []
    for match in re.finditer(r"(?m)^([ \t]*)(?:unsafe\s+)?impl\b([^{;]*)\{", text):
        indent, header = match.group(1), " ".join(match.group(2).split())
        end = re.compile(r"(?m)^" + re.escape(indent) + r"\}").search(text, match.end())
        body = text[match.end():end.start() if end else len(text)]
        if " for " in header:
            target = header.rsplit(" for ", 1)[1]
        else:
            target = re.sub(r"^<.*?>\s*", "", header)
        target = re.sub(r"^&\s*(?:'\w+\s+)?(?:mut\s+)?", "", target.strip())
        name = re.split(r"[<\s]", target.split("::")[-1])[0]
        blocks.append((header, name, body, indent))
    return blocks


class Tree:
    """The workspace's Rust files, split by test-scope.sh into legacy and shared."""

    def __init__(self):
        self.files = list(rust_files())
        classes = classify(self.files)
        self.legacy = sorted(p for p in self.files if classes[p] == "legacy")
        self.shared_engine = sorted(p for p in self.files if classes[p] != "legacy"
                                    and p.startswith(ENGINE_SRC))
        core = {module_of(p) for p in self.legacy if p.startswith(CORE_SRC)}
        # A root is a legacy module whose parent module is shared.
        self.roots = sorted(m for m in core if m.rpartition("::")[0] not in core)
        self.code = {p: strip_comments(read(p)) for p in self.files}


def legacy_types(tree):
    names = set()
    for path in tree.legacy:
        names |= set(re.findall(r"\b(?:struct|enum|union|type|trait)\s+(\w+)",
                                tree.code[path]))
    return names


def shared_methods(tree):
    """{(Type, method)} that legacy files add to the reviewed shared types."""
    methods = set()
    for path in tree.legacy:
        for _, self_type, body, indent in impl_blocks(tree.code[path]):
            if self_type in REVIEWED_SHARED_SELF_TYPES:
                inner = re.escape(indent) + "    "
                methods |= {(self_type, m) for m in re.findall(
                    r"(?m)^" + inner + r"(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?"
                    r"(?:unsafe\s+)?fn\s+(\w+)", body)}
    return methods


def references(tree):
    """{shared engine file: {reference}} into legacy modules."""
    found = defaultdict(set)
    exported = set()
    for root in tree.roots:
        parent, _, leaf = root.rpartition("::")
        absolute = re.compile(r"\b(?:crate|x86jit_core)::" + re.escape(root)
                              + r"\b(?:::(\{[^}]*\}|\w+|\*))?")
        bare = re.compile(r"(?<![\w:])(?:super::|self::)?" + leaf
                          + r"::(\{[^}]*\}|\w+|\*)")
        declared = re.compile(r"\bmod\s+" + leaf + r"\s*[;{]")
        for path in tree.shared_engine:
            text = tree.code[path]
            patterns = [absolute]
            # The parent module's file and its other children (through
            # `use super::*`) name the module without a prefix.
            if path.startswith(CORE_SRC) and parent in (
                    module_of(path), module_of(path).rpartition("::")[0]):
                patterns.append(bare)
                if declared.search(text):
                    found[path].add(f"{root}::mod")
            for pattern in patterns:
                for m in pattern.finditer(text):
                    found[path] |= {f"{root}::{i}" for i in items(m.group(1) or "*")}
                for statement in re.findall(r"\bpub(?:\([^)]*\))?\s+use\s+([^;]+);", text):
                    for m in pattern.finditer(statement):
                        exported |= {(root, i) for i in items(m.group(1) or "*")}
    for root, name in exported:
        word = re.compile(r"\b" + re.escape(name) + r"\b")
        for path in tree.shared_engine:
            if word.search(tree.code[path]):
                found[path].add(f"{root}::{name}")
    for self_type, method in shared_methods(tree):
        call = re.compile(r"(?:\.|::)" + method + r"\s*(?:::\s*<[^()]*>\s*)?\(")
        for path in tree.shared_engine:
            if call.search(tree.code[path]):
                found[path].add(f"{self_type}::{method}")
    for path in tree.shared_engine:
        for include in re.findall(r'#\[path\s*=\s*"([^"]+)"\]', read(path)):
            target = os.path.normpath(os.path.join(os.path.dirname(path), include))
            if target in tree.legacy:
                found[path].add(f"#[path] {target}")
    return {path: refs for path, refs in found.items() if refs}


def test_name(path):
    """Selection unit of an integration-test file, else None.

    `<name>` for a binary `x86jit-*/tests/<name>.rs`, `<group>/<name>` for a
    module `x86jit-*/tests/<group>/<name>.rs` of a grouped binary, whose crate
    root is `<group>/main.rs` (scripts/test-scope.sh, legacy_units).
    """
    match = re.fullmatch(r"x86jit-[\w-]+/tests/(\w+)\.rs", path)
    if match:
        return match.group(1)
    match = re.fullmatch(r"(x86jit-[\w-]+/tests/(\w+))/(\w+)\.rs", path)
    if match and match.group(3) != "main" and (ROOT / match.group(1) / "main.rs").is_file():
        return f"{match.group(2)}/{match.group(3)}"
    return None


def binary_names(files):
    """Integration-test binary names: `<name>.rs` files and `<name>/main.rs` groups."""
    names = set()
    for path in files:
        match = re.fullmatch(r"x86jit-[\w-]+/tests/(\w+)(?:\.rs|/main\.rs)", path)
        if match:
            names.add(match.group(1))
    return names


def selected_units(plan_line):
    """{unit} that a legacy-only nextest command selects, and its --test binaries.

    The -E filterset is `binary(=<name>)` for a whole binary and
    `(binary(=<group>) & test(/^(<a>|<b>)::/))` for modules, joined by ` | `.
    """
    binaries = set(re.findall(r"--test (\w+)", plan_line))
    expression = re.search(r"-E '([^']*)'", plan_line).group(1)
    units = set()
    for term in expression.split(" | "):
        whole = re.fullmatch(r"binary\(=(\w+)\)", term)
        modules = re.fullmatch(r"\(binary\(=(\w+)\) & test\(/\^\(([\w|]+)\)::/\)\)", term)
        if whole:
            units.add(whole.group(1))
        elif modules:
            units |= {f"{modules.group(1)}/{m}" for m in modules.group(2).split("|")}
        else:
            raise AssertionError(f"unexpected filter term {term!r}")
    return units, binaries


def selects_legacy(text):
    return any(LEGACY_MARKERS.search(line) and "unreachable!" not in line
               for line in strip_comments(text).splitlines())


class LegacyScopeSoundness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = Tree()

    def test_shared_code_enters_legacy_modules_only_at_reviewed_points(self):
        found = references(self.tree)
        new = {path: sorted(refs - REVIEWED_ENTRY_POINTS.get(path, set()))
               for path, refs in found.items()}
        new = {path: refs for path, refs in new.items() if refs}
        self.assertEqual(new, {}, (
            "shared code gained references into legacy modules. If each one is "
            "reached only from Real16 or a Machine profile, add it to "
            "REVIEWED_ENTRY_POINTS with the reason; otherwise remove the module "
            "from legacy_paths in scripts/test-scope.sh."))
        stale = {path: sorted(refs - found.get(path, set()))
                 for path, refs in REVIEWED_ENTRY_POINTS.items()}
        stale = {path: refs for path, refs in stale.items() if refs}
        self.assertEqual(stale, {}, "REVIEWED_ENTRY_POINTS lists references "
                         "that no longer exist; remove them")

    def test_legacy_files_implement_only_legacy_types(self):
        own = legacy_types(self.tree) | REVIEWED_SHARED_SELF_TYPES
        foreign = sorted(f"{path}: impl {header}"
                         for path in self.tree.legacy
                         for header, self_type, _, _ in impl_blocks(self.tree.code[path])
                         if self_type not in own)
        self.assertEqual(foreign, [], (
            "a legacy file implements a shared type, which shared code can reach "
            "without naming the legacy module. Move the impl to a shared file, or "
            "review its callers and add the type to REVIEWED_SHARED_SELF_TYPES."))

    def test_legacy_functions_on_the_application_path_are_unchanged(self):
        for (path, name), pinned in PINNED_BODIES.items():
            with self.subTest(function=name):
                self.assertIn(path, self.tree.legacy)
                text = self.tree.code[path]
                start = re.search(r"(?m)^([ \t]*)(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?fn\s+"
                                  + name + r"\b", text)
                self.assertIsNotNone(start, f"{name} not found in {path}")
                end = re.compile(r"(?m)^" + start.group(1) + r"\}").search(text, start.end())
                body = " ".join(text[start.start():end.end()].split())
                self.assertEqual(body, pinned, (
                    f"{name} runs on Long64 and Compat32 dispatch; a change to it is "
                    "a shared change. Review its application callers before "
                    "updating PINNED_BODIES."))

    def test_legacy_files_have_no_doctests(self):
        # The legacy-only scope skips doctests.
        fenced = [path for path in self.tree.legacy if "```" in read(path)]
        self.assertEqual(fenced, [])

    def test_legacy_test_set_covers_tests_that_select_legacy_execution(self):
        plan = legacy_plan()
        tests = next(line for line in plan.splitlines() if "--test " in line)
        selected, binaries = selected_units(tests)
        existing = ({test_name(p) for p in self.tree.files} - {None}) | binary_names(
            self.tree.files)
        self.assertEqual(selected - existing, set(), "the legacy-only plan names test "
                         "targets that do not exist")
        self.assertEqual(binaries, {unit.split("/")[0] for unit in selected},
                         "the legacy-only plan builds binaries it does not select from")
        unit_tests = next(line for line in plan.splitlines() if line.endswith(" --lib"))
        crates = {p.split("/")[0] for p in self.tree.files
                  if "/src/" in p and selects_legacy(read(p))}
        self.assertEqual(crates - set(re.findall(r"-p (\S+)", unit_tests)), set(),
                         "these crates' unit tests select legacy execution but the "
                         "legacy-only scope does not run them")
        helpers = {Path(p).stem for p in self.tree.files
                   if p.startswith("x86jit-tests/src/") and selects_legacy(read(p))}
        uses_helper = re.compile(r"\bx86jit_tests::(?:\{[^}]*\b)?(" + "|".join(
            map(re.escape, sorted(helpers))) + r")\b")
        missing = []
        for path in self.tree.files:
            name = test_name(path)
            if name is None or name in selected or name.split("/")[0] in selected:
                continue
            text = read(path)
            includes = [os.path.normpath(os.path.join(os.path.dirname(path), p))
                        for p in re.findall(r'#\[path\s*=\s*"([^"]+)"\]', text)]
            if (selects_legacy(text) or uses_helper.search(strip_comments(text))
                    or any(selects_legacy(read(p)) for p in includes
                           if p.startswith(("x86jit-tests/tests/", "x86jit-core/tests/",
                                            "x86jit-cranelift/tests/")))):
                missing.append(path)
        self.assertEqual(missing, [], (
            "these tests select Real16 or a CPU model but the legacy-only scope "
            "does not run them; add them to legacy_tests in scripts/test-scope.sh"))

    def test_legacy_support_files_are_included_only_by_legacy_files(self):
        legacy = set(self.tree.legacy)
        includers = []
        for path in self.tree.files:
            if path in legacy:
                continue
            for include in re.findall(r'#\[path\s*=\s*"([^"]+)"\]', read(path)):
                target = os.path.normpath(os.path.join(os.path.dirname(path), include))
                if target in legacy:
                    includers.append(f"{path} -> {target}")
        self.assertEqual(includers, [])


if __name__ == "__main__":
    unittest.main()
