"""Run a local-only test suite and stamp the code it was run against.

Two suites are checked here before a commit. The `realdata` pins skip wherever
their inputs are absent, CI included, so only a local run checks them. The
physics-informed tests need the `pinn` extra, which only CI's extras job
installs, so a local run is the check before a push. This script runs one suite in full and
writes `output/.stamp-<suite>.json`, recording a fingerprint of the code the
suite covers as it was on disk. The Claude Code hook
`.claude/hooks/guard_local_suites.py` refuses a commit that stages covered code
unless the staged content matches a passing stamp for that suite.
`docs/design/agent-guards.md` says why.

    python scripts/dev/stamp.py realdata
    python scripts/dev/stamp.py pinn
    python scripts/dev/stamp.py realdata --workers 2

A suite always runs in full: a stamp is a claim about the whole suite, so no
pytest selection is accepted. `--workers N` runs it in N pytest-xdist
processes, one test file per process; serial is the default, and the count is
recorded in the stamp. A suite accepts only the counts it has been measured
at (`max_workers`). On 2026-10-16 the realdata suite passed 43 of 43 serially
in 728 s, with 2 workers in 404 s and with 3 in 340 s, on a 16 GB machine
where 3 left under 1 GB of swap free. `docs/design/agent-guards.md` has the
table.

The fingerprint ignores what cannot change a result. A covered `.py` file is
fingerprinted by its parsed syntax tree, with module, class and function
docstrings removed, so an edit to a comment, a `# type: ignore`, a docstring
or the blank lines keeps an existing stamp valid. Any other edit moves it,
since the tree differs. A file that does not parse, and every covered file
that is not `.py`, is fingerprinted by its git blob, byte for byte. The tree
is written out by `_canonical`, not `ast.dump`, whose text differs between
Python versions, because the hook may run under another interpreter than the
stamp did. This is safe because no covered code reads its own source or a
docstring to compute anything; the one `__doc__` it reads is an argparse
epilog. `docs/design/agent-guards.md` says more.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple


# A NamedTuple rather than a dataclass: tests load this file by path without
# registering it in sys.modules, which a dataclass needs at class creation.
class Suite(NamedTuple):
    covered: tuple[str, ...]
    pytest_args: tuple[str, ...]
    requires: tuple[str, ...] = ()
    #: The most xdist workers the suite has passed in full under.
    max_workers: int = 1


# The realdata paths are also named in src/pyvwf/AGENTS.md and CONTRIBUTING.md;
# change them together. The pinn suite requires its extra outright, because
# its tests skip without it and a run of skips would stamp nothing tested.
SUITES = {
    "realdata": Suite(
        # Everything the pins reach: the harness, the adapters and dataset
        # processing they call, and the numerics below them. Until 2026-09-24
        # this was harness/, metrics.py, correction.py and data.py only, so a
        # change to wind.py that moved two published rows needed no stamp.
        covered=(
            "src/pyvwf/harness/",
            "src/pyvwf/sources/",
            "src/pyvwf/datasets/",
            "src/pyvwf/extensions/",
            "src/pyvwf/loaders/",
            "src/pyvwf/metrics.py",
            "src/pyvwf/correction.py",
            "src/pyvwf/data.py",
            "src/pyvwf/wind.py",
            "src/pyvwf/curves.py",
            "src/pyvwf/clustering.py",
            "src/pyvwf/era5.py",
            "src/pyvwf/country_level.py",
            "src/pyvwf/sampling.py",
            "src/pyvwf/config.py",
            "src/pyvwf/time_utils.py",
            "src/pyvwf/geospatial.py",
            "src/pyvwf/utils.py",
            "src/pyvwf/provenance.py",
        ),
        pytest_args=("-m", "realdata"),
        max_workers=3,
    ),
    "pinn": Suite(
        covered=("src/pyvwf/pinn/",),
        pytest_args=(
            "tests/test_pinn_physics.py",
            "tests/test_pinn_era5_record.py",
            "tests/test_pinn_country_cache.py",
            "tests/test_pinn_level_spatial_gwa.py",
            "tests/test_pinn_cache_override.py",
        ),
        requires=("torch", "rasterio"),
    ),
}


#: pytest's result line, with or without the `=` rule around it: counts such as
#: "43 passed, 1232 deselected, 33 warnings", or "no tests ran", then the time.
#: Under `-q` (the ini sets it) the rule is absent, so the last `=` line is a
#: section header such as "warnings summary", not the result.
RESULT_LINE = re.compile(r"(?:no tests ran|\d+ \w+(?:, \d+ \w+)*) in \d+(?:\.\d+)?s\b")


def stamp_path(name: str) -> Path:
    return Path(f"output/.stamp-{name}.json")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout


def _digest(entries: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(entries)).encode()).hexdigest()


_DOC_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _canonical(node: object) -> str:
    """The tree as text: each node's type and fields, with no line or column numbers.

    Fields that are None or empty are left out, so a field a newer Python added
    with an empty default reads the same as its absence under an older one.
    """
    if isinstance(node, ast.AST):
        fields = [
            f"{name}={_canonical(value)}"
            for name, value in ast.iter_fields(node)
            if value is not None and value != []
        ]
        return f"{type(node).__name__}({', '.join(fields)})"
    if isinstance(node, list):
        return "[" + ", ".join(_canonical(item) for item in node) + "]"
    return repr(node)


def _code_digest(source: bytes) -> str | None:
    """A digest of the code in a Python source, docstrings removed; None if it does not parse."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        if (
            isinstance(node, _DOC_OWNERS)
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
    return hashlib.sha256(_canonical(tree).encode()).hexdigest()


def _entry(path: str, blob: str, read) -> str:
    """One file's line in the fingerprint: its code digest for `.py`, else its blob."""
    if path.endswith(".py"):
        digest = _code_digest(read())
        if digest is not None:
            return f"ast:{digest} {path}"
    return f"{blob} {path}"


def _blob_contents(root: Path, blobs: list[str]) -> dict[str, bytes]:
    """The content of each blob, read in one `git cat-file --batch` call."""
    if not blobs:
        return {}
    out = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=root,
        input="".join(f"{b}\n" for b in blobs).encode(),
        capture_output=True,
        check=True,
    ).stdout
    contents, pos = {}, 0
    for blob in blobs:
        header_end = out.index(b"\n", pos)
        size = int(out[pos:header_end].split()[2])
        contents[blob] = out[header_end + 1 : header_end + 1 + size]
        pos = header_end + 1 + size + 1
    return contents


def worktree_fingerprint(root: Path, covered: tuple[str, ...]) -> str:
    """Fingerprint of the covered files as they are on disk, untracked included."""
    listed = _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "--", *covered)
    paths = [p for p in listed.splitlines() if (root / p).is_file()]
    if not paths:
        return _digest([])
    blobs = _git(root, "hash-object", "--", *paths).split()
    return _digest(
        [
            _entry(path, blob, lambda path=path: (root / path).read_bytes())
            for blob, path in zip(blobs, paths)
        ]
    )


def index_fingerprint(root: Path, covered: tuple[str, ...]) -> str:
    """Fingerprint of the covered files as staged, comparable to the worktree one."""
    staged = []
    for line in _git(root, "ls-files", "--stage", "--", *covered).splitlines():
        meta, path = line.split("\t", 1)
        staged.append((meta.split()[1], path))
    contents = _blob_contents(root, sorted({b for b, p in staged if p.endswith(".py")}))
    return _digest([_entry(path, blob, lambda blob=blob: contents[blob]) for blob, path in staged])


def staged_covered_paths(root: Path, covered: tuple[str, ...]) -> list[str]:
    """Covered paths that differ between HEAD and the index."""
    return _git(root, "diff", "--cached", "--name-only", "--", *covered).split()


def main(name: str, workers: int = 1) -> int:
    """Run one suite in full and write its stamp; return pytest's exit code."""
    suite = SUITES[name]
    if not 1 <= workers <= suite.max_workers:
        print(
            f"The {name} suite has been measured with at most {suite.max_workers} "
            f"worker(s), not {workers}. No stamp written.",
            file=sys.stderr,
        )
        return 2

    requires = suite.requires + (("xdist",) if workers > 1 else ())
    missing = [m for m in requires if importlib.util.find_spec(m) is None]
    if missing:
        print(
            f"The {name} suite needs {', '.join(missing)}; its tests would skip. "
            "Install the extra and rerun. No stamp written.",
            file=sys.stderr,
        )
        return 2

    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
    before = worktree_fingerprint(root, suite.covered)

    started = datetime.now(timezone.utc)
    log_path = root / "output" / f"stamp-{name}-{started:%Y%m%dT%H%M%SZ}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    parallel = ["-n", str(workers), "--dist", "loadfile"] if workers > 1 else []
    command = [sys.executable, "-m", "pytest", "-rs", *parallel, *suite.pytest_args]
    lines: list[str] = []
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            lines.append(line.rstrip("\n"))
        code = proc.wait()

    if worktree_fingerprint(root, suite.covered) != before:
        print(
            "\nThe covered code changed while the suite ran. No stamp written; rerun.",
            file=sys.stderr,
        )
        return 1

    summary = next((ln for ln in reversed(lines) if RESULT_LINE.match(ln.strip("= "))), "")
    stamp = {
        "suite": name,
        "fingerprint": before,
        "head": _git(root, "rev-parse", "HEAD").strip(),
        "exit_code": code,
        "workers": workers,
        "summary": summary.strip("= "),
        "log": str(log_path.relative_to(root)),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (root / stamp_path(name)).write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    print(f"\nStamp written: {stamp_path(name)} ({stamp['summary']}). Read the skips above.")
    return code


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("suite", choices=sorted(SUITES), help="the suite to run")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="pytest-xdist processes, one test file each (default 1, serial)",
    )
    args = parser.parse_args(argv)
    return main(args.suite, args.workers)


if __name__ == "__main__":
    raise SystemExit(cli())
