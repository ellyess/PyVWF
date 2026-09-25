"""Run a local-only test suite and stamp the code it was run against.

Two suites cannot be checked by CI. The `realdata` pins skip wherever their
inputs are absent, CI included. The physics-informed tests need the `pinn`
extra, which CI does not install. This script runs one suite in full and
writes `output/.stamp-<suite>.json`, recording a fingerprint of the code the
suite covers as it was on disk. The Claude Code hook
`.claude/hooks/guard_local_suites.py` refuses a commit that stages covered code
unless the staged content matches a passing stamp for that suite.
`docs/design/agent-guards.md` says why.

    python scripts/dev/stamp.py realdata
    python scripts/dev/stamp.py pinn

A suite always runs in full: a stamp is a claim about the whole suite, so no
pytest selection is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple


# A NamedTuple rather than a dataclass: tests load this file by path without
# registering it in sys.modules, which a dataclass needs at class creation.
class Suite(NamedTuple):
    covered: tuple[str, ...]
    pytest_args: tuple[str, ...]
    requires: tuple[str, ...] = ()


# The realdata paths are also named in src/vwf/AGENTS.md and CONTRIBUTING.md;
# change them together. The pinn suite requires its extra outright, because
# its tests skip without it and a run of skips would stamp nothing tested.
SUITES = {
    "realdata": Suite(
        # Everything the pins reach: the harness, the adapters and dataset
        # processing they call, and the numerics below them. Until 2026-09-24
        # this was harness/, metrics.py, correction.py and data.py only, so a
        # change to wind.py that moved two published rows needed no stamp.
        covered=(
            "src/vwf/harness/",
            "src/vwf/sources/",
            "src/vwf/datasets/",
            "src/vwf/extensions/",
            "src/vwf/loaders/",
            "src/vwf/metrics.py",
            "src/vwf/correction.py",
            "src/vwf/data.py",
            "src/vwf/wind.py",
            "src/vwf/curves.py",
            "src/vwf/clustering.py",
            "src/vwf/country_level.py",
            "src/vwf/sampling.py",
            "src/vwf/config.py",
            "src/vwf/time_utils.py",
            "src/vwf/geospatial.py",
            "src/vwf/utils.py",
            "src/vwf/provenance.py",
        ),
        pytest_args=("-m", "realdata"),
    ),
    "pinn": Suite(
        covered=("src/vwf/pinn/",),
        pytest_args=(
            "tests/test_pinn_physics.py",
            "tests/test_pinn_era5_record.py",
            "tests/test_pinn_country_cache.py",
            "tests/test_pinn_level_spatial_gwa.py",
        ),
        requires=("torch", "rasterio"),
    ),
}


def stamp_path(name: str) -> Path:
    return Path(f"output/.stamp-{name}.json")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout


def _digest(entries: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(entries)).encode()).hexdigest()


def worktree_fingerprint(root: Path, covered: tuple[str, ...]) -> str:
    """Fingerprint of the covered files as they are on disk, untracked included."""
    listed = _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "--", *covered)
    paths = [p for p in listed.splitlines() if (root / p).is_file()]
    if not paths:
        return _digest([])
    blobs = _git(root, "hash-object", "--", *paths).split()
    return _digest([f"{blob} {path}" for blob, path in zip(blobs, paths)])


def index_fingerprint(root: Path, covered: tuple[str, ...]) -> str:
    """Fingerprint of the covered files as staged, comparable to the worktree one."""
    entries = []
    for line in _git(root, "ls-files", "--stage", "--", *covered).splitlines():
        meta, path = line.split("\t", 1)
        entries.append(f"{meta.split()[1]} {path}")
    return _digest(entries)


def staged_covered_paths(root: Path, covered: tuple[str, ...]) -> list[str]:
    """Covered paths that differ between HEAD and the index."""
    return _git(root, "diff", "--cached", "--name-only", "--", *covered).split()


def main(name: str) -> int:
    """Run one suite in full and write its stamp; return pytest's exit code."""
    suite = SUITES[name]

    missing = [m for m in suite.requires if importlib.util.find_spec(m) is None]
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
    command = [sys.executable, "-m", "pytest", "-rs", *suite.pytest_args]
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

    summary = next((ln for ln in reversed(lines) if ln.startswith("=")), "")
    stamp = {
        "suite": name,
        "fingerprint": before,
        "head": _git(root, "rev-parse", "HEAD").strip(),
        "exit_code": code,
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
    args = parser.parse_args(argv)
    return main(args.suite)


if __name__ == "__main__":
    raise SystemExit(cli())
