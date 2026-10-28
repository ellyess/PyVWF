"""Every recorded command line of a study driver or analysis script still parses.

The drivers under ``scripts/`` took their arguments from ``sys.argv`` by
position until 2026-09-18, and their findings documents and docstrings record
the command lines that produced each result. Each now parses with
``argparse`` in a ``cli(argv)`` function, and a path it used to hardcode is a
flag whose default is that path. This file keeps the recorded command lines
working: for each, it runs ``cli`` with the entry functions replaced, and
checks the call they receive, which is the call the ``sys.argv`` version made.

A driver that needs an optional dependency to import is skipped where that
dependency is missing. A driver that fails to import a ``pyvwf`` name fails:
that is the script breaking against the package, not a missing extra.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from support import ROOT

ANALYSIS = ROOT / "scripts" / "analysis"
BACKFILL = Path("output/validation/curve_resolution_backfill_2026-09-11")

# script: [(argv, (entry function, positional args, keyword args)), ...]
RECORDED: dict[str, list[tuple[list[str], tuple[str, tuple, dict]]]] = {
    # Development guards (docs/design/agent-guards.md). The command lines are
    # the ones AGENTS.md and src/pyvwf/AGENTS.md give.
    "scripts/dev/run_locked.py": [
        (
            ["--", "pyvwf-validate", "train", "--region", "configs/regions/nz.toml"],
            (
                "main",
                (["pyvwf-validate", "train", "--region", "configs/regions/nz.toml"],),
                {"allow_dirty": False},
            ),
        ),
        (
            [
                "--allow-dirty",
                "--",
                "pyvwf-validate",
                "train",
                "--region",
                "configs/regions/nz.toml",
            ],
            (
                "main",
                (["pyvwf-validate", "train", "--region", "configs/regions/nz.toml"],),
                {"allow_dirty": True},
            ),
        ),
    ],
    "scripts/dev/stamp.py": [
        (["realdata"], ("main", ("realdata", 1), {})),
        (["pinn"], ("main", ("pinn", 1), {})),
        (["realdata", "--workers", "2"], ("main", ("realdata", 2), {})),
    ],
    # The docstring gives both.
    "scripts/dev/ci_local.py": [
        ([], ("main", (set(), False), {})),
        (["--skip", "docs", "--skip", "pytest"], ("main", ({"docs", "pytest"}, False), {})),
    ],
    # The command docs/design/ai-native-development.md gives.
    # The commands the driver's docstring and docs/findings/agent-critics-prereg.md
    # give, and the docs suite's from docs/findings/agent-critics-docs-prereg.md.
    "scripts/dev/agent_evals/evals.py": [
        (
            [
                "prepare",
                "--label",
                "2026-10-16",
                "--critic-model",
                "claude-opus-5-5",
                "--judge-model",
                "claude-sonnet-5-5",
            ],
            (
                "main",
                (
                    "prepare",
                    "2026-10-16",
                    "claude-opus-5-5",
                    "claude-sonnet-5-5",
                    None,
                    "proposals",
                ),
                {},
            ),
        ),
        (
            ["judge", "--label", "2026-10-16"],
            ("main", ("judge", "2026-10-16", None, None, None, "proposals"), {}),
        ),
        (
            ["score", "--label", "2026-10-16"],
            ("main", ("score", "2026-10-16", None, None, None, "proposals"), {}),
        ),
        (
            [
                "prepare",
                "--suite",
                "docs",
                "--label",
                "<date>",
                "--judge-model",
                "claude-sonnet-5-5",
            ],
            ("main", ("prepare", "<date>", None, "claude-sonnet-5-5", None, "docs"), {}),
        ),
    ],
    # The start-year backfill's own docstring gives both.
    "scripts/region_tools/backfill_gwpt_start_years.py": [
        (
            ["--dry-run"],
            (
                "main",
                (),
                {"out": ROOT / "configs/curation/gwpt_start_years.csv", "dry_run": True},
            ),
        ),
        (
            [],
            (
                "main",
                (),
                {"out": ROOT / "configs/curation/gwpt_start_years.csv", "dry_run": False},
            ),
        ),
    ],
    "scripts/analysis/baseline_bootstrap.py": [
        (
            ["DK", "output/curve_library_study_2026-09-11/baseline_bootstrap"],
            (
                "main",
                ("DK", "output/curve_library_study_2026-09-11/baseline_bootstrap"),
                {"backfill": BACKFILL},
            ),
        ),
    ],
    "scripts/analysis/common_row_rescore.py": [
        (
            ["CL", "output/curve_library_study_2026-09-11/common_row_rescore"],
            (
                "main",
                ("CL", "output/curve_library_study_2026-09-11/common_row_rescore"),
                {"backfill": BACKFILL},
            ),
        ),
        (
            [
                "CL",
                "output/curve_library_study_2026-09-11/common_row_rescore",
                "--joint",
                "output/validation/common_row_rerun_2026-09-11/CL/evaluate-2024-rerun",
                "B",
            ],
            (
                "joint",
                (
                    "CL",
                    "output/curve_library_study_2026-09-11/common_row_rescore",
                    ["output/validation/common_row_rerun_2026-09-11/CL/evaluate-2024-rerun", "B"],
                ),
                {},
            ),
        ),
    ],
    "scripts/analysis/extent_audit.py": [
        (["output/extent_audit_2026-09-12"], ("main", ("output/extent_audit_2026-09-12", []), {})),
        (
            ["output/extent_audit_2026-09-12", "DK", "FR"],
            ("main", ("output/extent_audit_2026-09-12", ["DK", "FR"]), {}),
        ),
    ],
    "scripts/analysis/missing_value_audit.py": [
        (
            ["DK", "output/curve_library_study_2026-09-11/missing_value_audit"],
            (
                "main",
                ("DK", "output/curve_library_study_2026-09-11/missing_value_audit"),
                {"backfill": BACKFILL},
            ),
        ),
    ],
    "scripts/analysis/off_curve_sensitivity.py": [
        (
            ["DK", "output/curve_library_study_2026-09-11/off_curve_sensitivity"],
            (
                "main",
                ("DK", "output/curve_library_study_2026-09-11/off_curve_sensitivity"),
                {"backfill": BACKFILL},
            ),
        ),
    ],
    "scripts/analysis/unit_concentration.py": [
        (
            ["DK", "output/curve_library_study_2026-09-11/unit_concentration"],
            (
                "main",
                ("DK", "output/curve_library_study_2026-09-11/unit_concentration"),
                {"backfill": BACKFILL},
            ),
        ),
    ],
    "scripts/analysis/training_objective_check.py": [
        (
            ["ES", "fixed", "4", "output/curve_library_study_2026-09-11/training_objective_check"],
            (
                "main",
                (
                    "ES",
                    "fixed",
                    "4",
                    "output/curve_library_study_2026-09-11/training_objective_check",
                ),
                {"refresh": Path("output/validation/refresh_2026-08-24")},
            ),
        ),
    ],
}


def load(script: str):
    """Import a script as its own command line would: its directory and the tools on the path."""
    path = ROOT / script
    for extra in (str(path.parent), str(ANALYSIS)):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    spec = importlib.util.spec_from_file_location(f"cli_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        # A missing optional dependency is a skip. A missing pyvwf name is the
        # script breaking against the package, which must fail, not skip.
        if exc.name and (exc.name == "pyvwf" or exc.name.startswith("pyvwf.")):
            raise
        pytest.skip(f"{script} needs {exc.name}")
    return module


CASES = [(script, argv, expected) for script, cases in RECORDED.items() for argv, expected in cases]


@pytest.mark.parametrize(
    "script,argv,expected", CASES, ids=[f"{Path(s).stem}-{i}" for i, (s, _, _) in enumerate(CASES)]
)
def test_recorded_command_line_makes_the_recorded_call(script, argv, expected, monkeypatch):
    module = load(script)
    calls = []
    for name in ("main", "joint"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, lambda *a, _n=name, **k: calls.append((_n, a, k)))
    module.cli(argv)
    assert calls == [expected]


def test_no_driver_reads_its_arguments_by_position_and_every_parser_is_pinned():
    """A driver reads sys.argv only through argparse, and each cli has a recorded case."""
    positional, unpinned = [], []
    for path in sorted((ROOT / "scripts").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text()
        if "__main__" not in text:
            continue
        if "sys.argv[" in text:
            positional.append(rel)
        if "\ndef cli(" in text and rel not in RECORDED:
            unpinned.append(rel)
    assert positional == []
    assert unpinned == []


def test_a_broken_vwf_import_fails_rather_than_skips(tmp_path):
    script = tmp_path / "broken_driver.py"
    script.write_text("from pyvwf.harness.driver import no_such_name\n")
    try:
        load(str(script))
    except ImportError:
        return
    except pytest.skip.Exception:
        pytest.fail("a broken pyvwf import was skipped, which CI reports as green")
    pytest.fail("the broken import raised nothing")


def test_a_missing_optional_dependency_skips(tmp_path):
    script = tmp_path / "optional_driver.py"
    script.write_text("import no_such_optional_dependency_xyz\n")
    with pytest.raises(pytest.skip.Exception):
        load(str(script))
