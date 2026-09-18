"""Every recorded command line of a study driver or analysis script still parses.

The drivers under ``scripts/`` took their arguments from ``sys.argv`` by
position until 2026-09-18, and their findings documents and docstrings record
the command lines that produced each result. Each now parses with
``argparse`` in a ``cli(argv)`` function, and a path it used to hardcode is a
flag whose default is that path. This file keeps the recorded command lines
working: for each, it runs ``cli`` with the entry functions replaced, and
checks the call they receive, which is the call the ``sys.argv`` version made.

A driver that needs an optional dependency to import is skipped where that
dependency is missing.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "scripts" / "analysis"
BACKFILL = Path("output/validation/curve_resolution_backfill_2026-09-11")
POOL = Path("output/pyvwf_to_grid/all_corrections_centroids.csv")

# script: [(argv, (entry function, positional args, keyword args)), ...]
RECORDED: dict[str, list[tuple[list[str], tuple[str, tuple, dict]]]] = {
    "scripts/analysis/baseline_bootstrap.py": [
        (["DK", "output/curve_library_study_2026-09-11/baseline_bootstrap"],
         ("main", ("DK", "output/curve_library_study_2026-09-11/baseline_bootstrap"),
          {"backfill": BACKFILL})),
    ],
    "scripts/analysis/common_row_rescore.py": [
        (["CL", "output/curve_library_study_2026-09-11/common_row_rescore"],
         ("main", ("CL", "output/curve_library_study_2026-09-11/common_row_rescore"),
          {"backfill": BACKFILL})),
        (["CL", "output/curve_library_study_2026-09-11/common_row_rescore", "--joint",
          "output/validation/common_row_rerun_2026-09-11/CL/evaluate-2024-rerun", "B"],
         ("joint", ("CL", "output/curve_library_study_2026-09-11/common_row_rescore",
                    ["output/validation/common_row_rerun_2026-09-11/CL/evaluate-2024-rerun", "B"]),
          {})),
    ],
    "scripts/analysis/eu_rerun_compare.py": [
        (["SE", "output/eu_rerun_2026-09-12/analysis", "published=A", "oldfiles_derived=B", "new=C"],
         ("main", ("SE", "output/eu_rerun_2026-09-12/analysis",
                   ["published=A", "oldfiles_derived=B", "new=C"]), {"tag": "rerun"})),
        (["DK", "output/curve_library_study_2026-09-13/analysis", "--tag=T1", "T0=A", "T1=B"],
         ("main", ("DK", "output/curve_library_study_2026-09-13/analysis", ["T0=A", "T1=B"]),
          {"tag": "T1"})),
    ],
    "scripts/analysis/extent_audit.py": [
        (["output/extent_audit_2026-09-12"], ("main", ("output/extent_audit_2026-09-12", []), {})),
        (["output/extent_audit_2026-09-12", "DK", "FR"],
         ("main", ("output/extent_audit_2026-09-12", ["DK", "FR"]), {})),
    ],
    "scripts/studies/scorecard/missing_value_audit.py": [
        (["DK", "output/curve_library_study_2026-09-11/missing_value_audit"],
         ("main", ("DK", "output/curve_library_study_2026-09-11/missing_value_audit"),
          {"backfill": BACKFILL})),
    ],
    "scripts/studies/scorecard/off_curve_sensitivity.py": [
        (["DK", "output/curve_library_study_2026-09-11/off_curve_sensitivity"],
         ("main", ("DK", "output/curve_library_study_2026-09-11/off_curve_sensitivity"),
          {"backfill": BACKFILL})),
    ],
    "scripts/studies/scorecard/unit_concentration.py": [
        (["DK", "output/curve_library_study_2026-09-11/unit_concentration"],
         ("main", ("DK", "output/curve_library_study_2026-09-11/unit_concentration"),
          {"backfill": BACKFILL})),
    ],
    "scripts/studies/scorecard/training_objective_check.py": [
        (["ES", "fixed", "4", "output/curve_library_study_2026-09-11/training_objective_check"],
         ("main", ("ES", "fixed", "4", "output/curve_library_study_2026-09-11/training_objective_check"),
          {"refresh": Path("output/validation/refresh_2026-08-24")})),
    ],
    "scripts/studies/method-cluster-selection/cluster_selection_study.py": [
        (["output/cluster_selection_2026-09-15"], ("main", ("output/cluster_selection_2026-09-15",), {})),
        (["output/cluster_selection_2026-09-15", "DK onshore", "UK offshore"],
         ("main", ("output/cluster_selection_2026-09-15", "DK onshore", "UK offshore"), {})),
    ],
    "scripts/studies/method-cluster-selection/cluster_selection_gaps.py": [
        (["output/cluster_selection_2026-09-15", "UK offshore", "50"],
         ("main", ("output/cluster_selection_2026-09-15", "UK offshore", "50"), {})),
    ],
    "scripts/studies/method-cluster-selection/cluster_sweep_cost.py": [
        # The mode is now passed explicitly; "all" is main's own default.
        (["be", "output/cluster_sweep_cost_2026-09-15"],
         ("main", ("be", "output/cluster_sweep_cost_2026-09-15", "all"), {"pool": POOL})),
        (["dk", "output/cluster_sweep_cost_2026-09-15", "onshore"],
         ("main", ("dk", "output/cluster_sweep_cost_2026-09-15", "onshore"), {"pool": POOL})),
    ],
    "scripts/studies/method-correction-identifiability/correction_identifiability.py": [
        (["output/identifiability_2026-09-16"],
         ("main", ("output/identifiability_2026-09-16",), {"pool_path": POOL})),
    ],
    "scripts/studies/method-correction-identifiability/loco_reference_wind.py": [
        (["output/loco_reference_2026-09-16"],
         ("main", ("output/loco_reference_2026-09-16",), {"pool_path": ROOT / POOL})),
    ],
    "scripts/studies/method-correction-identifiability/pivot_probe.py": [
        (["output/pivot_probe_2026-09-16"],
         ("main", ("output/pivot_probe_2026-09-16",),
          {"selection": ROOT / "output/cluster_selection_2026-09-15",
           "era5_dir": ROOT / "input/era5/EU_2026-09"})),
        (["output/pivot_probe_2026-09-16", "DK", "UK"],
         ("main", ("output/pivot_probe_2026-09-16", "DK", "UK"),
          {"selection": ROOT / "output/cluster_selection_2026-09-15",
           "era5_dir": ROOT / "input/era5/EU_2026-09"})),
    ],
    "scripts/studies/method-country-level/chapter_capacity_weights.py": [
        (["output/chapter_capacity_weights_2026-09-16"],
         ("main", ("output/chapter_capacity_weights_2026-09-16",),
          {"runs": ROOT / "output/runs/turbine_grid",
           "gwpt": ROOT / "input/reference/gwpt/Global-Wind-Power-Tracker-February-2026.xlsx"})),
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
        pytest.skip(f"{script} needs {exc.name}")
    return module


CASES = [(script, argv, expected) for script, cases in RECORDED.items()
         for argv, expected in cases]


@pytest.mark.parametrize("script,argv,expected", CASES,
                         ids=[f"{Path(s).stem}-{i}" for i, (s, _, _) in enumerate(CASES)])
def test_recorded_command_line_makes_the_recorded_call(script, argv, expected, monkeypatch):
    module = load(script)
    calls = []
    for name in ("main", "joint"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name,
                                lambda *a, _n=name, **k: calls.append((_n, a, k)))
    module.cli(argv)
    assert calls == [expected]


def test_every_driver_without_a_parser_is_listed():
    """A driver that reads sys.argv by position has no cli; the list says which remain."""
    missing = []
    for path in sorted((ROOT / "scripts").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text()
        if rel.startswith("scripts/pinn/") or "__main__" not in text:
            continue
        if "sys.argv[" in text and rel not in RECORDED:
            missing.append(rel)
    assert missing == sorted(NOT_YET_CONVERTED), missing


# Still read sys.argv by position; converted one study directory per commit.
NOT_YET_CONVERTED: list[str] = [
    "scripts/studies/manuscript-chapters-45/refit_control_points.py",
    "scripts/studies/method-curve-library/curve_library_study.py",
    "scripts/studies/method-curve-library/curve_library_tables.py",
    "scripts/studies/method-distance-mask/unmasked_surface_bands.py",
    "scripts/studies/method-domain-split/domain_split_study.py",
    "scripts/studies/method-eu-rerun/era5_overlap_check.py",
    "scripts/studies/method-loco-interpolation/loco_interpolation.py",
    "scripts/studies/method-national-single-cluster/national_single_cluster_study.py",
    "scripts/studies/method-offshore-pool/offshore_pool_study.py",
    "scripts/studies/method-roughness-treatment/roughness_treatment_study.py",
    "scripts/studies/method-why-corrections-do-not-transfer/pool_as_training_set.py",
    "scripts/studies/method-why-corrections-do-not-transfer/regime_coverage.py",
]
