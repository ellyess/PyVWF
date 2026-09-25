"""Every recorded command line of a study driver or analysis script still parses.

The drivers under ``scripts/`` took their arguments from ``sys.argv`` by
position until 2026-09-18, and their findings documents and docstrings record
the command lines that produced each result. Each now parses with
``argparse`` in a ``cli(argv)`` function, and a path it used to hardcode is a
flag whose default is that path. This file keeps the recorded command lines
working: for each, it runs ``cli`` with the entry functions replaced, and
checks the call they receive, which is the call the ``sys.argv`` version made.

A driver that needs an optional dependency to import is skipped where that
dependency is missing. A driver that fails to import a ``vwf`` name fails:
that is the script breaking against the package, not a missing extra.
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
RUNS = Path("output/runs/turbine_grid")
SHAPES = Path("input/reference/shapes")

# script: [(argv, (entry function, positional args, keyword args)), ...]
RECORDED: dict[str, list[tuple[list[str], tuple[str, tuple, dict]]]] = {
    # Development guards (docs/design/agent-guards.md). The command lines are
    # the ones AGENTS.md and src/vwf/AGENTS.md give.
    "scripts/dev/run_locked.py": [
        (
            ["--", "bash", "scripts/pinn/run_overnight.sh"],
            ("main", (["bash", "scripts/pinn/run_overnight.sh"],), {"allow_dirty": False}),
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
        (["realdata"], ("main", ("realdata",), {})),
        (["pinn"], ("main", ("pinn",), {})),
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
    "scripts/analysis/eu_rerun_compare.py": [
        (
            [
                "SE",
                "output/eu_rerun_2026-09-12/analysis",
                "published=A",
                "oldfiles_derived=B",
                "new=C",
            ],
            (
                "main",
                (
                    "SE",
                    "output/eu_rerun_2026-09-12/analysis",
                    ["published=A", "oldfiles_derived=B", "new=C"],
                ),
                {"tag": "rerun"},
            ),
        ),
        (
            ["DK", "output/curve_library_study_2026-09-13/analysis", "--tag=T1", "T0=A", "T1=B"],
            (
                "main",
                ("DK", "output/curve_library_study_2026-09-13/analysis", ["T0=A", "T1=B"]),
                {"tag": "T1"},
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
    "scripts/studies/scorecard/missing_value_audit.py": [
        (
            ["DK", "output/curve_library_study_2026-09-11/missing_value_audit"],
            (
                "main",
                ("DK", "output/curve_library_study_2026-09-11/missing_value_audit"),
                {"backfill": BACKFILL},
            ),
        ),
    ],
    "scripts/studies/scorecard/off_curve_sensitivity.py": [
        (
            ["DK", "output/curve_library_study_2026-09-11/off_curve_sensitivity"],
            (
                "main",
                ("DK", "output/curve_library_study_2026-09-11/off_curve_sensitivity"),
                {"backfill": BACKFILL},
            ),
        ),
    ],
    "scripts/studies/scorecard/unit_concentration.py": [
        (
            ["DK", "output/curve_library_study_2026-09-11/unit_concentration"],
            (
                "main",
                ("DK", "output/curve_library_study_2026-09-11/unit_concentration"),
                {"backfill": BACKFILL},
            ),
        ),
    ],
    "scripts/studies/scorecard/training_objective_check.py": [
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
    "scripts/studies/method-terrain-wind-deficit/terrain_deficit.py": [
        # The registered command line takes no arguments: every path is a flag
        # whose default is the registered one, and the years, clusters, seed
        # and gates are constants (method-terrain-wind-deficit-prereg.md).
        (
            [],
            (
                "main",
                (),
                {
                    "config": Path("configs/regions/scorecard/us_k250.toml"),
                    "train_run": Path("output/validation/bracketed_2026-09-19/US/train-bracketed"),
                    "evaluate_run": Path(
                        "output/validation/bracketed_2026-09-19/US/evaluate-2022-bracketed"
                    ),
                    "etopo": Path("input/reference/terrain/etopo_global.nc"),
                    "audit": Path("scripts/analysis/curve_match_audit.py"),
                    "out": Path("output/terrain_wind_deficit_2026-09-20_corrected"),
                },
            ),
        ),
    ],
    "scripts/studies/method-cluster-selection/cluster_selection_study.py": [
        (
            ["output/cluster_selection_2026-09-15"],
            ("main", ("output/cluster_selection_2026-09-15",), {}),
        ),
        (
            ["output/cluster_selection_2026-09-15", "DK onshore", "UK offshore"],
            ("main", ("output/cluster_selection_2026-09-15", "DK onshore", "UK offshore"), {}),
        ),
    ],
    "scripts/studies/method-cluster-selection/cluster_selection_gaps.py": [
        (
            ["output/cluster_selection_2026-09-15", "UK offshore", "50"],
            ("main", ("output/cluster_selection_2026-09-15", "UK offshore", "50"), {}),
        ),
    ],
    "scripts/studies/method-cluster-selection/cluster_sweep_cost.py": [
        # The mode is now passed explicitly; "all" is main's own default.
        (
            ["be", "output/cluster_sweep_cost_2026-09-15"],
            ("main", ("be", "output/cluster_sweep_cost_2026-09-15", "all"), {"pool": POOL}),
        ),
        (
            ["dk", "output/cluster_sweep_cost_2026-09-15", "onshore"],
            ("main", ("dk", "output/cluster_sweep_cost_2026-09-15", "onshore"), {"pool": POOL}),
        ),
    ],
    "scripts/studies/method-correction-identifiability/correction_identifiability.py": [
        (
            ["output/identifiability_2026-09-16"],
            ("main", ("output/identifiability_2026-09-16",), {"pool_path": POOL}),
        ),
    ],
    "scripts/studies/method-correction-identifiability/loco_reference_wind.py": [
        (
            ["output/loco_reference_2026-09-16"],
            ("main", ("output/loco_reference_2026-09-16",), {"pool_path": ROOT / POOL}),
        ),
    ],
    "scripts/studies/method-correction-identifiability/pivot_probe.py": [
        (
            ["output/pivot_probe_2026-09-16"],
            (
                "main",
                ("output/pivot_probe_2026-09-16",),
                {
                    "selection": ROOT / "output/cluster_selection_2026-09-15",
                    "era5_dir": ROOT / "input/era5/EU_2026-09",
                },
            ),
        ),
        (
            ["output/pivot_probe_2026-09-16", "DK", "UK"],
            (
                "main",
                ("output/pivot_probe_2026-09-16", "DK", "UK"),
                {
                    "selection": ROOT / "output/cluster_selection_2026-09-15",
                    "era5_dir": ROOT / "input/era5/EU_2026-09",
                },
            ),
        ),
    ],
    "scripts/studies/method-country-level/chapter_capacity_weights.py": [
        (
            ["output/chapter_capacity_weights_2026-09-16"],
            (
                "main",
                ("output/chapter_capacity_weights_2026-09-16",),
                {
                    "runs": ROOT / "output/runs/turbine_grid",
                    "gwpt": ROOT
                    / "input/reference/gwpt/Global-Wind-Power-Tracker-February-2026.xlsx",
                },
            ),
        ),
    ],
    "scripts/studies/manuscript-chapters-45/refit_control_points.py": [
        (
            ["output/refit_control_points_2026-09-15"],
            ("main", ("output/refit_control_points_2026-09-15",), {"pool_path": POOL}),
        ),
        (
            ["output/refit_control_points_2026-09-15", "dk", "uk"],
            ("main", ("output/refit_control_points_2026-09-15", "dk", "uk"), {"pool_path": POOL}),
        ),
    ],
    "scripts/studies/method-curve-library/curve_library_study.py": [
        (
            [
                "DK",
                "T1",
                "output/curve_library_study_2026-09-13/T1",
                "output/curve_library_study_2026-09-13/tables/T1_overrides.csv",
            ],
            (
                "main",
                (
                    "DK",
                    "T1",
                    "output/curve_library_study_2026-09-13/T1",
                    "output/curve_library_study_2026-09-13/tables/T1_overrides.csv",
                ),
                {},
            ),
        ),
    ],
    "scripts/studies/method-curve-library/curve_library_tables.py": [
        (
            ["output/curve_library_study_2026-09-13/tables"],
            (
                "main",
                (
                    "output/curve_library_study_2026-09-13/tables",
                    (
                        Path("input/reference/models.csv"),
                        Path("input/combined/reference/models_with_library.csv"),
                        Path("output/eu_rerun_2026-09-12/new"),
                        Path("output/validation/refresh_2026-08-24"),
                    ),
                ),
                {},
            ),
        ),
    ],
    "scripts/studies/method-distance-mask/unmasked_surface_bands.py": [
        (
            ["output/unmasked_bands_2026-09-15"],
            ("main", ("output/unmasked_bands_2026-09-15",), {"pool_path": POOL, "shapes": SHAPES}),
        ),
    ],
    "scripts/studies/method-domain-split/domain_split_study.py": [
        (
            ["output/domain_split_2026-09-15"],
            (
                "main",
                ("output/domain_split_2026-09-15",),
                {
                    "pool_path": POOL,
                    "runs": RUNS,
                    "shapes": SHAPES,
                    "era5": Path("input/era5/EU_2026-09"),
                    "era5_chapter": Path("input/era5/EU"),
                },
            ),
        ),
    ],
    "scripts/studies/method-eu-rerun/era5_overlap_check.py": [
        (
            ["output/era5_overlap_2026-09-12"],
            (
                "main",
                ("output/era5_overlap_2026-09-12",),
                {"old_dir": Path("input/era5/EU"), "new_dir": Path("input/era5/EU_2026-09")},
            ),
        ),
    ],
    "scripts/studies/method-hourly-resolution/hourly_resolution_test.py": [
        (
            [],
            (
                "main",
                (),
                {
                    "out": Path("output/hourly_test"),
                    "train_run": Path("output/validation/cl_matched_2026-07-24/CL/train-matched"),
                    "raw_cen": Path("input/raw/cen"),
                },
            ),
        ),
    ],
    "scripts/studies/method-joint-fit-reachability/reachability_pass.py": [
        (
            ["output/reachability_2026-09-25", "fr"],
            ("main", ("output/reachability_2026-09-25", "fr"), {}),
        ),
    ],
    "scripts/studies/method-joint-fit-reachability/reachability_offcurve.py": [
        (
            ["output/reachability_2026-09-25", "it"],
            ("main", ("output/reachability_2026-09-25", "it"), {}),
        ),
    ],
    "scripts/studies/method-joint-fit-reachability/reachability_tables.py": [
        (["output/reachability_2026-09-25"], ("main", ("output/reachability_2026-09-25",), {})),
    ],
    "scripts/studies/method-joint-fit-reachability/offcurve_tables.py": [
        (["output/reachability_2026-09-25"], ("main", ("output/reachability_2026-09-25",), {})),
    ],
    "scripts/studies/method-loco-interpolation/loco_interpolation.py": [
        (["output/loco_2026-09-13"], ("main", ("output/loco_2026-09-13",), {"pool_path": POOL})),
    ],
    "scripts/studies/method-national-single-cluster/national_single_cluster_study.py": [
        (
            ["output/national_single_cluster_2026-09-16"],
            ("main", ("output/national_single_cluster_2026-09-16",), {}),
        ),
        (
            ["output/national_single_cluster_2026-09-16", "be", "fr"],
            ("main", ("output/national_single_cluster_2026-09-16", "be", "fr"), {}),
        ),
    ],
    "scripts/studies/method-offshore-pool/offshore_pool_study.py": [
        (
            ["output/offshore_pool_2026-09-13"],
            (
                "main",
                ("output/offshore_pool_2026-09-13",),
                {"pool_path": POOL, "runs": RUNS, "shapes": SHAPES},
            ),
        ),
    ],
    "scripts/studies/method-roughness-treatment/roughness_treatment_study.py": [
        (
            ["DK", "R0_DIR", "R1_DIR", "output/roughness_treatment_2026-09-12/analysis"],
            (
                "main",
                ("DK", "R0_DIR", "R1_DIR", "output/roughness_treatment_2026-09-12/analysis"),
                {},
            ),
        ),
    ],
    "scripts/studies/method-scalar-bounds/min_cluster_size_tradeoff.py": [
        ([], ("main", (), {"out": Path("output/min_cluster_size")})),
    ],
    "scripts/studies/method-why-corrections-do-not-transfer/pool_as_training_set.py": [
        (
            ["output/pool_training_2026-09-16"],
            (
                "main",
                ("output/pool_training_2026-09-16",),
                {
                    "pool_path": ROOT / POOL,
                    "runs": ROOT / RUNS,
                    "selection": ROOT / "output/cluster_selection_2026-09-15",
                },
            ),
        ),
    ],
    "scripts/studies/method-why-corrections-do-not-transfer/regime_coverage.py": [
        (
            ["output/regime_coverage_2026-09-16"],
            (
                "main",
                ("output/regime_coverage_2026-09-16",),
                {
                    "pool_path": ROOT / POOL,
                    "loco_path": ROOT / "output/loco_reference_2026-09-16/loco_reference_wind.csv",
                    "refresh": ROOT / "output/validation/refresh_2026-08-24",
                },
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
        # A missing optional dependency is a skip. A missing vwf name is the
        # script breaking against the package, which must fail, not skip.
        if exc.name and (exc.name == "vwf" or exc.name.startswith("vwf.")):
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
        if rel.startswith("scripts/pinn/") or "__main__" not in text:
            continue
        if "sys.argv[" in text:
            positional.append(rel)
        if "\ndef cli(" in text and rel not in RECORDED:
            unpinned.append(rel)
    assert positional == []
    assert unpinned == []


def test_a_broken_vwf_import_fails_rather_than_skips(tmp_path):
    script = tmp_path / "broken_driver.py"
    script.write_text("from vwf.harness.driver import no_such_name\n")
    try:
        load(str(script))
    except ImportError:
        return
    except pytest.skip.Exception:
        pytest.fail("a broken vwf import was skipped, which CI reports as green")
    pytest.fail("the broken import raised nothing")


def test_a_missing_optional_dependency_skips(tmp_path):
    script = tmp_path / "optional_driver.py"
    script.write_text("import no_such_optional_dependency_xyz\n")
    with pytest.raises(pytest.skip.Exception):
        load(str(script))
