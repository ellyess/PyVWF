"""Invariants of the bundled open curve library.

The repo and the wheel ship the same three files (power curves, turbine
models, provenance); these tests pin the contract the rest of the pipeline
relies on: every model resolves to a curve column, curves are capacity
factors on the expected speed grid, and every curve column has a recorded
source and license.
"""
import numpy as np
import pandas as pd
import pytest

RESOURCE_FILES = ["power_curves.csv", "models.csv", "power_curves_provenance.csv"]


@pytest.fixture(scope="module")
def bundled():
    from importlib import resources

    root = resources.files("vwf.resources")
    return {name: pd.read_csv(str(root / name)) for name in RESOURCE_FILES}


def test_repo_input_matches_the_bundled_resources():
    """input/ and vwf/resources/ must not drift apart: reference_file prefers
    the former in a checkout and falls back to the latter when installed, and
    both paths have to produce identical numbers."""
    from importlib import resources
    from pathlib import Path

    if not Path("input/power_curves.csv").is_file():
        pytest.skip("not running from a repository checkout")

    root = resources.files("vwf.resources")
    for name in RESOURCE_FILES:
        repo = Path("input") / name
        assert repo.read_bytes() == (root / name).read_bytes(), (
            f"{name} differs between input/ and vwf/resources/"
        )


def test_every_model_has_a_curve_column(bundled):
    curve_cols = set(bundled["power_curves.csv"].columns) - {"data$speed"}
    models = set(bundled["models.csv"]["model"])
    missing = models - curve_cols
    assert not missing, f"models without a curve: {sorted(missing)}"


def test_curves_are_capacity_factors_on_the_expected_grid(bundled):
    curves = bundled["power_curves.csv"]
    speed = curves["data$speed"]
    assert float(speed.iloc[0]) == 0.0
    assert float(speed.iloc[-1]) == 40.0
    # 0 to 40 m/s at 0.01: same grid as the real VWF library files.
    assert len(curves) == 4001
    values = curves.drop(columns=["data$speed"])
    assert float(values.min().min()) >= 0.0
    assert float(values.max().max()) <= 1.0 + 1e-9
    # Zero output in calm air, for every machine.
    assert (values.iloc[0] == 0.0).all()


def test_provenance_covers_every_curve_column(bundled):
    curve_cols = set(bundled["power_curves.csv"].columns) - {"data$speed"}
    prov = bundled["power_curves_provenance.csv"]
    assert set(prov["column"]) == curve_cols
    assert (prov["license"] == "BSD-3-Clause").all()
    assert prov["doi"].notna().all()


def test_model_p_density_is_consistent_with_capacity_and_diameter(bundled):
    """add_models matches on p_density; a stale column would silently skew
    every assignment. The capacity column is rounded to whole kW (p_density
    was computed from the unrounded rating, e.g. the Kestrel 2.5 kW is listed
    as 2), so the check is that p_density implies a capacity within rounding
    distance of the stated one."""
    m = bundled["models.csv"]
    implied_kw = m["p_density"] * np.pi * (m["diameter"] / 2.0) ** 2 / 1000.0
    assert (implied_kw - m["capacity"]).abs().le(0.55).all()


def test_composites_are_curve_only(bundled):
    """The normalized composites are dimensionless profiles: they belong in
    the curve file (as generic fallbacks a user can point default_model at)
    but must NOT be in models.csv, where add_models would treat their
    made-up capacity/diameter as a real machine."""
    prov = bundled["power_curves_provenance.csv"]
    composites = set(prov.loc[prov["is_normalized_composite"], "column"])
    assert composites, "expected the flagged composite profiles"
    in_models = composites & set(bundled["models.csv"]["model"])
    assert not in_models, f"composites leaked into models.csv: {sorted(in_models)}"
