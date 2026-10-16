"""The country weighting study's Stage 0 driver, on synthetic grids.

``scripts/studies/method-country-weighting/stage0.py`` compares two weightings
of one simulation, so its correctness rests on three checks: that the two grids
differ only in capacity, that A's grid is the registered one, and that the
national mean and the kill-rule quantities are computed as registered. Each
check is shown to refuse a known bad input and to pass a good one.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "studies"
    / "method-country-weighting"
    / "stage0.py"
)
_SPEC = importlib.util.spec_from_file_location("stage0", _PATH)
stage0 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stage0)


def grid(capacity=(10.0, 30.0, 0.0)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lat": [50.0, 51.0, 52.0],
            "lon": [4.0, 4.5, 3.0],
            "ID": ["g0", "g1", "offshore_52.0000_3.0000"],
            "height": [100.0] * 3,
            "model": ["Vestas.V90.3000"] * 3,
            "capacity": list(capacity),
            "cluster": [0, 1, 0],
        }
    )


def test_grids_that_differ_only_in_capacity_pass():
    stage0.check_same_points(grid(), grid(capacity=(5.0, 30.0, 20.0)), "XX 2015")


@pytest.mark.parametrize("column, value", [("lat", 50.25), ("cluster", 1), ("height", 80.0)])
def test_grids_that_differ_in_a_site_are_refused(column, value):
    moved = grid()
    moved.loc[0, column] = value
    with pytest.raises(ValueError, match="other than capacity"):
        stage0.check_same_points(grid(), moved, "XX 2015")


def test_the_national_mean_leaves_a_missing_point_out_of_both_sums():
    sim = pd.DataFrame(
        {"g0": [0.2, np.nan], "g1": [0.4, 0.4], "offshore_52.0000_3.0000": [0.9, 0.9]},
        index=pd.MultiIndex.from_tuples([(2015, 1), (2015, 2)], names=["year", "month"]),
    )
    capacity = grid().set_index("ID")["capacity"]
    out = stage0.national(sim, capacity)
    # (10 * 0.2 + 30 * 0.4) / 40, then only g1 has a value and a weight.
    assert out.tolist() == pytest.approx([0.35, 0.4])


def test_the_kill_rule_quantities_are_as_registered():
    monthly = pd.DataFrame({"year": [2015] * 2, "cf_a": [0.30, 0.30], "cf_b": [0.33, 0.27]})
    s = stage0.summarise(monthly)
    assert s["rms_difference"] == pytest.approx(0.03)
    assert s["scalar_shift"] == pytest.approx(0.0)
    assert s["ratio_midyear_to_a"] == pytest.approx(1.0)
    # A large RMS with no mean shift does not clear: both thresholds are needed.
    assert s["clears"] is False
    shifted = monthly.assign(cf_b=[0.33, 0.33])
    assert stage0.summarise(shifted)["clears"] is True


def test_zero_capacity_reports_a_points_and_a_clusters_absence_in_any_grid():
    mids = {2015: grid(capacity=(0.0, 30.0, np.nan))}
    out = stage0.zero_capacity(grid(), mids)
    points = out[out["level"] == "point"]
    assert set(map(tuple, points[["grid", "ID"]].to_numpy())) == {
        ("A", "offshore_52.0000_3.0000"),
        ("2015", "g0"),
        ("2015", "offshore_52.0000_3.0000"),
    }
    clusters = out[out["level"] == "cluster"]
    assert clusters[["grid", "cluster"]].to_numpy().tolist() == [["2015", 0]]


def test_a_grid_a_that_is_not_the_registered_one_is_refused(tmp_path, monkeypatch):
    grid_dir = tmp_path / "grid_points" / "xx"
    grid_dir.mkdir(parents=True)
    path = grid_dir / "xx_grid_points_2021.csv"
    grid().to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    listed = tmp_path / "after.sha256"
    monkeypatch.setattr(stage0, "grid_dir", lambda code: tmp_path / "grid_points" / code.lower())
    monkeypatch.setattr(stage0, "A_CHECKSUMS", listed)

    listed.write_text(f"{digest}  xx_grid_points_2021.csv\n")
    assert stage0.checked_grid_a("XX", 2021) == path

    listed.write_text(f"{'0' * 64}  xx_grid_points_2021.csv\n")
    with pytest.raises(ValueError, match="not the grid registered"):
        stage0.checked_grid_a("XX", 2021)


# --- the denominator table's flat rule -------------------------------------------

_DSPEC = importlib.util.spec_from_file_location("denominators", _PATH.with_name("denominators.py"))
denominators = importlib.util.module_from_spec(_DSPEC)
_DSPEC.loader.exec_module(denominators)


def test_three_consecutive_years_within_a_megawatt_are_flat():
    assert denominators.is_flat(pd.Series([7000.0, 8354.0, 8354.0, 8354.5, 9000.0]))


def test_two_equal_years_or_a_steady_rise_are_not_flat():
    assert not denominators.is_flat(pd.Series([8354.0, 8354.0, 9000.0, 9500.0]))
    assert not denominators.is_flat(pd.Series([1000.0, 1002.0, 1004.0, 1006.0]))


def test_january_figure_is_the_first_january_reading_with_its_spread():
    idx = pd.to_datetime(
        ["2015-01-01 00:00", "2015-01-20 00:00", "2015-02-01 00:00", "2016-01-01 00:00"],
        utc=True,
    )
    obs = pd.DataFrame({"capacity_mw": [100.0, 104.0, 999.0, 120.0]}, index=idx)
    out = denominators.january_figures(obs, [2015, 2016]).set_index("year")
    assert out.loc[2015, "jan1_mw"] == 100.0
    assert out.loc[2015, "january_spread_mw"] == 4.0
    assert out.loc[2016, "jan1_mw"] == 120.0
