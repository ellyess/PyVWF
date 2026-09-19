"""Scoring a gridded correction at observations (src/vwf/extensions/grid/evaluate.py).

Ported from the `development` branch, where it carried no tests. The two that
matter most pin the silent failures: neutral fills are counted and returned
rather than substituted quietly, and a broken metric raises rather than
producing a row of missing values in a results table.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from vwf.extensions.grid import evaluate


def surface(scalar=0.8, offset=0.5, axes=("x", "y"), masked=False):
    lon = np.array([0.0, 1.0, 2.0, 3.0])
    lat = np.array([50.0, 51.0, 52.0])
    s = np.full((len(lat), len(lon)), float(scalar))
    o = np.full((len(lat), len(lon)), float(offset))
    if masked:
        s[:, -1] = 1.0
        o[:, -1] = 0.0
    lon_name, lat_name = axes
    return xr.Dataset(
        {"scalar": ((lat_name, lon_name), s), "offset": ((lat_name, lon_name), o)},
        coords={lon_name: lon, lat_name: lat},
    )


def units(ids=("a", "b"), lons=(0.5, 1.5), lats=(50.5, 51.5), capacity=(100.0, 200.0)):
    return pd.DataFrame(
        {"ID": list(ids), "lon": list(lons), "lat": list(lats), "capacity": list(capacity)}
    )


def test_a_unit_takes_the_correction_at_its_own_position():
    got, summary = evaluate.corrections_at(surface(), units())
    assert list(got["ID"]) == ["a", "b"]
    assert got["scalar"].tolist() == pytest.approx([0.8, 0.8])
    assert got["offset"].tolist() == pytest.approx([0.5, 0.5])
    assert summary["n_neutral"] == 0 and summary["neutral_share"] == 0.0


def test_a_unit_off_the_grid_is_neutral_and_counted():
    """The original substituted neutral values silently, so a run in which most
    units got no correction scored as an ordinary result."""
    got, summary = evaluate.corrections_at(
        surface(), units(ids=("a", "off"), lons=(0.5, 40.0), lats=(50.5, 50.5))
    )
    assert got.loc[1, "scalar"] == 1.0 and got.loc[1, "offset"] == 0.0
    assert bool(got.loc[1, "neutral"]) and not bool(got.loc[0, "neutral"])
    assert summary["n_neutral"] == 1 and summary["n_off_grid"] == 1
    assert summary["neutral_share"] == pytest.approx(0.5)


def test_a_unit_in_a_masked_region_counts_as_neutral_too():
    """Masked cells hold scalar 1 and offset 0, which is the surface declining
    to answer rather than a correction that happens to be the identity."""
    got, summary = evaluate.corrections_at(
        surface(masked=True), units(ids=("in", "masked"), lons=(0.5, 3.0), lats=(50.5, 50.0))
    )
    assert summary["n_neutral"] == 1 and summary["n_off_grid"] == 0
    assert bool(got.loc[1, "neutral"])


@pytest.mark.parametrize("axes", [("x", "y"), ("lon", "lat")])
def test_both_axis_conventions_are_read(axes):
    got, _ = evaluate.corrections_at(surface(axes=axes), units())
    assert got["scalar"].tolist() == pytest.approx([0.8, 0.8])


def test_a_surface_with_neither_convention_is_refused():
    bad = xr.Dataset(
        {"scalar": (("a", "b"), np.ones((2, 2))), "offset": (("a", "b"), np.zeros((2, 2)))},
        coords={"a": [0.0, 1.0], "b": [1.0, 2.0]},
    )
    with pytest.raises(KeyError, match="expected x and y"):
        evaluate.corrections_at(bad, units())


def test_units_without_the_columns_it_needs_are_refused():
    with pytest.raises(ValueError, match="missing 'lat'"):
        evaluate.corrections_at(surface(), units().drop(columns=["lat"]))


def speeds(ids=("a", "b"), model="M", values=((8.0, 9.0), (10.0, 11.0))):
    times = pd.date_range("2023-01-01", periods=len(values), freq="D")
    da = xr.DataArray(
        np.array(values, dtype=float),
        dims=("time", "turbine"),
        coords={"time": times, "turbine": list(ids)},
    )
    return da.assign_coords(model=("turbine", [model] * len(ids)))


def curves():
    speed = np.arange(0.0, 30.1, 1.0)
    return pd.DataFrame({"data$speed": speed, "M": np.clip(speed / 20.0, 0, 1)})


def test_the_correction_is_applied_to_speed_before_the_curve():
    """A scalar of 0.5 on a speed of 10 gives 5, which is a capacity factor of
    0.25 on this curve, not half of the capacity factor at 10, which is 0.25
    either way here only because the curve is linear below rated. The offset is
    what separates them."""
    corrections = pd.DataFrame(
        {"ID": ["a", "b"], "scalar": [0.5, 1.0], "offset": [0.0, 0.0], "neutral": [False, False]}
    )
    got, summary = evaluate.corrected_capacity_factors(speeds(), corrections, curves())
    assert got["a"].iloc[1] == pytest.approx(10.0 * 0.5 / 20.0)
    assert got["b"].iloc[1] == pytest.approx(11.0 / 20.0)
    assert summary["n_off_curve"] == 0


def test_a_unit_with_no_correction_is_refused_by_name():
    corrections = pd.DataFrame({"ID": ["a"], "scalar": [1.0], "offset": [0.0], "neutral": [False]})
    with pytest.raises(ValueError, match="have no correction"):
        evaluate.corrected_capacity_factors(speeds(), corrections, curves())


def test_a_speed_past_the_curve_is_missing_rather_than_clipped_and_is_counted():
    """The chapter's text says corrected speeds are clipped to physically
    admissible bounds; the code clips the capacity factor instead and leaves
    the speed alone, so an over-corrected unit falls off the end of the curve
    table and comes back missing. A missing value then drops out of the metrics
    without trace, which is why the count is returned."""
    corrections = pd.DataFrame(
        {"ID": ["a", "b"], "scalar": [10.0, 1.0], "offset": [0.0, 0.0], "neutral": [False, False]}
    )
    got, summary = evaluate.corrected_capacity_factors(speeds(), corrections, curves())
    assert got["a"].isna().all()
    assert got["b"].notna().all()
    assert summary["n_off_curve"] == 2
    assert summary["off_curve_share"] == pytest.approx(0.5)


def test_capacity_factors_on_the_curve_stay_inside_zero_and_one():
    corrections = pd.DataFrame(
        {"ID": ["a", "b"], "scalar": [2.5, 0.1], "offset": [0.0, 0.0], "neutral": [False, False]}
    )
    got, _ = evaluate.corrected_capacity_factors(speeds(), corrections, curves())
    values = got[["a", "b"]].to_numpy()
    assert np.nanmin(values) >= 0.0 and np.nanmax(values) <= 1.0


def country_frames():
    times = pd.date_range("2023-01-01", periods=60, freq="D")
    sim = pd.DataFrame({"time": times, "a": 0.30, "b": 0.50})
    obs = pd.DataFrame({"time": times, "obs": 0.40})
    return sim, obs


def test_country_skill_weights_grid_points_by_capacity():
    sim, obs = country_frames()
    got = evaluate.country_skill(sim, obs, units(capacity=(100.0, 100.0)))
    assert got["bias"] == pytest.approx(0.0, abs=1e-12)
    heavier = evaluate.country_skill(sim, obs, units(capacity=(300.0, 100.0)))
    assert heavier["bias"] == pytest.approx(0.35 - 0.40)
    assert heavier["n_months"] == 3


def test_a_grid_point_with_no_value_leaves_the_weight_as_well_as_the_value():
    sim, obs = country_frames()
    sim.loc[:, "b"] = np.nan
    got = evaluate.country_skill(sim, obs, units(capacity=(100.0, 900.0)))
    assert got["bias"] == pytest.approx(0.30 - 0.40)


def test_country_skill_refuses_rather_than_returning_missing_values():
    sim, obs = country_frames()
    with pytest.raises(ValueError, match="key on different identifiers"):
        evaluate.country_skill(sim, obs, units(ids=("x", "y")))
    empty = pd.DataFrame({"time": pd.to_datetime(["2019-01-01"]), "obs": [0.4]})
    with pytest.raises(ValueError, match="no month has both"):
        evaluate.country_skill(sim, empty, units())


def test_the_skill_dispatcher_refuses_an_unknown_level():
    sim, obs = country_frames()
    with pytest.raises(ValueError, match="unknown obs_level"):
        evaluate.skill(sim, obs, units(), "farm")
