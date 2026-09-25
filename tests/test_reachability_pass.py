"""The reachability pass's range brackets every capacity factor the joint fit can produce.

``scripts/studies/method-joint-fit-reachability/reachability_pass.py``
(issue #68) classifies a national period as unreachable when its observation
lies outside the range. The range must therefore contain whatever the fit can
reach, and must not be understated by its grid: an understated maximum would
call a reachable period unreachable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vwf import correction
from vwf.wind import interpolate_wind, train_simulate_wind_from_ws

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/studies/method-joint-fit-reachability/reachability_pass.py"
    spec = importlib.util.spec_from_file_location("reachability_pass", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fleet():
    return pd.DataFrame(
        {
            "ID": ["a", "b", "c", "d"],
            "lat": [55.2, 55.4, 55.6, 55.8],
            "lon": [8.2, 8.4, 8.6, 8.8],
            "height": [100.0] * 4,
            "capacity": [1000.0, 3000.0, 2000.0, 2000.0],
            "model": ["GE.1.5sle"] * 4,
            "cluster": [0, 0, 1, 1],
        }
    )


def _range(module, fleet, reanalysis, power_curve):
    return module.reachable_range(2020, "1/1", {0: 1.0, 1: 1.0}, fleet, reanalysis, power_curve)


def test_the_range_brackets_the_simulation_at_many_offsets(fleet, reanalysis, power_curve):
    module = _module()
    rng = _range(module, fleet, reanalysis, power_curve)
    period = reanalysis.sel(time=reanalysis.time.dt.year == 2020)
    ws = {c: interpolate_wind(period, fleet[fleet["cluster"] == c]) for c in (0, 1)}
    weights = fleet.groupby("cluster")["capacity"].sum() / fleet["capacity"].sum()
    rs = np.random.default_rng(0)
    for offsets in rs.uniform(-9.9, 9.9, size=(40, 2)):
        national = sum(
            weights[c] * float(train_simulate_wind_from_ws(ws[c], power_curve, 1.0, offsets[c]))
            for c in (0, 1)
        )
        # Far enough below zero every speed leaves the curve and the simulation
        # returns no value; that is no capacity factor, reachable or not.
        if np.isnan(national):
            continue
        assert rng["lo"] - 1e-9 <= national <= rng["hi"] + 1e-9


def test_the_refined_maximum_is_not_below_the_grid_maximum(fleet, reanalysis, power_curve):
    """The refinement can only raise the maximum, so the grid never understates it."""
    module = _module()
    period = reanalysis.sel(time=reanalysis.time.dt.year == 2020)
    ws = interpolate_wind(period, fleet[fleet["cluster"] == 0])
    _, hi = module.cluster_extremes(ws, power_curve, 1.0)
    grid = np.arange(-9.75, 10.0, 0.25)
    grid_max = max(float(train_simulate_wind_from_ws(ws, power_curve, 1.0, o)) for o in grid)
    assert hi >= grid_max


def test_an_observation_above_the_range_is_accepted_with_the_whole_gap_as_error(
    fleet, reanalysis, power_curve
):
    """Known positive: the joint fit accepts a target it cannot reach.

    The curve is derated to half its output, so the range tops out at 0.5 on
    the rated plateau, inside the offset bounds. An observation 0.05 above it
    was expected to be refused; it is not. The fit converges on the plateau
    with no offset on a bound, reports success, and leaves the whole gap as
    error, which no existing refusal sees. This is the case the pass labels
    unreachable. If the fit is changed to refuse it, this test says so.
    """
    module = _module()
    power_curve = power_curve.copy()
    model = [c for c in power_curve.columns if c != "data$speed"][0]
    power_curve[model] = power_curve[model] * 0.5
    rng = _range(module, fleet, reanalysis, power_curve)
    obs = rng["hi"] + 0.05
    assert obs < 1.0
    offsets = correction.find_offsets_country_level(
        year=2020,
        time_slice="1/1",
        obs_country_cf=obs,
        scalars_by_cluster={0: 1.0, 1: 1.0},
        turb_info=fleet,
        reanalysis=reanalysis,
        powerCurveFile=power_curve,
    )
    assert all(np.isfinite(v) and abs(v) < 9.9 for v in offsets.values())
    period = reanalysis.sel(time=reanalysis.time.dt.year == 2020)
    weights = fleet.groupby("cluster")["capacity"].sum() / fleet["capacity"].sum()
    national = sum(
        weights[c]
        * float(
            train_simulate_wind_from_ws(
                interpolate_wind(period, fleet[fleet["cluster"] == c]), power_curve, 1.0, offsets[c]
            )
        )
        for c in (0, 1)
    )
    assert national == pytest.approx(rng["hi"], abs=1e-6)
    assert obs - national == pytest.approx(0.05, abs=1e-6)


def test_an_observation_inside_the_range_is_matched(fleet, reanalysis, power_curve):
    """Known negative: inside the range, the fit closes the error."""
    module = _module()
    rng = _range(module, fleet, reanalysis, power_curve)
    obs = rng["lo"] + 0.5 * (rng["hi"] - rng["lo"])
    offsets = correction.find_offsets_country_level(
        year=2020,
        time_slice="1/1",
        obs_country_cf=obs,
        scalars_by_cluster={0: 1.0, 1: 1.0},
        turb_info=fleet,
        reanalysis=reanalysis,
        powerCurveFile=power_curve,
    )
    assert all(np.isfinite(v) for v in offsets.values())


def _offcurve():
    path = ROOT / "scripts/studies/method-joint-fit-reachability/reachability_offcurve.py"
    spec = importlib.util.spec_from_file_location("reachability_offcurve", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_counting_off_curve_steps_as_zero_can_only_lower_the_floor(fleet, reanalysis, power_curve):
    """At -10 m/s every synthetic speed is below zero: dropped, there is no value;
    counted as zero, the capacity factor is zero, and the floor falls to it."""
    offcurve = _offcurve()
    period = reanalysis.sel(time=reanalysis.time.dt.year == 2020)
    ws = interpolate_wind(period, fleet[fleet["cluster"] == 0])
    assert np.isnan(float(train_simulate_wind_from_ws(ws, power_curve, 1.0, -9.99)))
    assert offcurve.cf_zero(ws, power_curve, 1.0, -9.99) == 0.0
    ext = offcurve.cluster_extremes_zero(ws, power_curve, 1.0)
    lo_drop, _ = _module().cluster_extremes(ws, power_curve, 1.0)
    assert ext["lo"] <= lo_drop
    assert ext["lo"] == 0.0 and ext["lo_off_curve"] > 0


def test_on_the_curve_the_two_simulations_agree(fleet, reanalysis, power_curve):
    offcurve = _offcurve()
    period = reanalysis.sel(time=reanalysis.time.dt.year == 2020)
    ws = interpolate_wind(period, fleet[fleet["cluster"] == 0])
    for o in (-2.0, 0.0, 3.0):
        assert offcurve.cf_zero(ws, power_curve, 1.0, o) == pytest.approx(
            float(train_simulate_wind_from_ws(ws, power_curve, 1.0, o)), abs=1e-12
        )
