"""Tests for the linear (scalar + offset) bias-correction scheme."""
import numpy as np
import pandas as pd
import pytest

from vwf.correction import (
    calculate_scalar,
    find_offset,
    _find_offset_iterative,
    _find_offset_scipy,
)
from vwf.wind import interpolate_wind, prepare_offset_arrays, fast_simulate_cf


def _bias_df(obs, sim):
    return pd.DataFrame({
        "fixed": ["1/1"] * len(obs),
        "cluster": [0] * len(obs),
        "year": [2020] * len(obs),
        "obs": obs,
        "sim": sim,
        "capacity": [1.0] * len(obs),
    })


def test_scalar_is_obs_over_sim_ratio():
    df = _bias_df([0.4, 0.4], [0.5, 0.5])
    out = calculate_scalar(df, "fixed")
    assert out["scalar"].iloc[0] == pytest.approx(0.8)


def test_scalar_capacity_weighted():
    df = pd.DataFrame({
        "fixed": ["1/1", "1/1"],
        "cluster": [0, 0],
        "year": [2020, 2020],
        "obs": [0.6, 0.2],
        "sim": [0.5, 0.5],
        "capacity": [3.0, 1.0],  # heavier weight on the 0.6 obs
    })
    out = calculate_scalar(df, "fixed")
    # weighted obs = (0.6*3 + 0.2*1)/4 = 0.5 ; weighted sim = 0.5 ; scalar = 1.0
    assert out["scalar"].iloc[0] == pytest.approx(1.0)


def test_scalar_all_nan_group_returns_nan():
    df = _bias_df([np.nan, np.nan], [0.5, 0.5])
    out = calculate_scalar(df, "fixed")
    assert np.isnan(out["scalar"].iloc[0])


@pytest.fixture
def offset_setup(make_reanalysis, power_curve):
    ds = make_reanalysis(n_hours=72, mean_speed=8.0, seed=5)
    turb = pd.DataFrame({
        "ID": ["a"], "lat": [55.5], "lon": [8.5], "height": [100.0],
        "model": ["GE.1.5sle"], "capacity": [1.0], "cluster": [0],
    })
    ws = interpolate_wind(ds, turb)
    arrays = prepare_offset_arrays(ws, power_curve)
    return ds, turb, arrays


def test_iterative_offset_recovers_target(offset_setup):
    """Construct a target CF that is achievable, then check the iterative
    solver finds an offset reproducing it."""
    _, _, arrays = offset_setup
    true_offset = 1.5
    target_cf = fast_simulate_cf(arrays, 1.0, true_offset)
    row = pd.Series({"obs": target_cf, "sim": fast_simulate_cf(arrays, 1.0, 0.0),
                     "scalar": 1.0, "year": 2020, "cluster": 0, "time_slice": "1/1"})
    found = _find_offset_iterative(row, arrays)
    achieved = fast_simulate_cf(arrays, 1.0, found)
    assert achieved == pytest.approx(target_cf, abs=2e-3)


def test_scipy_offset_fallback_recovers_target(offset_setup):
    _, _, arrays = offset_setup
    target_cf = fast_simulate_cf(arrays, 1.0, -0.8)
    row = pd.Series({"obs": target_cf, "sim": fast_simulate_cf(arrays, 1.0, 0.0),
                     "scalar": 1.0, "year": 2020, "cluster": 0, "time_slice": "1/1"})
    found = _find_offset_scipy(row, arrays, bounds=(-3, 3))
    achieved = fast_simulate_cf(arrays, 1.0, found)
    assert achieved == pytest.approx(target_cf, abs=2e-3)


def test_find_offset_end_to_end(offset_setup, power_curve):
    ds, turb, arrays = offset_setup
    target_cf = fast_simulate_cf(arrays, 1.0, 1.0)
    row = pd.Series({"obs": target_cf, "sim": fast_simulate_cf(arrays, 1.0, 0.0),
                     "scalar": 1.0, "year": 2020, "cluster": 0, "time_slice": "1/1"})
    offset = find_offset(row, turb, ds, power_curve)
    assert np.isfinite(offset)
    assert offset == pytest.approx(1.0, abs=0.1)


def test_find_offset_empty_cluster_returns_nan(offset_setup, power_curve):
    ds, turb, arrays = offset_setup
    row = pd.Series({"obs": 0.4, "sim": 0.3, "scalar": 1.0,
                     "year": 2020, "cluster": 99, "time_slice": "1/1"})
    assert np.isnan(find_offset(row, turb, ds, power_curve))
