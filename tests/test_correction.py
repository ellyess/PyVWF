"""Tests for the linear (scalar + offset) bias-correction scheme."""

import numpy as np
import pandas as pd
import pytest

from vwf.correction import (
    calculate_scalar,
    find_offset,
)
from scipy import interpolate as interp

import vwf.correction as correction
from vwf.wind import interpolate_wind, prepare_offset_arrays, fast_simulate_cf


def _bias_df(obs, sim):
    return pd.DataFrame(
        {
            "fixed": ["1/1"] * len(obs),
            "cluster": [0] * len(obs),
            "year": [2020] * len(obs),
            "obs": obs,
            "sim": sim,
            "capacity": [1.0] * len(obs),
        }
    )


def test_scalar_is_obs_over_sim_ratio():
    df = _bias_df([0.4, 0.4], [0.5, 0.5])
    out = calculate_scalar(df, "fixed")
    assert out["scalar"].iloc[0] == pytest.approx(0.8)


def test_scalar_capacity_weighted():
    df = pd.DataFrame(
        {
            "fixed": ["1/1", "1/1"],
            "cluster": [0, 0],
            "year": [2020, 2020],
            "obs": [0.6, 0.2],
            "sim": [0.5, 0.5],
            "capacity": [3.0, 1.0],  # heavier weight on the 0.6 obs
        }
    )
    out = calculate_scalar(df, "fixed")
    # weighted obs = (0.6*3 + 0.2*1)/4 = 0.5 ; weighted sim = 0.5 ; scalar = 1.0
    assert out["scalar"].iloc[0] == pytest.approx(1.0)


def test_scalar_all_nan_group_returns_nan():
    df = _bias_df([np.nan, np.nan], [0.5, 0.5])
    out = calculate_scalar(df, "fixed")
    assert np.isnan(out["scalar"].iloc[0])


def test_scalar_ignores_capacity_of_non_reporting_plants():
    """A plant that reported nothing must not dilute the observed mean.

    ``sim`` exists for every plant but ``obs`` does not, so the weighted mean
    of ``obs`` must divide by the capacity that ACTUALLY REPORTED, not by the
    whole fleet's capacity. Dividing by the whole fleet scales ``obs`` down by
    the reporting fraction while leaving ``sim`` untouched, so the scalar comes
    out as ``true_scalar * reporting_fraction``.

    Here two of four equal plants report, obs 0.4 against sim 0.5, so the
    scalar is 0.8. Diluting by the whole fleet gives 0.8 * 0.5 = 0.4.

    Observed on the real US fleet: only 43% of capacity reports monthly, so the
    fitted scalar was 0.47 where the correct value is 1.09 - the correction
    pushed wind DOWN 53% when it should have nudged it UP 9%.
    """
    df = pd.DataFrame(
        {
            "fixed": ["1/1"] * 4,
            "cluster": [0] * 4,
            "year": [2020] * 4,
            "obs": [0.4, 0.4, np.nan, np.nan],  # half the fleet reports
            "sim": [0.5, 0.5, 0.5, 0.5],  # sim is never missing
            "capacity": [1.0, 1.0, 1.0, 1.0],
        }
    )
    out = calculate_scalar(df, "fixed")
    assert out["scalar"].iloc[0] == pytest.approx(0.8)


def test_scalar_reporting_weights_stay_capacity_weighted():
    """Must-distinguish: the fix has to weight by capacity, not count rows.

    The two reporting plants carry very different capacities, and the silent
    plant carries most of the fleet. A fix that simply counted reporting rows
    (or dropped weighting) would return 0.4/0.5 = 0.8; the capacity-weighted
    answer over the reporting plants is (0.6*3 + 0.2*1)/4 = 0.5, scalar 1.0.
    The uncorrected dilution bug returns 0.02/0.5 = 0.04.
    """
    df = pd.DataFrame(
        {
            "fixed": ["1/1"] * 3,
            "cluster": [0] * 3,
            "year": [2020] * 3,
            "obs": [0.6, 0.2, np.nan],
            "sim": [0.5, 0.5, 0.5],
            "capacity": [3.0, 1.0, 96.0],  # the silent plant dominates the fleet
        }
    )
    out = calculate_scalar(df, "fixed")
    assert out["scalar"].iloc[0] == pytest.approx(1.0)


def test_scalar_averages_obs_and_sim_over_the_same_plants():
    """obs and sim must come from the SAME sample, not each from its own.

    The two tests above hold ``sim`` uniform across the fleet, so the reporting
    and non-reporting plants have identical simulated output and the bug is
    invisible. Here the silent plant simulates much higher than the reporter.

    Masking each column against only its own presence averages ``obs`` over the
    reporters and ``sim`` over everyone, so the ratio compares two different
    samples: 0.36 / mean(0.30, 0.50) = 0.36 / 0.40 = 0.90. Comparing like with
    like over the reporter alone gives 0.36 / 0.30 = 1.20, which is the true
    bias applied to that plant.

    This is not a corner case. Reporting is not independent of output, and at
    the 43% reporting rate cited above a moderate correlation biases the scalar
    by about 7% and a strong one by about 14%, always in the same direction.
    """
    df = pd.DataFrame(
        {
            "fixed": ["1/1"] * 2,
            "cluster": [0] * 2,
            "year": [2020] * 2,
            "obs": [0.36, np.nan],
            "sim": [0.30, 0.50],  # the silent plant simulates far higher
            "capacity": [100.0, 100.0],
        }
    )
    out = calculate_scalar(df, "fixed")
    assert out["scalar"].iloc[0] == pytest.approx(1.2)
    assert out["sim"].iloc[0] == pytest.approx(0.30)  # not 0.40


@pytest.fixture
def offset_setup(make_reanalysis, power_curve):
    ds = make_reanalysis(n_hours=72, mean_speed=8.0, seed=5)
    turb = pd.DataFrame(
        {
            "ID": ["a"],
            "lat": [55.5],
            "lon": [8.5],
            "height": [100.0],
            "model": ["GE.1.5sle"],
            "capacity": [1.0],
            "cluster": [0],
        }
    )
    ws = interpolate_wind(ds, turb)
    arrays = prepare_offset_arrays(ws, power_curve)
    return ds, turb, arrays


def test_find_offset_end_to_end(offset_setup, power_curve):
    ds, turb, arrays = offset_setup
    target_cf = fast_simulate_cf(arrays, 1.0, 1.0)
    row = pd.Series(
        {
            "obs": target_cf,
            "sim": fast_simulate_cf(arrays, 1.0, 0.0),
            "scalar": 1.0,
            "year": 2020,
            "cluster": 0,
            "time_slice": "1/1",
        }
    )
    offset = find_offset(row, turb, ds, power_curve)
    assert np.isfinite(offset)
    assert offset == pytest.approx(1.0, abs=0.1)


def test_find_offset_empty_cluster_returns_nan(offset_setup, power_curve):
    ds, turb, arrays = offset_setup
    row = pd.Series(
        {"obs": 0.4, "sim": 0.3, "scalar": 1.0, "year": 2020, "cluster": 99, "time_slice": "1/1"}
    )
    assert np.isnan(find_offset(row, turb, ds, power_curve))


# ------------------------------------------------------------- helpers ----


def _arrays(speeds, capacity=(1.0,)):
    """Offset arrays with one linear curve, so the root is analytic."""
    grid = np.arange(0.0, 30.01, 0.5)
    curve = interp.Akima1DInterpolator(grid, np.clip(grid / 20.0, 0, 1))
    ws = np.asarray(speeds, dtype=float).reshape(-1, 1)
    return {
        "ws_data": ws,
        "model_groups": [(curve, np.array([True]))],
        "capacities": np.asarray(capacity, dtype=float),
    }


def _row(obs, sim, scalar=1.0):
    return pd.Series(
        {"obs": obs, "sim": sim, "scalar": scalar, "cluster": 0, "year": 2020, "time_slice": "1/1"}
    )


# ------------------------------------------------------- bracketed search ----
# The search find_offset uses: a bracket found by stepping outward from zero,
# then Brent's method, refusing explicitly rather than returning a value at a
# bound (issue #18).


@pytest.mark.parametrize("true_offset", [2.0, -1.25, 0.3])
def test_the_bracketed_search_returns_the_root_to_solver_precision(true_offset):
    arrays = _arrays([8.0, 10.0, 12.0])
    target = fast_simulate_cf(arrays, 1.0, true_offset)
    row = _row(target, fast_simulate_cf(arrays, 1.0, 0.0))
    got = correction._find_offset_bracketed(row, arrays)
    assert got == pytest.approx(true_offset, abs=1e-5)
    assert abs(target - fast_simulate_cf(arrays, 1.0, got)) <= correction.BRACKETED_MAX_RESIDUAL


def test_the_bracketed_search_refuses_a_root_outside_the_bounds():
    arrays = _arrays([8.0, 10.0, 12.0])
    target = fast_simulate_cf(arrays, 1.0, 8.0)
    row = _row(target, fast_simulate_cf(arrays, 1.0, 0.0))
    assert np.isnan(correction._find_offset_bracketed(row, arrays, bounds=(-3, 3)))
    assert correction._find_offset_bracketed(row, arrays, bounds=(-3, 12)) == pytest.approx(
        8.0, abs=1e-5
    )


def test_the_bracketed_search_refuses_a_root_at_a_bound():
    arrays = _arrays([8.0, 10.0, 12.0])
    target = fast_simulate_cf(arrays, 1.0, 3.0)
    row = _row(target, fast_simulate_cf(arrays, 1.0, 0.0))
    assert np.isnan(correction._find_offset_bracketed(row, arrays, bounds=(-3, 3)))
    assert correction._find_offset_bracketed(row, arrays, bounds=(-3, 3.5)) == pytest.approx(
        3.0, abs=1e-5
    )


def test_the_bracketed_search_takes_the_root_nearest_zero_past_cut_out():
    """With a cut-out the capacity factor rises and then falls with the offset,
    so a target below the peak is reached twice. The nearer root is the one."""
    grid = np.arange(0.0, 30.01, 0.5)
    cf = np.where(grid <= 25.0, np.clip(grid / 20.0, 0, 1), 0.0)
    curve = interp.Akima1DInterpolator(grid, cf)
    arrays = {
        "ws_data": np.array([[10.0]]),
        "model_groups": [(curve, np.array([True]))],
        "capacities": np.array([1.0]),
    }
    target = 0.8  # reached at offset +6 (16 m/s), and again only after cut-out
    got = correction._find_offset_bracketed(_row(target, 0.5), arrays)
    assert got == pytest.approx(6.0, abs=1e-4)


def test_the_bracketed_search_refuses_a_sign_change_that_is_a_jump():
    """A sample past the end of the curve grid is off the curve and leaves the
    mean. If it sat above cut-out, at zero, the mean jumps up as it leaves, and
    can jump across the observation without crossing it. Brent's method then
    converges onto the jump, and the residual check refuses it."""
    grid = np.arange(0.0, 30.01, 0.5)
    cf = np.where(grid <= 25.0, np.clip(grid / 20.0, 0, 1), 0.0)
    curve = interp.Akima1DInterpolator(grid, cf)
    arrays = {
        "ws_data": np.array([[29.0], [5.0]]),
        "model_groups": [(curve, np.array([True]))],
        "capacities": np.array([1.0]),
    }
    # At offset 0 the mean is (0 + 0.25) / 2; it rises to 0.15 at offset 1,
    # where the 29 m/s sample passes 30 m/s and leaves, and the mean jumps to 0.3.
    assert fast_simulate_cf(arrays, 1.0, 0.99) < 0.2 < fast_simulate_cf(arrays, 1.0, 1.01)
    assert np.isnan(correction._find_offset_bracketed(_row(0.2, 0.125), arrays))
