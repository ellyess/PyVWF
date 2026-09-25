"""Tests for the error metrics that back PyVWF's published numbers.

The strategy throughout is known-answer testing: construct simulated and
observed capacity factors whose error is analytically obvious (a perfect
simulation, a constant additive bias, a bias that differs between turbines of
known capacity), then assert the reported RMSE/MAE/MBE equal the value worked
out by hand. Snapshot-style assertions would lock in whatever the code happens
to do today, including its bugs; these pin what the metrics are *supposed* to
mean.

Capacity weighting is exercised explicitly because it is the easiest thing to
get silently wrong: an unweighted mean over turbines of unequal size looks
plausible and is off by a few percent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyvwf.metrics import (
    REMOVED_MODES,
    calculate_error,
    prepare_monthly_data,
    weighted_average_vectorized,
)


# ---------------------------------------------------------------------------
# Builders for the on-disk schema the metrics layer consumes
# ---------------------------------------------------------------------------


def wide_cf(cf_by_id: dict[str, float], years=(2020,), months=12) -> pd.DataFrame:
    """Wide CF frame (`time` + one column per turbine), as written to disk."""
    times = pd.date_range(f"{min(years)}-01-01", periods=months * len(years), freq="MS")
    df = pd.DataFrame({turb: np.full(len(times), cf) for turb, cf in cf_by_id.items()})
    df.insert(0, "time", times)
    return df


def fleet(capacities: dict[str, float], **extra) -> pd.DataFrame:
    df = pd.DataFrame({"ID": list(capacities), "capacity": list(capacities.values())})
    for col, values in extra.items():
        df[col] = values
    return df


# ---------------------------------------------------------------------------
# weighted_average_vectorized
# ---------------------------------------------------------------------------


def test_weighted_average_equal_weights_is_plain_mean():
    df = pd.DataFrame({"v": [0.1, 0.2, 0.6], "w": [1.0, 1.0, 1.0]})
    assert weighted_average_vectorized(df, "v", "w") == pytest.approx(0.3)


def test_weighted_average_respects_weights():
    # 0.2 at weight 3 and 0.6 at weight 1 -> (0.6 + 0.6) / 4 = 0.3
    df = pd.DataFrame({"v": [0.2, 0.6], "w": [3.0, 1.0]})
    assert weighted_average_vectorized(df, "v", "w") == pytest.approx(0.3)


def test_weighted_average_is_dominated_by_the_large_turbine():
    """A 10 MW turbine at CF 0.5 and a 0.1 MW turbine at CF 0.0 must report
    close to 0.5, not the unweighted 0.25."""
    df = pd.DataFrame({"v": [0.5, 0.0], "w": [10.0, 0.1]})
    assert weighted_average_vectorized(df, "v", "w") == pytest.approx(0.5 / 1.01, rel=1e-9)


# ---------------------------------------------------------------------------
# prepare_monthly_data
# ---------------------------------------------------------------------------


def test_prepare_monthly_data_test_schema_melts_to_long():
    sim = wide_cf({"A": 0.4, "B": 0.5})
    obs = wide_cf({"A": 0.3, "B": 0.3})
    sim_m, obs_m = prepare_monthly_data(sim, obs)

    assert set(sim_m.columns) >= {"year", "month", "ID", "cf"}
    assert set(obs_m.columns) >= {"year", "month", "ID", "cf"}
    assert sorted(sim_m["ID"].unique()) == ["A", "B"]
    assert len(sim_m) == 24  # 2 turbines x 12 months


def test_prepare_monthly_data_does_not_mutate_inputs():
    sim = wide_cf({"A": 0.4})
    obs = wide_cf({"A": 0.3})
    sim_before, obs_before = sim.copy(), obs.copy()

    prepare_monthly_data(sim, obs)

    pd.testing.assert_frame_equal(sim, sim_before)
    pd.testing.assert_frame_equal(obs, obs_before)


# ---------------------------------------------------------------------------
# calculate_error: known answers
# ---------------------------------------------------------------------------


def test_perfect_simulation_has_zero_error():
    sim = wide_cf({"A": 0.35, "B": 0.35})
    obs = wide_cf({"A": 0.35, "B": 0.35})
    rmse, mae, mbe = calculate_error("total", sim, obs, fleet({"A": 1000.0, "B": 2000.0}))

    assert rmse == pytest.approx(0.0, abs=1e-12)
    assert mae == pytest.approx(0.0, abs=1e-12)
    assert mbe == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("bias", [0.05, -0.05])
def test_constant_bias_is_reported_exactly(bias):
    """With every turbine biased by the same amount b: MBE = b (signed),
    MAE = |b|, RMSE = |b|."""
    obs_cf = 0.30
    sim = wide_cf({"A": obs_cf + bias, "B": obs_cf + bias})
    obs = wide_cf({"A": obs_cf, "B": obs_cf})

    rmse, mae, mbe = calculate_error("total", sim, obs, fleet({"A": 1000.0, "B": 2000.0}))

    assert mbe == pytest.approx(bias)
    assert mae == pytest.approx(abs(bias))
    assert rmse == pytest.approx(abs(bias))


def test_error_is_capacity_weighted_across_turbines():
    """Turbine A (1 MW) over-predicts by 0.10; turbine B (3 MW) is perfect.
    Capacity-weighted MBE = (1*0.10 + 3*0.0) / 4 = 0.025, not the unweighted
    0.05 an equal-weight mean would give."""
    obs = wide_cf({"A": 0.30, "B": 0.30})
    sim = wide_cf({"A": 0.40, "B": 0.30})

    rmse, mae, mbe = calculate_error("total", sim, obs, fleet({"A": 1000.0, "B": 3000.0}))

    assert mbe == pytest.approx(0.025)
    assert mae == pytest.approx(0.025)
    # RMSE weights the *squared* error: sqrt((1*0.01 + 3*0) / 4) = 0.05
    assert rmse == pytest.approx(0.05)
    assert mbe != pytest.approx(0.05), "metric ignored capacity weighting"


def test_seasonal_error_cancels_in_mbe_but_not_rmse():
    """A simulation that is too high in winter and too low in summer has ~zero
    mean bias yet a real RMSE: the case MBE alone would hide."""
    times = pd.date_range("2020-01-01", periods=12, freq="MS")
    swing = np.where(times.month <= 6, 0.10, -0.10)
    sim = pd.DataFrame({"time": times, "A": 0.30 + swing})
    obs = pd.DataFrame({"time": times, "A": np.full(12, 0.30)})

    rmse, mae, mbe = calculate_error("total", sim, obs, fleet({"A": 1000.0}))

    assert mbe == pytest.approx(0.0, abs=1e-12)
    assert mae == pytest.approx(0.10)
    assert rmse == pytest.approx(0.10)


def test_calculate_error_rejects_unknown_mode():
    sim = wide_cf({"A": 0.4})
    obs = wide_cf({"A": 0.3})
    with pytest.raises(ValueError, match="Unknown error type"):
        calculate_error("not-a-mode", sim, obs, fleet({"A": 1000.0}))


@pytest.mark.parametrize("mode", REMOVED_MODES)
def test_a_removed_mode_says_it_was_removed(mode):
    sim = wide_cf({"A": 0.4})
    obs = wide_cf({"A": 0.3})
    with pytest.raises(ValueError, match="removed with the legacy path"):
        calculate_error(mode, sim, obs, fleet({"A": 1000.0}))


def test_calculate_error_does_not_mutate_turb_info():
    """Callers evaluate many configurations against one fleet table."""
    sim = wide_cf({"A": 0.4})
    obs = wide_cf({"A": 0.3})
    turb = fleet({"A": 1000.0})
    turb["ID"] = turb["ID"].astype("object")
    before = turb.copy()

    calculate_error("total", sim, obs, turb)

    pd.testing.assert_frame_equal(turb, before)
