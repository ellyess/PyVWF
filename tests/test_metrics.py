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

from vwf.metrics import (
    calculate_error,
    overall_error,
    prepare_monthly_data,
    weighted_average_vectorized,
)


# ---------------------------------------------------------------------------
# Builders for the two on-disk schemas the metrics layer consumes
# ---------------------------------------------------------------------------

def wide_cf(cf_by_id: dict[str, float], years=(2020,), months=12) -> pd.DataFrame:
    """Wide CF frame (`time` + one column per turbine), as written to disk."""
    times = pd.date_range(f"{min(years)}-01-01", periods=months * len(years), freq="MS")
    df = pd.DataFrame({turb: np.full(len(times), cf) for turb, cf in cf_by_id.items()})
    df.insert(0, "time", times)
    return df


def long_obs(cf_by_id: dict[str, float], years=(2020,)) -> pd.DataFrame:
    """Long observation frame (ID, year, month, obs) — the `train=True` schema."""
    rows = [
        {"ID": turb, "year": year, "month": month, "obs": cf}
        for turb, cf in cf_by_id.items()
        for year in years
        for month in range(1, 13)
    ]
    return pd.DataFrame(rows)


def fleet(capacities: dict[str, float], **extra) -> pd.DataFrame:
    df = pd.DataFrame(
        {"ID": list(capacities), "capacity": list(capacities.values())}
    )
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


@pytest.mark.parametrize(
    "years",
    [
        (2015, 2016, 2017, 2018, 2019),  # the original hard-coded window
        (2016, 2017, 2018),              # shorter: used to raise ValueError
        (2016, 2017, 2018, 2019, 2020),  # 60 months, but shifted by a year
        (2021,),                         # a single year
    ],
)
def test_prepare_monthly_data_train_preserves_real_years(years):
    """Regression: the training branch used to discard the real (year, month)
    index and overwrite it with a hard-coded 2015-01..2019-12 range. Any window
    that wasn't exactly those 60 months either crashed or — for a 60-month
    window starting elsewhere — was silently relabelled, merging observations
    against the wrong year's simulation."""
    obs = long_obs({"A": 0.3, "B": 0.3}, years=years)
    sim = wide_cf({"A": 0.4, "B": 0.4}, years=years)

    _, obs_m = prepare_monthly_data(sim, obs, train=True)

    assert sorted(int(y) for y in obs_m["year"].unique()) == sorted(years)
    assert len(obs_m) == 2 * 12 * len(years)


def test_prepare_monthly_data_train_merges_onto_matching_years():
    """The point of preserving the years: every observation must line up with
    the simulation from the *same* month."""
    years = (2016, 2017, 2018, 2019, 2020)
    obs = long_obs({"A": 0.3}, years=years)
    sim = wide_cf({"A": 0.4}, years=years)

    sim_m, obs_m = prepare_monthly_data(sim, obs, train=True)
    merged = sim_m.merge(obs_m, on=["ID", "year", "month"], suffixes=("_sim", "_obs"))

    assert len(merged) == 12 * len(years)  # nothing dropped by a year mismatch


def test_prepare_monthly_data_does_not_mutate_inputs():
    sim = wide_cf({"A": 0.4})
    obs = wide_cf({"A": 0.3})
    sim_before, obs_before = sim.copy(), obs.copy()

    prepare_monthly_data(sim, obs)

    pd.testing.assert_frame_equal(sim, sim_before)
    pd.testing.assert_frame_equal(obs, obs_before)


# ---------------------------------------------------------------------------
# calculate_error — known answers
# ---------------------------------------------------------------------------

ERROR_MODES = ["total", "spatial-focus", "temporal-focus"]


@pytest.mark.parametrize("mode", ERROR_MODES)
def test_perfect_simulation_has_zero_error(mode):
    sim = wide_cf({"A": 0.35, "B": 0.35})
    obs = wide_cf({"A": 0.35, "B": 0.35})
    rmse, mae, mbe = calculate_error(mode, sim, obs, fleet({"A": 1000.0, "B": 2000.0}))

    assert rmse == pytest.approx(0.0, abs=1e-12)
    assert mae == pytest.approx(0.0, abs=1e-12)
    assert mbe == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("mode", ERROR_MODES)
@pytest.mark.parametrize("bias", [0.05, -0.05])
def test_constant_bias_is_reported_exactly(mode, bias):
    """With every turbine biased by the same amount b: MBE = b (signed),
    MAE = |b|, RMSE = |b|."""
    obs_cf = 0.30
    sim = wide_cf({"A": obs_cf + bias, "B": obs_cf + bias})
    obs = wide_cf({"A": obs_cf, "B": obs_cf})

    rmse, mae, mbe = calculate_error(mode, sim, obs, fleet({"A": 1000.0, "B": 2000.0}))

    assert mbe == pytest.approx(bias)
    assert mae == pytest.approx(abs(bias))
    assert rmse == pytest.approx(abs(bias))


@pytest.mark.parametrize("mode", ["total", "spatial-focus"])
def test_error_is_capacity_weighted_across_turbines(mode):
    """Turbine A (1 MW) over-predicts by 0.10; turbine B (3 MW) is perfect.
    Capacity-weighted MBE = (1*0.10 + 3*0.0) / 4 = 0.025, not the unweighted
    0.05 an equal-weight mean would give."""
    obs = wide_cf({"A": 0.30, "B": 0.30})
    sim = wide_cf({"A": 0.40, "B": 0.30})

    rmse, mae, mbe = calculate_error(mode, sim, obs, fleet({"A": 1000.0, "B": 3000.0}))

    assert mbe == pytest.approx(0.025)
    assert mae == pytest.approx(0.025)
    # RMSE weights the *squared* error: sqrt((1*0.01 + 3*0) / 4) = 0.05
    assert rmse == pytest.approx(0.05)
    assert mbe != pytest.approx(0.05), "metric ignored capacity weighting"


def test_temporal_focus_aggregates_over_turbines_before_scoring():
    """Temporal focus scores the country-level monthly series, so equal and
    opposite turbine errors within a month cancel — unlike spatial focus,
    which scores each turbine and cannot cancel."""
    obs = wide_cf({"A": 0.30, "B": 0.30})
    sim = wide_cf({"A": 0.40, "B": 0.20})  # +0.10 and -0.10, equal capacity
    turb = fleet({"A": 1000.0, "B": 1000.0})

    _, _, mbe_temporal = calculate_error("temporal-focus", sim, obs, turb)
    rmse_spatial, mae_spatial, _ = calculate_error("spatial-focus", sim, obs, turb)

    assert mbe_temporal == pytest.approx(0.0, abs=1e-12)
    assert mae_spatial == pytest.approx(0.10)
    assert rmse_spatial == pytest.approx(0.10)


def test_seasonal_error_cancels_in_mbe_but_not_rmse():
    """A simulation that is too high in winter and too low in summer has ~zero
    mean bias yet a real RMSE — the case MBE alone would hide."""
    times = pd.date_range("2020-01-01", periods=12, freq="MS")
    swing = np.where(times.month <= 6, 0.10, -0.10)
    sim = pd.DataFrame({"time": times, "A": 0.30 + swing})
    obs = pd.DataFrame({"time": times, "A": np.full(12, 0.30)})

    rmse, mae, mbe = calculate_error("temporal-focus", sim, obs, fleet({"A": 1000.0}))

    assert mbe == pytest.approx(0.0, abs=1e-12)
    assert mae == pytest.approx(0.10)
    assert rmse == pytest.approx(0.10)


def test_calculate_error_rejects_unknown_mode():
    sim = wide_cf({"A": 0.4})
    obs = wide_cf({"A": 0.3})
    with pytest.raises(ValueError, match="Unknown error type"):
        calculate_error("not-a-mode", sim, obs, fleet({"A": 1000.0}))


def test_calculate_error_does_not_mutate_turb_info():
    """Callers evaluate many configurations against one fleet table."""
    sim = wide_cf({"A": 0.4})
    obs = wide_cf({"A": 0.3})
    turb = fleet({"A": 1000.0})
    turb["ID"] = turb["ID"].astype("object")
    before = turb.copy()

    calculate_error("total", sim, obs, turb)

    pd.testing.assert_frame_equal(turb, before)


# ---------------------------------------------------------------------------
# calculate_error — the grouped report modes
# ---------------------------------------------------------------------------

def test_monthly_error_reports_one_diff_per_month_plus_both():
    obs = wide_cf({"A": 0.30, "B": 0.30})
    sim = wide_cf({"A": 0.35, "B": 0.35})
    turb = fleet({"A": 1000.0, "B": 1000.0}, **{"In training?": ["Yes", "No"]})

    out = calculate_error("monthly-error", sim, obs, turb)

    assert set(out.columns) >= {"diff", "In training?"}
    assert set(out["In training?"].unique()) == {"Yes", "No", "Both"}
    both = out[out["In training?"] == "Both"]
    assert len(both) == 12
    assert both["diff"].to_numpy() == pytest.approx(np.full(12, 0.05))


def test_cluster_error_reports_one_diff_per_cluster():
    obs = wide_cf({"A": 0.30, "B": 0.30})
    sim = wide_cf({"A": 0.40, "B": 0.30})  # cluster 0 biased, cluster 1 perfect
    turb = fleet({"A": 1000.0, "B": 1000.0}, cluster=[0, 1])

    diffs, counts = calculate_error("cluster-error", sim, obs, turb)

    assert diffs.loc[0] == pytest.approx(0.10)
    assert diffs.loc[1] == pytest.approx(0.0, abs=1e-12)
    assert counts.loc[0] == 1


def test_turbine_error_reports_one_diff_per_turbine():
    obs = wide_cf({"A": 0.30, "B": 0.30})
    sim = wide_cf({"A": 0.40, "B": 0.25})
    turb = fleet({"A": 1000.0, "B": 1000.0}, distance=[0.1, 0.2])

    out = calculate_error("turbine-error", sim, obs, turb)

    assert out.loc["A", "diff"] == pytest.approx(0.10)
    assert out.loc["B", "diff"] == pytest.approx(-0.05)


# ---------------------------------------------------------------------------
# overall_error — sweeps a run directory
# ---------------------------------------------------------------------------

def _write_run(tmp_path, country="DK", year=2020):
    """A run directory holding obs, uncorrected and two corrected variants."""
    cf_dir = tmp_path / "results" / "capacity-factor"
    cf_dir.mkdir(parents=True)

    wide_cf({"A": 0.30, "B": 0.30}).to_csv(cf_dir / f"{country}_{year}_obs_cf.csv", index=False)
    # Uncorrected over-predicts by 0.10; each corrected variant is closer.
    wide_cf({"A": 0.40, "B": 0.40}).to_csv(cf_dir / f"{country}_{year}_unc_cf.csv", index=False)
    for n_clu, cf in [(1, 0.34), (10, 0.31)]:
        wide_cf({"A": cf, "B": cf}).to_csv(
            cf_dir / f"{country}_{year}_fixed_{n_clu}_cor_cf.csv", index=False
        )
    return tmp_path


def test_overall_error_sweeps_clusters_and_resolutions(tmp_path):
    run = _write_run(tmp_path)
    turb = fleet({"A": 1000.0, "B": 1000.0})

    metrics = overall_error(
        "total", str(run), "DK", turb, [1, 10], ["fixed"], False, 2020
    )

    # One uncorrected baseline row + one row per (cluster, time_res) pair
    assert len(metrics) == 3
    assert set(metrics.columns) == {"num_clu", "time_res", "rmse", "mae", "mbe"}

    baseline = metrics[metrics["time_res"] == "uncorrected"].iloc[0]
    assert baseline["mbe"] == pytest.approx(0.10)

    corrected = metrics[metrics["time_res"] == "fixed"].set_index("num_clu")
    assert corrected.loc[1, "mbe"] == pytest.approx(0.04)
    assert corrected.loc[10, "mbe"] == pytest.approx(0.01)

    # The whole point of the sweep: more clusters -> less error, and every
    # corrected variant beats the uncorrected baseline.
    assert corrected.loc[10, "rmse"] < corrected.loc[1, "rmse"] < baseline["rmse"]
