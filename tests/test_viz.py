"""Tests for the vwf.viz distributional plotting layer."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from vwf.viz import (
    Results,
    load_results,
    plot_cf_distribution,
    plot_qq,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def cf_series(rng):
    """Synthetic CF series with the bias structure we want to visualise.

    A Weibull wind field passed through a synthetic power curve, plus a
    reanalysis-like over-dispersed copy and a mean-matching linear correction.
    No external dependencies — used to exercise the plot functions.
    """
    n = 4000

    def power_curve(speed):
        cf = np.zeros_like(speed, dtype=float)
        ramp = (speed >= 3.0) & (speed < 13.0)
        cf[ramp] = ((speed[ramp] - 3.0) / 10.0) ** 3
        cf[(speed >= 13.0) & (speed <= 25.0)] = 1.0
        return np.clip(cf, 0.0, 1.0)

    obs_wind = np.clip(rng.weibull(2.0, n) * 8.0, 0, None)
    m = obs_wind.mean()
    biased = np.clip(m + 1.2 + (obs_wind - m) * 1.6, 0, None)
    lin_wind = biased + (obs_wind.mean() - biased.mean())

    return {
        "obs": power_curve(obs_wind),
        "uncorrected": power_curve(biased),
        "linear": power_curve(lin_wind),
    }


# ---------------------------------------------------------------------------
# load_results
# ---------------------------------------------------------------------------

def _write_wide_cf(path, ids, values):
    """Write a wide CF csv matching the on-disk schema."""
    times = pd.date_range("2020-01-01", periods=values.shape[0], freq="h")
    df = pd.DataFrame(values, columns=ids)
    df.insert(0, "time", times)
    df.to_csv(path, index=False)


def _build_run_dir(tmp_path, country="DK", year=2020, include_corrected=True):
    run = tmp_path / "run"
    cf_dir = run / "results" / "capacity-factor"
    train_dir = run / "training" / "simulated-turbines"
    fac_dir = run / "training" / "correction-factors"
    for d in (cf_dir, train_dir, fac_dir):
        d.mkdir(parents=True)

    ids = ["A", "B", "C"]
    n = 200
    rng = np.random.default_rng(0)
    obs = np.clip(rng.normal(0.35, 0.15, (n, 3)), 0, 1)
    unc = np.clip(rng.normal(0.45, 0.20, (n, 3)), 0, 1)
    _write_wide_cf(cf_dir / f"{country}_{year}_obs_cf.csv", ids, obs)
    _write_wide_cf(cf_dir / f"{country}_{year}_unc_cf.csv", ids, unc)

    if include_corrected:
        cor = np.clip(rng.normal(0.35, 0.18, (n, 3)), 0, 1)
        _write_wide_cf(cf_dir / f"{country}_{year}_fixed_5_cor_cf.csv", ids, cor)
        pd.DataFrame({
            "cluster": [0, 1, 2, 3, 4],
            "fixed": ["1/1"] * 5,
            "scalar": [1.0, 0.9, 1.1, 1.0, 0.95],
            "offset": [0.0, 0.1, -0.1, 0.05, -0.05],
        }).to_csv(fac_dir / f"{country}_factors_fixed_5.csv", index=False)

    pd.DataFrame({
        "ID": ids,
        "capacity": [1000.0, 500.0, 2000.0],
        "lat": [55.0, 55.5, 56.0],
        "lon": [8.0, 8.5, 9.0],
    }).to_csv(train_dir / f"{country}_{year}_turb_info.csv", index=False)

    return run


def test_load_results_roundtrip(tmp_path):
    run = _build_run_dir(tmp_path, include_corrected=True)
    res = load_results(run, "DK", 2020)

    assert isinstance(res, Results)
    assert res.country == "DK"
    assert res.year == 2020
    assert len(res.obs) == 200
    assert len(res.uncorrected) == 200

    # Discovered the corrected file by glob, not by user input
    assert (5, "fixed") in res.corrected
    assert (5, "fixed") in res.factors
    assert set(res.factors[(5, "fixed")].columns) >= {"cluster", "scalar", "offset"}


def test_load_results_missing_corrected_ok(tmp_path):
    run = _build_run_dir(tmp_path, include_corrected=False)
    res = load_results(run, "DK", 2020)
    assert res.corrected == {}
    assert res.factors == {}


def test_load_results_capacity_weighting_changes_series(tmp_path):
    run = _build_run_dir(tmp_path, include_corrected=False)
    weighted = load_results(run, "DK", 2020, weight_by_capacity=True).obs
    plain = load_results(run, "DK", 2020, weight_by_capacity=False).obs
    # Capacities are not equal, so the two aggregations should disagree
    assert not np.allclose(weighted.to_numpy(), plain.to_numpy())


def test_load_results_missing_run_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path / "does-not-exist", "DK", 2020)


def test_load_results_aligns_sim_cadence_to_obs(tmp_path):
    """Turbine-level runs store obs monthly and sims daily; without alignment
    the distributions are 12 vs 366 samples and not comparable."""
    run = tmp_path / "run"
    cf = run / "results" / "capacity-factor"
    train = run / "training" / "simulated-turbines"
    fac = run / "training" / "correction-factors"
    for d in (cf, train, fac):
        d.mkdir(parents=True)

    ids = ["A", "B"]
    rng = np.random.default_rng(1)
    obs = pd.DataFrame(rng.uniform(0.1, 0.5, (12, 2)), columns=ids)
    obs.insert(0, "time", pd.date_range("2020-01-01", periods=12, freq="MS"))
    obs.to_csv(cf / "DK_2020_obs_cf.csv", index=False)
    daily_idx = pd.date_range("2020-01-01", periods=366, freq="D")
    for name in ["DK_2020_unc_cf.csv", "DK_2020_fixed_5_cor_cf.csv"]:
        df = pd.DataFrame(rng.uniform(0.1, 0.6, (366, 2)), columns=ids)
        df.insert(0, "time", daily_idx)
        df.to_csv(cf / name, index=False)
    pd.DataFrame({"ID": ids, "capacity": [1.0, 1.0]}).to_csv(
        train / "DK_2020_turb_info.csv", index=False
    )

    aligned = load_results(run, "DK", 2020)
    assert len(aligned.obs) == 12
    assert len(aligned.uncorrected) == 12
    assert len(aligned.corrected[(5, "fixed")]) == 12
    assert (aligned.uncorrected.index == aligned.obs.index).all()

    raw = load_results(run, "DK", 2020, align_to_obs=False)
    assert len(raw.uncorrected) == 366


# ---------------------------------------------------------------------------
# plot_cf_distribution
# ---------------------------------------------------------------------------

def test_plot_cf_distribution_returns_figure(cf_series):
    obs = cf_series.pop("obs")
    fig = plot_cf_distribution(obs, cf_series)
    # Two main panels (hist, ecdf); inset lives as a child of the hist axis
    assert len(fig.axes) == 2
    ax_hist = fig.axes[0]
    assert len(ax_hist.child_axes) == 1
    legend_texts = [t.get_text() for t in ax_hist.get_legend().get_texts()]
    assert any("uncorrected" in t for t in legend_texts)
    assert any("linear" in t and "KS=" in t for t in legend_texts)


def test_plot_cf_distribution_no_inset(cf_series):
    obs = cf_series.pop("obs")
    fig = plot_cf_distribution(obs, cf_series, tail_inset=False)
    assert len(fig.axes) == 2
    assert fig.axes[0].child_axes == []


def test_plot_cf_distribution_one_sim_only(cf_series):
    obs = cf_series.pop("obs")
    only_unc = {"uncorrected": cf_series["uncorrected"]}
    fig = plot_cf_distribution(obs, only_unc, tail_inset=False)
    assert len(fig.axes) == 2


# ---------------------------------------------------------------------------
# plot_qq
# ---------------------------------------------------------------------------

def test_plot_qq_returns_figure(cf_series):
    obs = cf_series.pop("obs")
    fig = plot_qq(obs, cf_series)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    # Diagonal + one line per sim
    assert len(ax.get_lines()) == 1 + len(cf_series)
    assert "quantile" in ax.get_xlabel().lower()
    assert "quantile" in ax.get_ylabel().lower()


def test_plot_qq_linear_closer_to_diagonal_than_uncorrected(cf_series):
    """Sanity check that the synthetic fixture has the bias structure we expect:
    the linear correction's QQ curve should sit closer to y=x than uncorrected."""
    obs = cf_series["obs"]
    q = np.linspace(0.005, 0.995, 199)
    obs_q = np.quantile(obs, q)
    unc_dev = np.abs(np.quantile(cf_series["uncorrected"], q) - obs_q).mean()
    lin_dev = np.abs(np.quantile(cf_series["linear"], q) - obs_q).mean()
    assert lin_dev < unc_dev
