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
    plot_correction_factor_map,
    plot_error_vs_clusters,
    plot_factor_joint,
    plot_qq,
    plot_sim_vs_obs,
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
    pd.DataFrame({
        "ID": ids[:2],
        "capacity": [1000.0, 500.0],
        "lat": [55.0, 55.5],
        "lon": [8.0, 8.5],
    }).to_csv(train_dir / f"{country}_train_turb_info.csv", index=False)

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

    # Both fleets load: simulated-year and the training fleet the factors
    # were fitted on (needed by plot_correction_factor_map)
    assert len(res.turb_info) == 3
    assert len(res.train_turb_info) == 2


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


# ---------------------------------------------------------------------------
# plot_correction_factor_map
# ---------------------------------------------------------------------------

@pytest.fixture
def turbine_fleet(rng):
    """Synthetic fleet: 60 turbines scattered over a ~2°x2° box."""
    n = 60
    return pd.DataFrame({
        "ID": [f"T{i:03d}" for i in range(n)],
        "lat": rng.uniform(55.0, 57.0, n),
        "lon": rng.uniform(8.0, 10.0, n),
        "capacity": rng.uniform(0.5, 3.0, n),
    })


@pytest.fixture
def fixed_factors(rng):
    """Fixed-resolution factors table: one row per cluster."""
    n_clu = 5
    return pd.DataFrame({
        "cluster": np.arange(n_clu),
        "fixed": ["1/1"] * n_clu,
        "scalar": rng.uniform(0.85, 1.15, n_clu),
        "offset": rng.uniform(-0.5, 0.5, n_clu),
    })


def test_plot_correction_factor_map_returns_figure(fixed_factors, turbine_fleet):
    fig = plot_correction_factor_map(fixed_factors, turbine_fleet)
    # One map axes per factor column, plus one colorbar axes each
    assert len(fig.axes) == 4
    titles = {ax.get_title() for ax in fig.axes}
    assert {"Scalar", "Offset"} <= titles


def test_plot_correction_factor_map_single_column(fixed_factors, turbine_fleet):
    fig = plot_correction_factor_map(
        fixed_factors, turbine_fleet, columns=("scalar",)
    )
    assert len(fig.axes) == 2  # map + colorbar


def test_plot_correction_factor_map_with_boundary(fixed_factors, turbine_fleet):
    from shapely.geometry import box

    boundary = box(8.2, 55.2, 9.8, 56.8)
    fig = plot_correction_factor_map(
        fixed_factors, turbine_fleet, boundary=boundary
    )
    assert len(fig.axes) == 4


def test_plot_correction_factor_map_seasonal_period(rng, turbine_fleet):
    n_clu = 4
    seasons = ["winter", "spring", "summer", "autumn"]
    factors = pd.DataFrame({
        "cluster": np.repeat(np.arange(n_clu), len(seasons)),
        "season": seasons * n_clu,
        "scalar": rng.uniform(0.85, 1.15, n_clu * len(seasons)),
        "offset": rng.uniform(-0.5, 0.5, n_clu * len(seasons)),
    })
    # A specific period slice and the cross-period average both plot
    fig = plot_correction_factor_map(factors, turbine_fleet, period="winter")
    assert "winter" in fig.get_suptitle()
    fig2 = plot_correction_factor_map(factors, turbine_fleet)
    assert len(fig2.axes) == 4

    with pytest.raises(ValueError, match="not found"):
        plot_correction_factor_map(factors, turbine_fleet, period="noon")


def test_plot_correction_factor_map_one_sided_factors(turbine_fleet):
    """TwoSlopeNorm needs vmin < center < vmax; all-above-neutral factors
    must not crash the diverging scale."""
    factors = pd.DataFrame({
        "cluster": np.arange(3),
        "scalar": [1.05, 1.10, 1.20],
        "offset": [0.1, 0.2, 0.3],
    })
    fig = plot_correction_factor_map(factors, turbine_fleet)
    assert len(fig.axes) == 4


def test_plot_correction_factor_map_too_few_turbines(fixed_factors):
    tiny = pd.DataFrame({"lat": [55.0, 55.1], "lon": [8.0, 8.1]})
    with pytest.raises(ValueError, match="training fleet"):
        plot_correction_factor_map(fixed_factors, tiny)


def test_plot_correction_factor_map_missing_columns(turbine_fleet):
    with pytest.raises(ValueError, match="missing columns"):
        plot_correction_factor_map(
            pd.DataFrame({"cluster": [0], "scalar": [1.0]}), turbine_fleet
        )


# ---------------------------------------------------------------------------
# plot_error_vs_clusters
# ---------------------------------------------------------------------------

@pytest.fixture
def metrics_table(rng):
    """Synthetic evaluator output: error falls with clusters, then flattens."""
    rows = []
    for tres, base in [("fixed", 0.16), ("season", 0.15),
                       ("bimonth", 0.145), ("month", 0.14)]:
        for n in [1, 10, 100, 1000]:
            err = base / (1 + 0.3 * np.log10(max(n, 1) + 1))
            rows.append({
                "country": "DK",
                "correction_type": "corrected",
                "time_res": tres,
                "n_clusters": n,
                "rmse": err,
                "mae": err * 0.8,
            })
    rows.append({
        "country": "DK",
        "correction_type": "uncorrected",
        "time_res": None,
        "n_clusters": None,
        "rmse": 0.2,
        "mae": 0.16,
    })
    return pd.DataFrame(rows)


def test_plot_error_vs_clusters_returns_figure(metrics_table):
    fig = plot_error_vs_clusters(metrics_table)
    assert len(fig.axes) == 2
    ax_rmse, ax_mae = fig.axes
    assert ax_rmse.get_ylabel() == "RMSE"
    assert ax_mae.get_ylabel() == "MAE"
    assert ax_rmse.get_xscale() == "log"
    # 4 temporal resolutions + 1 uncorrected baseline per panel
    legend_labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert legend_labels == [
        "Fixed", "Seasonal", "Bimonthly", "Monthly", "Uncorrected"
    ]


def test_plot_error_vs_clusters_no_baseline(metrics_table):
    corrected_only = metrics_table[
        metrics_table["correction_type"] == "corrected"
    ]
    fig = plot_error_vs_clusters(corrected_only)
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "Uncorrected" not in labels

    fig2 = plot_error_vs_clusters(metrics_table, show_uncorrected=False)
    labels2 = [t.get_text() for t in fig2.legends[0].get_texts()]
    assert "Uncorrected" not in labels2


def test_plot_error_vs_clusters_accepts_n_clu_column(metrics_table):
    renamed = metrics_table.rename(columns={"n_clusters": "n_clu"})
    fig = plot_error_vs_clusters(renamed)
    assert len(fig.axes) == 2


def test_plot_error_vs_clusters_single_metric(metrics_table):
    fig = plot_error_vs_clusters(metrics_table, metric_cols=("rmse",))
    assert len(fig.axes) == 1


def test_plot_error_vs_clusters_requires_corrected_rows(metrics_table):
    baseline_only = metrics_table[
        metrics_table["correction_type"] == "uncorrected"
    ]
    with pytest.raises(ValueError, match="no corrected rows"):
        plot_error_vs_clusters(baseline_only)


def test_plot_error_vs_clusters_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        plot_error_vs_clusters(pd.DataFrame({"time_res": ["fixed"]}))


# ---------------------------------------------------------------------------
# plot_factor_joint
# ---------------------------------------------------------------------------

def test_plot_factor_joint_returns_figure(fixed_factors):
    fig = plot_factor_joint(fixed_factors)
    # Main scatter + two marginal histogram axes
    assert len(fig.axes) == 3
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Scalar"
    assert ax.get_ylabel() == "Offset"
    # Main axis keeps its ticks despite the marginals sharing axes
    assert len(ax.get_xticks()) > 0
    assert len(ax.get_yticks()) > 0


def test_plot_factor_joint_seasonal_period(rng):
    n_clu = 6
    seasons = ["winter", "spring", "summer", "autumn"]
    factors = pd.DataFrame({
        "cluster": np.repeat(np.arange(n_clu), len(seasons)),
        "season": seasons * n_clu,
        "scalar": rng.uniform(0.85, 1.15, n_clu * len(seasons)),
        "offset": rng.uniform(-0.5, 0.5, n_clu * len(seasons)),
    })
    fig = plot_factor_joint(factors, period="summer")
    assert "summer" in fig.get_suptitle()
    # One point per cluster when a single period is selected
    assert len(fig.axes[0].collections[0].get_offsets()) == n_clu

    # Pooled: one point per (cluster, period) pair
    fig2 = plot_factor_joint(factors)
    assert len(fig2.axes[0].collections[0].get_offsets()) == n_clu * len(seasons)

    with pytest.raises(ValueError, match="not found"):
        plot_factor_joint(factors, period="noon")


def test_plot_factor_joint_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        plot_factor_joint(pd.DataFrame({"cluster": [0], "scalar": [1.0]}))


# ---------------------------------------------------------------------------
# plot_sim_vs_obs
# ---------------------------------------------------------------------------

@pytest.fixture
def wide_cf_pair(rng):
    """Sim/obs wide CF frames in the on-disk schema, sim biased high."""
    ids = [f"T{i:03d}" for i in range(30)]
    times = pd.date_range("2020-01-01", periods=100, freq="D")
    obs = pd.DataFrame(
        np.clip(rng.normal(0.30, 0.10, (100, 30)), 0, 1), columns=ids
    )
    sim = pd.DataFrame(
        np.clip(obs.to_numpy() + rng.normal(0.05, 0.02, (100, 30)), 0, 1),
        columns=ids,
    )
    obs.insert(0, "time", times)
    sim.insert(0, "time", times)
    return sim, obs, ids


def test_plot_sim_vs_obs_returns_figure(wide_cf_pair):
    sim, obs, ids = wide_cf_pair
    fig = plot_sim_vs_obs(sim, obs)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    # One scatter collection + the y=x diagonal
    assert len(ax.collections) == 1
    assert len(ax.collections[0].get_offsets()) == len(ids)
    assert len(ax.get_lines()) == 1
    # Biased-high fixture shows up in the MBE annotation
    annotation = ax.texts[0].get_text()
    assert "MBE = +0" in annotation
    assert f"n = {len(ids)}" in annotation


def test_plot_sim_vs_obs_coloured_by_type(wide_cf_pair):
    sim, obs, ids = wide_cf_pair
    turb_info = pd.DataFrame({
        "ID": ids,
        "type": ["onshore"] * 20 + ["offshore"] * 10,
    })
    fig = plot_sim_vs_obs(sim, obs, turb_info=turb_info)
    ax = fig.axes[0]
    assert len(ax.collections) == 2  # one scatter per type
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert set(labels) == {"Onshore", "Offshore"}


def test_plot_sim_vs_obs_intersects_ids(wide_cf_pair):
    sim, obs, ids = wide_cf_pair
    fig = plot_sim_vs_obs(sim.drop(columns=ids[:5]), obs)
    assert len(fig.axes[0].collections[0].get_offsets()) == len(ids) - 5


def test_plot_sim_vs_obs_accepts_series(wide_cf_pair):
    sim, obs, ids = wide_cf_pair
    sim_means = sim.drop(columns=["time"]).mean()
    obs_means = obs.drop(columns=["time"]).mean()
    fig = plot_sim_vs_obs(sim_means, obs_means)
    assert len(fig.axes[0].collections[0].get_offsets()) == len(ids)


def test_plot_sim_vs_obs_no_common_ids_raises(wide_cf_pair):
    sim, obs, _ = wide_cf_pair
    renamed = obs.rename(columns=lambda c: f"X{c}" if c != "time" else c)
    with pytest.raises(ValueError, match="no turbine IDs"):
        plot_sim_vs_obs(sim, renamed)
