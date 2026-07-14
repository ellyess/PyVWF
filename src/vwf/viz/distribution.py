"""Distributional diagnostic plots for PyVWF capacity-factor outputs.

Two figures, designed as a pair, that show how well a corrected simulation
reproduces the observed capacity-factor distribution: an overlay of
histograms and ECDFs (with an optional upper-tail inset), and a quantile-
quantile plot against the y=x diagonal.

The :func:`load_results` helper reads the on-disk schema produced by
:meth:`vwf.vwf.PyVWF.simulate_cf` into a single :class:`Results` object so
the plot functions are self-contained — no path-juggling at the call site.

Free functions; pass anything dict-shaped (``{label: series}``) and they
will plot it. Returns ``matplotlib.figure.Figure`` so the caller decides
what to do next (``fig.savefig(...)``, further tweaks, etc.).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import stats


__all__ = ["Results", "load_results", "plot_cf_distribution", "plot_qq"]


def _ks_distance(sim: np.ndarray, obs: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov distance (max CDF gap) over finite values."""
    s = sim[np.isfinite(sim)]
    o = obs[np.isfinite(obs)]
    if s.size == 0 or o.size == 0:
        return float("nan")
    return float(stats.ks_2samp(s, o).statistic)


# Stable colour mapping so plot_cf_distribution and plot_qq read as a pair.
# Obs is always black; sims pick up tab10 colours by canonical label, with
# anything unrecognised falling through to the matplotlib prop cycle.
_DEFAULT_COLOURS = {
    "obs": "black",
    "observed": "black",
    "uncorrected": "#1f77b4",
    "linear": "#ff7f0e",
}


# ---------------------------------------------------------------------------
# Results loader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Results:
    """Capacity-factor series and supporting metadata for one run × year.

    Series are country-aggregated (capacity-weighted by default) so they
    can be compared distributionally without first reducing across turbines.

    Attributes:
        country: Country code (e.g. ``"DK"``).
        year: Simulated year.
        obs: Observed CF, indexed by time.
        uncorrected: Uncorrected simulated CF, indexed by time.
        corrected: Linear-corrected CF, one entry per ``(n_clu, time_res)``.
        factors: Linear correction factor tables ``(scalar, offset)`` keyed by
            ``(n_clu, time_res)``.
        turb_info: Fleet metadata for the simulated year.
        train_turb_info: Fleet metadata for the training period — the fleet
            the correction factors were fitted on, which
            :func:`vwf.viz.plot_correction_factor_map` needs to reproduce
            the cluster IDs.
    """

    country: str
    year: int
    obs: pd.Series
    uncorrected: pd.Series
    corrected: dict[tuple[int, str], pd.Series] = field(default_factory=dict)
    factors: dict[tuple[int, str], pd.DataFrame] = field(default_factory=dict)
    turb_info: pd.DataFrame | None = None
    train_turb_info: pd.DataFrame | None = None


def _aggregate_cf(
    wide: pd.DataFrame,
    turb_info: pd.DataFrame | None,
    weight_by_capacity: bool,
) -> pd.Series:
    """Reduce a wide per-turbine CF frame to a single country series.

    Uses capacity-weighted average when ``weight_by_capacity`` and a
    turb_info with ``capacity`` is available; otherwise a plain row mean.
    """
    df = wide.set_index("time") if "time" in wide.columns else wide.copy()
    df.index = pd.to_datetime(df.index)

    if not weight_by_capacity or turb_info is None or "capacity" not in turb_info.columns:
        return df.mean(axis=1)

    caps = turb_info.set_index(turb_info["ID"].astype(str))["capacity"]
    cols = df.columns.astype(str)
    weights = caps.reindex(cols)
    # Drop columns with no matching capacity rather than silently dropping them
    # to NaN-weighted zero; this preserves the unweighted fallback semantics
    # if turb_info doesn't cover every ID in the CF file.
    valid = weights.notna()
    if not valid.any():
        return df.mean(axis=1)
    df = df.loc[:, valid.values]
    w = weights[valid].to_numpy(dtype=float)
    arr = df.to_numpy(dtype=float)
    return pd.Series(np.nansum(arr * w, axis=1) / w.sum(), index=df.index)


_COR_RE = re.compile(r"^(?P<country>[A-Z]+)_(?P<year>\d{4})_(?P<tr>[a-z]+)_(?P<n>\d+)_cor_cf\.csv$")
_FACT_RE = re.compile(r"^(?P<country>[A-Z]+)_factors_(?P<tr>[a-z]+)_(?P<n>\d+)\.csv$")


def _align_to_obs_cadence(series: pd.Series, obs_index: pd.Index) -> pd.Series:
    """Aggregate ``series`` into the windows defined by ``obs_index``.

    For turbine-level PyVWF runs, observations are stored monthly while the
    simulated CFs are daily, so the two series can't be compared
    distributionally as-written: 12 vs 366 samples is not apples-to-apples.
    This helper takes each obs timestamp ``t_i`` as the left edge of a
    half-open window ``[t_i, t_{i+1})`` and returns the mean of ``series``
    inside each window, indexed by ``obs_index``. If ``series`` is already at
    or coarser than obs (≤ obs count), it is returned unchanged.
    """
    if len(series) <= len(obs_index):
        return series
    obs_times = pd.DatetimeIndex(obs_index)
    last_step = (
        obs_times[-1] - obs_times[-2] if len(obs_times) >= 2 else pd.Timedelta(days=1)
    )
    edges = obs_times.append(pd.DatetimeIndex([obs_times[-1] + last_step]))
    out = []
    for i in range(len(obs_times)):
        mask = (series.index >= edges[i]) & (series.index < edges[i + 1])
        out.append(series.loc[mask].mean())
    return pd.Series(out, index=obs_times, name=series.name)


def load_results(
    run_dir: str | Path,
    country: str,
    year: int,
    *,
    weight_by_capacity: bool = True,
    align_to_obs: bool = True,
) -> Results:
    """Load PyVWF capacity-factor outputs for one country and test year.

    Args:
        run_dir: Path to a single ``PyVWF`` run directory (the value passed
            as the model's first positional argument, containing the
            ``results/`` and ``training/`` subfolders).
        country: Country code matching the on-disk filename prefix.
        year: Simulated year matching the on-disk filename prefix.
        weight_by_capacity: If True (default), aggregate per-turbine CF
            columns into a country series using ``capacity`` from
            ``turb_info``. If False, use an unweighted mean.
        align_to_obs: If True (default), resample sim series with finer
            cadence than obs into obs-defined windows so distributional
            comparisons are well-posed. Turbine-level runs store obs at
            monthly and sims at daily; without alignment the distributions
            are not comparable. Set False to keep on-disk cadences as-is.

    Returns:
        :class:`Results` with country-aggregated ``obs``, ``uncorrected``,
        and any linear-corrected variants discovered.

    Raises:
        FileNotFoundError: If the run directory or the obs/uncorrected CF
            files for ``(country, year)`` cannot be found.
    """
    run = Path(run_dir)
    if not run.exists():
        raise FileNotFoundError(f"Run directory not found: {run}")

    cf_dir = run / "results" / "capacity-factor"
    train_dir = run / "training" / "simulated-turbines"
    fac_dir = run / "training" / "correction-factors"

    obs_path = cf_dir / f"{country}_{year}_obs_cf.csv"
    unc_path = cf_dir / f"{country}_{year}_unc_cf.csv"
    if not obs_path.is_file():
        raise FileNotFoundError(f"Missing observed CF file: {obs_path}")
    if not unc_path.is_file():
        raise FileNotFoundError(f"Missing uncorrected CF file: {unc_path}")

    turb_path = train_dir / f"{country}_{year}_turb_info.csv"
    turb_info = pd.read_csv(turb_path) if turb_path.is_file() else None

    train_turb_path = train_dir / f"{country}_train_turb_info.csv"
    train_turb_info = (
        pd.read_csv(train_turb_path) if train_turb_path.is_file() else None
    )

    obs_wide = pd.read_csv(obs_path)
    unc_wide = pd.read_csv(unc_path)
    obs = _aggregate_cf(obs_wide, turb_info, weight_by_capacity).rename("obs")
    unc = _aggregate_cf(unc_wide, turb_info, weight_by_capacity).rename("uncorrected")
    if align_to_obs:
        unc = _align_to_obs_cadence(unc, obs.index)

    def _read_and_align(path):
        s = _aggregate_cf(pd.read_csv(path), turb_info, weight_by_capacity)
        return _align_to_obs_cadence(s, obs.index) if align_to_obs else s

    corrected: dict[tuple[int, str], pd.Series] = {}
    if cf_dir.is_dir():
        for p in sorted(cf_dir.iterdir()):
            m = _COR_RE.match(p.name)
            if m and m["country"] == country and int(m["year"]) == year:
                corrected[(int(m["n"]), m["tr"])] = _read_and_align(p)

    factors: dict[tuple[int, str], pd.DataFrame] = {}
    if fac_dir.is_dir():
        for p in sorted(fac_dir.iterdir()):
            m = _FACT_RE.match(p.name)
            if m and m["country"] == country:
                factors[(int(m["n"]), m["tr"])] = pd.read_csv(p)

    return Results(
        country=country,
        year=year,
        obs=obs,
        uncorrected=unc,
        corrected=corrected,
        factors=factors,
        turb_info=turb_info,
        train_turb_info=train_turb_info,
    )


# ---------------------------------------------------------------------------
# Helpers shared by the two plot functions
# ---------------------------------------------------------------------------

def _to_array(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


def _colour_for(label: str, fallback) -> str:
    return _DEFAULT_COLOURS.get(label.lower(), fallback)


def _legend_label(label: str, sim: np.ndarray, obs: np.ndarray) -> str:
    mu = float(np.mean(sim)) if sim.size else float("nan")
    ks = _ks_distance(sim, obs)
    return f"{label}  (μ={mu:.3f}, KS={ks:.3f})"


# ---------------------------------------------------------------------------
# Plot 1: CF distribution (hist + ECDF + tail inset)
# ---------------------------------------------------------------------------

def plot_cf_distribution(
    obs,
    sims: Mapping[str, "pd.Series | np.ndarray"],
    *,
    bins: int | None = None,
    tail_inset: bool = True,
    tail_threshold: float = 0.7,
    figsize: tuple[float, float] = (7.0, 5.5),
) -> Figure:
    """Overlay capacity-factor histograms and ECDFs for obs and each sim.

    The legend annotates each sim with its mean and KS distance to obs, and
    the optional tail inset (top panel, log-y) zooms into ``CF >= tail_threshold``,
    where distributional differences between sims often live.

    Args:
        obs: Observed CF (pandas Series or array-like).
        sims: Mapping of label -> simulated/corrected CF. Use canonical
            labels (``"uncorrected"``, ``"linear"``) to pick up the stable
            colour scheme; any others fall through to the matplotlib prop
            cycle.
        bins: Histogram bin count. If ``None`` (default), scales with the
            smallest series so monthly data (~12 points) doesn't render as
            single-sample spikes while hourly data (~8760 points) still
            gets the full 50 bins.
        tail_inset: Draw the upper-tail inset on the histogram panel.
        tail_threshold: Lower edge of the inset (default ``0.7``).
        figsize: Figure size in inches.

    Returns:
        The constructed ``matplotlib.figure.Figure``.
    """
    obs_arr = _to_array(obs)
    if bins is None:
        n_min = min([obs_arr.size] + [_to_array(s).size for s in sims.values()] or [1])
        bins = max(5, min(50, n_min // 5))

    fig, (ax_hist, ax_cdf) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.1},
        layout="constrained",
    )

    bin_edges = np.linspace(0.0, 1.0, bins + 1)

    # Observed series: filled grey histogram (reference) + black step CDF
    ax_hist.hist(
        obs_arr, bins=bin_edges, density=True,
        histtype="stepfilled", color="0.75", edgecolor="black",
        linewidth=1.2, label=f"obs  (μ={obs_arr.mean():.3f})",
    )
    _plot_ecdf(ax_cdf, obs_arr, color="black", label="obs", linewidth=1.4)

    inset = None
    if tail_inset:
        inset = ax_hist.inset_axes([0.55, 0.45, 0.42, 0.5])
        tail_edges = np.linspace(tail_threshold, 1.0, max(10, bins // 4))
        inset.hist(
            obs_arr[obs_arr >= tail_threshold], bins=tail_edges, density=True,
            histtype="stepfilled", color="0.75", edgecolor="black", linewidth=1.0,
        )

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cycle_iter = iter(c for c in prop_cycle if c not in _DEFAULT_COLOURS.values())

    for label, series in sims.items():
        arr = _to_array(series)
        if arr.size == 0:
            continue
        colour = _colour_for(label, next(cycle_iter, None) or "C0")
        ax_hist.hist(
            arr, bins=bin_edges, density=True,
            histtype="step", color=colour, linewidth=1.4,
            label=_legend_label(label, arr, obs_arr),
        )
        _plot_ecdf(ax_cdf, arr, color=colour, label=label, linewidth=1.2)
        if inset is not None:
            tail = arr[arr >= tail_threshold]
            if tail.size:
                inset.hist(tail, bins=tail_edges, density=True,
                           histtype="step", color=colour, linewidth=1.2)

    if inset is not None:
        inset.set_yscale("log")
        inset.set_xlim(tail_threshold, 1.0)
        inset.set_title(f"upper tail (CF ≥ {tail_threshold:.2f})", fontsize=8)
        inset.tick_params(labelsize=7)
        inset.set_xlabel("")

    ax_hist.set_ylabel("density")
    ax_hist.legend(loc="upper left", frameon=False, fontsize=8)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    ax_cdf.set_xlabel("capacity factor")
    ax_cdf.set_ylabel("ECDF")
    ax_cdf.set_xlim(0.0, 1.0)
    ax_cdf.set_ylim(0.0, 1.0)
    ax_cdf.spines["top"].set_visible(False)
    ax_cdf.spines["right"].set_visible(False)

    fig.suptitle("Capacity-factor distribution: obs vs simulated", fontsize=10)
    return fig


def _plot_ecdf(ax, arr: np.ndarray, *, color, label, linewidth=1.2) -> None:
    if arr.size == 0:
        return
    xs = np.sort(arr)
    ys = np.arange(1, xs.size + 1) / xs.size
    ax.plot(xs, ys, color=color, label=label, linewidth=linewidth)


# ---------------------------------------------------------------------------
# Plot 2: QQ
# ---------------------------------------------------------------------------

def plot_qq(
    obs,
    sims: Mapping[str, "pd.Series | np.ndarray"],
    *,
    quantiles: np.ndarray | None = None,
    figsize: tuple[float, float] = (5.5, 5.5),
) -> Figure:
    """Quantile-quantile plot of each sim against the observed distribution.

    A series whose distribution matches obs lies on the y=x diagonal;
    deviations show where (and how strongly) the sim's distribution differs
    from observed in the body and tails.

    Args:
        obs: Observed CF (pandas Series or array-like).
        sims: Mapping of label -> simulated/corrected CF.
        quantiles: Quantile grid to evaluate (default ``linspace(0.005, 0.995, 199)``).
        figsize: Figure size in inches.

    Returns:
        The constructed ``matplotlib.figure.Figure``.
    """
    if quantiles is None:
        quantiles = np.linspace(0.005, 0.995, 199)

    obs_arr = _to_array(obs)
    obs_q = np.quantile(obs_arr, quantiles)

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    lo = float(np.nanmin(obs_q))
    hi = float(np.nanmax(obs_q))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0,
            linestyle="--", label="y = x")

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cycle_iter = iter(c for c in prop_cycle if c not in _DEFAULT_COLOURS.values())

    for label, series in sims.items():
        arr = _to_array(series)
        if arr.size == 0:
            continue
        sim_q = np.quantile(arr, quantiles)
        colour = _colour_for(label, next(cycle_iter, None) or "C0")
        ax.plot(obs_q, sim_q, color=colour, linewidth=1.4, label=label)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("observed quantile (CF)")
    ax.set_ylabel("simulated quantile (CF)")
    ax.set_title("QQ: simulated vs observed capacity factor", fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig
