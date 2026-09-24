"""Distributional diagnostic plots for PyVWF capacity-factor outputs.

Two figures, designed as a pair, that show how well a corrected simulation
reproduces the observed capacity-factor distribution: an overlay of
histograms and ECDFs (with an optional upper-tail inset), and a quantile-
quantile plot against the y=x diagonal.

The :func:`load_results` helper reads a harness evaluate run into a single
:class:`Results` object, paired and scored as its ``metrics.csv`` was, so the
plot functions are self-contained: no path-juggling at the call site.

Free functions; pass anything dict-shaped (``{label: series}``) and they
will plot it. Returns ``matplotlib.figure.Figure`` so the caller decides
what to do next (``fig.savefig(...)``, further tweaks, etc.).
"""

from __future__ import annotations

import json
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
    """Capacity-factor series and supporting metadata for one evaluate run.

    Series are monthly fleet (or national) aggregates, capacity-weighted by
    default, over exactly the unit-months the run's ``metrics.csv`` scored, so
    they can be compared distributionally and read against its metrics.

    Attributes:
        country: The region code (e.g. ``"DK"``).
        year: The test year.
        obs: Observed CF, indexed by month.
        uncorrected: Uncorrected simulated CF, indexed by month.
        corrected: Corrected CF, one entry per ``(n_clu, time_res)``.
        factors: Linear correction factor tables ``(scalar, offset)`` keyed by
            ``(n_clu, time_res)``.
        turb_info: Fleet metadata for the simulated year.
        train_turb_info: Fleet metadata for the training period: the fleet
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


_COR_NAME = re.compile(r"^cor_cf_(?P<tr>[a-z]+)_(?P<n>\d+)\.csv$")
_FACTORS_NAME = re.compile(r"^factors_(?P<tr>[a-z]+)_(?P<n>\d+)\.csv$")


def _monthly(paired: pd.DataFrame, column: str, weight_by_capacity: bool) -> pd.Series:
    """One value per month from a paired frame, as a month-start time series."""
    if "ym" in paired.columns:  # national pairs are already monthly
        out = paired.set_index("ym")[column]
        out.index = pd.PeriodIndex(out.index).to_timestamp()
        return out.sort_index()
    frame = paired.assign(
        time=pd.to_datetime(dict(year=paired["year"], month=paired["month"], day=1))
    )
    if weight_by_capacity and "capacity" in frame.columns:
        w = frame["capacity"].astype(float)
        num = (frame[column] * w).groupby(frame["time"]).sum()
        return (num / w.groupby(frame["time"]).sum()).sort_index()
    return frame.groupby("time")[column].mean().sort_index()


def load_results(
    region,
    evaluate_run: str | Path,
    *,
    train_run: str | Path | None = None,
    source=None,
    weight_by_capacity: bool = True,
) -> Results:
    """Load a harness evaluate run as monthly series, paired as its metrics were.

    The evaluate run holds the simulated frames (``unc_cf.csv`` and one
    ``cor_cf_<time_res>_<n>.csv`` per variant) but not the observations, so
    these are read again through the region's adapter, as ``run_evaluate`` read
    them. Each variant is then paired with the observations by the harness's own
    functions and restricted to the rows every variant can score, the rows its
    ``metrics.csv`` was computed on, before it is reduced to one value per month.
    Reading the observations needs the region's input data.

    The monthly series weight each month equally. Where every unit reports
    every month, their mean difference is the metrics' MBE exactly; where some
    unit-months are missing (NZ scores 137 of 144), it differs slightly, since
    the metrics weight each unit-month.

    Args:
        region: The region config, a :class:`~vwf.harness.regions.RegionSpec` or
            a path to its TOML file.
        evaluate_run: The evaluate run directory.
        train_run: The training run the factors came from. Defaults to the
            evaluate manifest's ``trained_from``; when that path no longer
            exists, ``factors`` and ``train_turb_info`` are left empty.
        source: An observation adapter to use instead of the region's own, as
            for :func:`vwf.harness.driver.load_obs_and_fleet`.
        weight_by_capacity: Aggregate units capacity-weighted (default), as the
            metrics are, or as a plain mean. National series are unaffected.

    Returns:
        :class:`Results` with monthly ``obs``, ``uncorrected`` and one
        ``corrected`` series per variant, the factors, the test fleet and the
        training fleet.

    Raises:
        FileNotFoundError: If the evaluate run, its ``unc_cf.csv``, or an
            explicitly given ``train_run`` does not exist.
    """
    from vwf.harness.driver import SCOPE_KEYS, country_pairs, load_obs_and_fleet, tidy_eval_frame
    from vwf.harness.regions import RegionSpec, load_region
    from vwf.harness.skill import collapse_pseudo_replicates, restrict_to_common_rows

    spec = region if isinstance(region, RegionSpec) else load_region(region)
    run = Path(evaluate_run)
    if not run.is_dir():
        raise FileNotFoundError(f"Evaluate run not found: {run}")
    unc_path = run / "unc_cf.csv"
    if not unc_path.is_file():
        raise FileNotFoundError(f"Missing uncorrected CF file: {unc_path}")
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    year = int(manifest.get("evaluation_year", spec.test_years[0]))

    frames: dict[str, pd.DataFrame] = {"uncorrected": pd.read_csv(unc_path)}
    variants: dict[str, tuple[int, str]] = {}
    for p in sorted(run.iterdir()):
        m = _COR_NAME.match(p.name)
        if m:
            label = f"{m['tr']}_{m['n']}"
            frames[label] = pd.read_csv(p)
            variants[label] = (int(m["n"]), m["tr"])

    obs_cf, turb_info = load_obs_and_fleet(spec, year, source)
    if spec.obs_level == "country":
        paired = {k: country_pairs(f, obs_cf, turb_info) for k, f in frames.items()}
        keys, weight, _ = SCOPE_KEYS["national"]
    else:
        paired = {
            k: collapse_pseudo_replicates(tidy_eval_frame(f, obs_cf, turb_info), spec)
            for k, f in frames.items()
        }
        keys, weight, _ = SCOPE_KEYS["fleet"]
    common, _ = restrict_to_common_rows(paired, keys, weight=weight)

    obs = _monthly(common["uncorrected"], "cf_obs", weight_by_capacity).rename("obs")
    unc = _monthly(common["uncorrected"], "cf_sim", weight_by_capacity).rename("uncorrected")
    corrected = {
        key: _monthly(common[label], "cf_sim", weight_by_capacity).rename(label)
        for label, key in variants.items()
    }

    # A manifest without ``trained_from`` must not fall back to Path(""),
    # which is the working directory and exists.
    train_dir: Path | None
    if train_run is not None:
        train_dir = Path(train_run)
        if not train_dir.is_dir():
            raise FileNotFoundError(f"Training run not found: {train_dir}")
    else:
        recorded = manifest.get("trained_from")
        train_dir = Path(recorded) if recorded else None
    factors: dict[tuple[int, str], pd.DataFrame] = {}
    train_turb_info = None
    if train_dir is not None and train_dir.is_dir():
        for p in sorted(train_dir.iterdir()):
            m = _FACTORS_NAME.match(p.name)
            if m:
                factors[(int(m["n"]), m["tr"])] = pd.read_csv(p)
        fleets = sorted(train_dir.glob("train_turb_info_*.csv"))
        if fleets:
            # One file per cluster count, identical apart from ``cluster``;
            # the map re-clusters the units itself.
            train_turb_info = pd.read_csv(fleets[0]).drop(columns=["cluster"], errors="ignore")

    return Results(
        country=spec.code,
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
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.1},
        layout="constrained",
    )

    bin_edges = np.linspace(0.0, 1.0, bins + 1)

    # Observed series: filled grey histogram (reference) + black step CDF
    ax_hist.hist(
        obs_arr,
        bins=bin_edges,
        density=True,
        histtype="stepfilled",
        color="0.75",
        edgecolor="black",
        linewidth=1.2,
        label=f"obs  (μ={obs_arr.mean():.3f})",
    )
    _plot_ecdf(ax_cdf, obs_arr, color="black", label="obs", linewidth=1.4)

    inset = None
    if tail_inset:
        inset = ax_hist.inset_axes([0.55, 0.45, 0.42, 0.5])
        tail_edges = np.linspace(tail_threshold, 1.0, max(10, bins // 4))
        inset.hist(
            obs_arr[obs_arr >= tail_threshold],
            bins=tail_edges,
            density=True,
            histtype="stepfilled",
            color="0.75",
            edgecolor="black",
            linewidth=1.0,
        )

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cycle_iter = iter(c for c in prop_cycle if c not in _DEFAULT_COLOURS.values())

    for label, series in sims.items():
        arr = _to_array(series)
        if arr.size == 0:
            continue
        colour = _colour_for(label, next(cycle_iter, None) or "C0")
        ax_hist.hist(
            arr,
            bins=bin_edges,
            density=True,
            histtype="step",
            color=colour,
            linewidth=1.4,
            label=_legend_label(label, arr, obs_arr),
        )
        _plot_ecdf(ax_cdf, arr, color=colour, label=label, linewidth=1.2)
        if inset is not None:
            tail = arr[arr >= tail_threshold]
            if tail.size:
                inset.hist(
                    tail,
                    bins=tail_edges,
                    density=True,
                    histtype="step",
                    color=colour,
                    linewidth=1.2,
                )

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
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--", label="y = x")

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
