"""Evaluation diagnostics for PyVWF bias-correction runs.

Two figures for judging a run:

:func:`plot_error_vs_clusters` answers the model-selection question: *how
many clusters, and at what temporal resolution?* Error metrics are drawn
against cluster count (log x) with one line per temporal resolution, plus an
optional horizontal reference at the uncorrected error so the payoff of the
correction is visible at a glance. It takes a tidy metrics DataFrame, the
schema produced by ``scripts/analysis/evaluate_all_pyvwf_runs.py``
(``pyvwf_evaluation_metrics.csv``). Filtering to one country/run is the
caller's job; if several are mixed, lines will zig-zag.

:func:`plot_sim_vs_obs` answers the fit question: *where does the
simulation sit against observation, turbine by turbine?* Each turbine's mean
simulated CF is scattered against its mean observed CF; distance from the
y=x diagonal is that turbine's bias, annotated with fleet-level MBE/RMSE.

Free functions returning ``matplotlib.figure.Figure``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from vwf.viz.palettes import (
    TIME_RES_COLOURS,
    TIME_RES_LABELS,
    TIME_RES_LINESTYLES,
    TIME_RES_ORDER,
    TURBINE_TYPE_COLOURS,
)


__all__ = ["plot_error_vs_clusters", "plot_sim_vs_obs"]


def _ordered_time_res(values) -> list:
    """Sort temporal resolutions coarse -> fine, unknowns last."""
    return sorted(values, key=lambda x: TIME_RES_ORDER.get(x, 99))


def plot_error_vs_clusters(
    metrics: pd.DataFrame,
    *,
    metric_cols: Sequence[str] = ("rmse", "mae"),
    show_uncorrected: bool = True,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot error vs cluster count, one line per temporal resolution.

    Args:
        metrics: Tidy metrics table with one row per evaluated variant.
            Required columns: ``time_res``, a cluster-count column
            (``n_clusters`` or ``n_clu``), and each of ``metric_cols``.
            Rows with ``correction_type == "uncorrected"`` (or with a null
            ``time_res``) are treated as the uncorrected baseline rather
            than as a line. This is the schema written by
            ``scripts/analysis/evaluate_all_pyvwf_runs.py``.
        metric_cols: Error columns to plot, one panel each
            (default ``("rmse", "mae")``).
        show_uncorrected: Draw a horizontal reference line at the
            uncorrected error, when baseline rows are present.
        figsize: Figure size in inches. Defaults to
            ``(4.0 * len(metric_cols), 3.4)``.

    Returns:
        The constructed ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If required columns are missing or no corrected rows
            remain after filtering.
    """
    df = metrics.copy()
    if "n_clu" in df.columns and "n_clusters" not in df.columns:
        df = df.rename(columns={"n_clu": "n_clusters"})

    required = {"time_res", "n_clusters", *metric_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metrics table is missing columns: {sorted(missing)}")

    is_uncorrected = df["time_res"].isna() | (
        df["time_res"].astype(str).str.lower() == "uncorrected"
    )
    if "correction_type" in df.columns:
        is_uncorrected |= df["correction_type"] == "uncorrected"

    baseline = df[is_uncorrected]
    corrected = df[~is_uncorrected & df["n_clusters"].notna()].copy()
    if corrected.empty:
        raise ValueError("no corrected rows to plot in metrics table")
    corrected["n_clusters"] = corrected["n_clusters"].astype(int)

    if figsize is None:
        figsize = (4.0 * len(metric_cols), 3.4)
    fig, axes = plt.subplots(
        1, len(metric_cols), figsize=figsize, layout="constrained"
    )
    axes = np.atleast_1d(axes)

    time_res_vals = _ordered_time_res(corrected["time_res"].unique())

    for ax, metric in zip(axes, metric_cols):
        for tres in time_res_vals:
            subset = corrected[corrected["time_res"] == tres].sort_values("n_clusters")
            ax.plot(
                subset["n_clusters"],
                subset[metric],
                marker="o",
                markersize=3,
                linestyle=TIME_RES_LINESTYLES.get(tres, "-"),
                color=TIME_RES_COLOURS.get(tres, "#999999"),
                label=TIME_RES_LABELS.get(tres, tres),
                alpha=0.85,
            )

        if show_uncorrected and not baseline.empty:
            ref = float(baseline[metric].mean())
            if np.isfinite(ref):
                ax.axhline(
                    ref, color="grey", linewidth=1.0, linestyle="--",
                    label="Uncorrected", zorder=0,
                )

        ax.set_xscale("log")
        ax.set_xlabel(r"Number of clusters ($n_{\mathrm{clu}}$)")
        ax.set_ylabel(metric.upper() if metric in ("rmse", "mae", "mbe") else metric)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(labels),
        title=r"Temporal frequency ($t_{freq}$)",
        frameon=False,
    )
    return fig


def _per_turbine_mean(cf) -> pd.Series:
    """Reduce a wide CF frame (time x turbine IDs) to one mean per turbine.

    Accepts the on-disk schema (optional ``time`` column, one column per
    turbine ID) or an already-reduced per-turbine Series. IDs are coerced to
    str so sim and obs frames with mixed ID dtypes still align.
    """
    if isinstance(cf, pd.Series):
        out = cf.astype(float)
    else:
        df = cf.drop(columns=["time"], errors="ignore")
        out = df.mean(axis=0, numeric_only=True)
    out.index = out.index.astype(str)
    return out


def plot_sim_vs_obs(
    sim_cf,
    obs_cf,
    *,
    turb_info: pd.DataFrame | None = None,
    figsize: tuple[float, float] = (4.8, 4.8),
) -> Figure:
    """Scatter each turbine's mean simulated CF against its mean observed CF.

    A turbine on the y=x diagonal is simulated without bias; vertical
    distance from the diagonal is its mean bias. The panel is annotated with
    the fleet-level MBE and RMSE of the per-turbine means.

    Args:
        sim_cf: Simulated CF, a wide DataFrame in the on-disk schema
            (optional ``time`` column, one column per turbine ID, e.g. read
            from ``<run>/results/capacity-factor/*_unc_cf.csv``), or an
            already-reduced per-turbine Series of means.
        obs_cf: Observed CF, same accepted shapes.
        turb_info: Optional turbine metadata with ``ID`` and ``type``
            columns; when given, points are coloured by onshore/offshore.
        figsize: Figure size in inches.

    Returns:
        The constructed ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If sim and obs share no turbine IDs.
    """
    sim = _per_turbine_mean(sim_cf)
    obs = _per_turbine_mean(obs_cf)

    ids = sim.index.intersection(obs.index)
    if len(ids) == 0:
        raise ValueError("sim_cf and obs_cf share no turbine IDs")
    per_turb = pd.DataFrame({"obs": obs[ids], "sim": sim[ids]}).dropna()

    diff = per_turb["sim"] - per_turb["obs"]
    mbe = float(diff.mean())
    rmse = float(np.sqrt((diff**2).mean()))

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    if turb_info is not None and {"ID", "type"}.issubset(turb_info.columns):
        types = (
            turb_info.assign(ID=lambda d: d["ID"].astype(str))
            .drop_duplicates("ID")
            .set_index("ID")["type"]
            .reindex(per_turb.index)
            .fillna("unknown")
        )
        for turb_type, group in per_turb.groupby(types):
            ax.scatter(
                group["obs"], group["sim"], s=18,
                color=TURBINE_TYPE_COLOURS.get(turb_type, "#999999"),
                alpha=0.8, edgecolor="black", linewidths=0.3,
                label=str(turb_type).capitalize(),
            )
        ax.legend(loc="lower right", frameon=False)
    else:
        ax.scatter(
            per_turb["obs"], per_turb["sim"], s=18,
            color=TURBINE_TYPE_COLOURS["onshore"], alpha=0.8,
            edgecolor="black", linewidths=0.3,
        )

    hi = float(np.nanmax(per_turb.to_numpy())) * 1.05
    ax.plot([0, hi], [0, hi], color="black", linewidth=1.0, zorder=0)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.text(
        0.03, 0.97,
        f"MBE = {mbe:+.3f}\nRMSE = {rmse:.3f}\nn = {len(per_turb)}",
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
    )

    ax.set_xlabel("Observed mean CF")
    ax.set_ylabel("Simulated mean CF")
    ax.set_title("Per-turbine mean CF: simulated vs observed", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig
