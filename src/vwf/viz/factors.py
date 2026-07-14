"""Diagnostics for the learned correction factors.

Two figures that answer the core interpretability question for the method:
*what did the bias correction learn?*

:func:`plot_correction_factor_map` shows it spatially — each cluster's
Voronoi cell coloured by its learned ``scalar`` (multiplicative) and
``offset`` (additive) correction, on a diverging scale centred at the
neutral value (1 for scalar, 0 for offset) so over- and under-correction
read at a glance. Cluster geometry is rebuilt with
:func:`vwf.clustering.cluster_with_geometries`, which uses the same
deterministic KMeans configuration as training (``random_state=42``), so
passing the *training* turbine set reproduces the cluster IDs the factors
were fitted on.

:func:`plot_factor_joint` shows it in factor space — the joint distribution
of scalar vs offset across clusters, with marginal histograms and guides at
the neutral values. Tight clustering around (1, 0) means the reanalysis
needed little correction; spread or systematic displacement reveals the
structure of the bias.

Free functions; they take the factors table and turbine metadata the package
itself produces and return a ``matplotlib.figure.Figure`` so the caller
decides what to do next (``fig.savefig(...)``, further tweaks, etc.).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure


__all__ = ["plot_correction_factor_map", "plot_factor_joint"]


# Neutral value per factor column: the correction is a no-op at this value,
# so it anchors the centre of the diverging colour scale.
_FACTOR_MIDPOINTS = {"scalar": 1.0, "offset": 0.0}

_FACTOR_LABELS = {"scalar": "Scalar", "offset": "Offset"}


def _resolve_boundary(boundary):
    """Coerce a boundary argument into a single shapely geometry (or None).

    Accepts a shapely geometry, a GeoDataFrame/GeoSeries, or a path to any
    vector file geopandas can read (GeoJSON, shapefile, ...).
    """
    if boundary is None:
        return None

    import geopandas as gpd
    from shapely.geometry.base import BaseGeometry
    from shapely.ops import unary_union

    if isinstance(boundary, (str, Path)):
        boundary = gpd.read_file(boundary)
    if isinstance(boundary, gpd.GeoDataFrame):
        boundary = boundary.geometry
    if isinstance(boundary, gpd.GeoSeries):
        return unary_union(boundary.to_crs("EPSG:4326") if boundary.crs else boundary)
    if isinstance(boundary, BaseGeometry):
        return boundary
    raise TypeError(
        "boundary must be a shapely geometry, GeoDataFrame/GeoSeries, or a "
        f"path to a vector file; got {type(boundary).__name__}"
    )


def _select_period(factors: pd.DataFrame, period) -> pd.DataFrame:
    """Validate a factors table and optionally filter it to one period slice.

    Non-fixed temporal resolutions store one row per (cluster, period); the
    period column is whichever column is not cluster/scalar/offset (e.g.
    ``month``, ``season``). ``period=None`` returns the table unfiltered.
    """
    required = {"cluster", "scalar", "offset"}
    missing = required - set(factors.columns)
    if missing:
        raise ValueError(f"factors table is missing columns: {sorted(missing)}")

    if period is None:
        return factors

    slice_cols = [c for c in factors.columns if c not in required]
    if not slice_cols:
        raise ValueError(
            "period was given but the factors table has no period column "
            "(fixed temporal resolution has one row per cluster)"
        )
    sel = factors[factors[slice_cols[0]].astype(str) == str(period)]
    if sel.empty:
        available = sorted(factors[slice_cols[0]].astype(str).unique())
        raise ValueError(
            f"period {period!r} not found in column {slice_cols[0]!r}; "
            f"available: {available}"
        )
    return sel


def _factors_per_cluster(factors: pd.DataFrame, period) -> pd.DataFrame:
    """Reduce a factors table to one (scalar, offset) row per cluster.

    ``period`` selects one slice; ``None`` averages across slices.
    """
    sel = _select_period(factors, period)
    if period is not None:
        return sel[["cluster", "scalar", "offset"]]
    return sel.groupby("cluster", as_index=False)[["scalar", "offset"]].mean()


def _diverging_norm(values: np.ndarray, center: float) -> TwoSlopeNorm:
    """Build a TwoSlopeNorm centred at ``center`` that tolerates one-sided data.

    TwoSlopeNorm requires vmin < vcenter < vmax; when every cluster's factor
    sits on one side of neutral (common for offsets), pad the empty side so
    the neutral colour still marks "no correction".
    """
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    span = max(vmax - vmin, abs(vmax - center), abs(vmin - center), 1e-6)
    if vmin >= center:
        vmin = center - 0.05 * span
    if vmax <= center:
        vmax = center + 0.05 * span
    return TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)


def plot_correction_factor_map(
    factors: pd.DataFrame,
    turb_info: pd.DataFrame,
    *,
    boundary=None,
    columns: Sequence[str] = ("scalar", "offset"),
    period=None,
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Map the learned correction factors as a per-cluster choropleth.

    Rebuilds the training clusters from ``turb_info`` (same deterministic
    KMeans as training), tessellates them into Voronoi cells, and colours
    each cell by its correction factor on a diverging scale centred at the
    neutral value (scalar=1, offset=0).

    Args:
        factors: Correction-factor table for one ``(n_clu, time_res)``
            configuration, as produced by training and loaded by
            :func:`vwf.viz.load_results` (``Results.factors``). Must contain
            ``cluster``, ``scalar`` and ``offset`` columns; the cluster count
            is inferred from it.
        turb_info: Turbine metadata with ``lat``/``lon`` — the same
            *training* fleet the factors were fitted on
            (``Results.train_turb_info``), so cluster IDs reproduce. Passing
            a different fleet silently draws the factors over the wrong
            regions.
        boundary: Optional region outline to clip the Voronoi cells to: a
            shapely geometry, GeoDataFrame/GeoSeries, or a path to a vector
            file (GeoJSON, shapefile, ...). Without it, edge cells extend to
            the tessellation envelope.
        columns: Which factor columns to map, one panel each
            (default both ``scalar`` and ``offset``).
        period: For non-fixed temporal resolutions, the period slice to show
            (e.g. ``"winter"``, ``7``). Default ``None`` averages the factors
            across periods.
        cmap: Diverging colormap name (default ``"RdBu_r"``).
        figsize: Figure size in inches. Defaults to ``(4.5 * len(columns), 5)``.

    Returns:
        The constructed ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If the factors table is malformed, ``period`` doesn't
            exist, or ``turb_info`` has fewer turbines than clusters.
    """
    import geopandas as gpd  # noqa: F401 — hard requirement for the geometry path

    from vwf.clustering import cluster_with_geometries

    per_cluster = _factors_per_cluster(factors, period)
    n_clu = int(per_cluster["cluster"].nunique())

    if not {"lat", "lon"}.issubset(turb_info.columns):
        raise ValueError("turb_info must have 'lat' and 'lon' columns")
    if len(turb_info) < n_clu:
        raise ValueError(
            f"turb_info has {len(turb_info)} turbines but the factors table "
            f"implies {n_clu} clusters; pass the training fleet the factors "
            "were fitted on"
        )

    boundary_geom = _resolve_boundary(boundary)

    points = turb_info[["lat", "lon"]].copy()
    _, cluster_gdf = cluster_with_geometries(
        points,
        n_clu,
        method="kmeans",
        country_bounds=boundary_geom,
        geometry_type="voronoi",
    )
    if cluster_gdf is None:
        raise ImportError(
            "geopandas is required for plot_correction_factor_map"
        )

    cells = cluster_gdf.merge(per_cluster, on="cluster", how="left")
    cells = cells[~cells.geometry.is_empty & cells.geometry.notna()]

    unknown = [c for c in columns if c not in cells.columns]
    if unknown:
        raise ValueError(f"columns not present in factors table: {unknown}")

    if figsize is None:
        figsize = (4.5 * len(columns), 5.0)
    fig, axes = plt.subplots(1, len(columns), figsize=figsize, layout="constrained")
    axes = np.atleast_1d(axes)

    for ax, col in zip(axes, columns):
        values = cells[col].to_numpy(dtype=float)
        norm = _diverging_norm(values, _FACTOR_MIDPOINTS.get(col, 0.0))
        cells.plot(column=col, cmap=cmap, norm=norm, ax=ax, edgecolor="none")
        if boundary_geom is not None:
            gpd.GeoSeries([boundary_geom], crs="EPSG:4326").boundary.plot(
                ax=ax, edgecolor="black", linewidth=0.4
            )
        ax.set_axis_off()
        label = _FACTOR_LABELS.get(col, col)
        ax.set_title(label, fontsize=10)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(
            sm, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04,
            label=label,
        )

    suffix = f" ({period})" if period is not None else ""
    fig.suptitle(
        f"Learned correction factors, {n_clu} clusters{suffix}", fontsize=10
    )
    return fig


def plot_factor_joint(
    factors: pd.DataFrame,
    *,
    period=None,
    color: str = "#0072B2",
    bins: int | None = None,
    figsize: tuple[float, float] = (5.0, 5.0),
) -> Figure:
    """Joint distribution of the learned scalar and offset corrections.

    Scatter of scalar (x) vs offset (y) across clusters, with marginal
    histograms and dashed guides at the neutral values (scalar=1, offset=0).
    For non-fixed temporal resolutions each (cluster, period) pair is one
    point, so the seasonal spread of the correction is part of the picture;
    use ``period`` to isolate one slice.

    Args:
        factors: Correction-factor table for one ``(n_clu, time_res)``
            configuration (``Results.factors``), with ``cluster``,
            ``scalar`` and ``offset`` columns.
        period: For non-fixed temporal resolutions, plot only this period
            slice (e.g. ``"winter"``, ``7``). Default ``None`` pools all rows.
        color: Point and histogram colour (default Okabe-Ito blue).
        bins: Marginal histogram bin count. If ``None``, scales with the
            number of points.
        figsize: Figure size in inches.

    Returns:
        The constructed ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If the factors table is malformed or ``period`` doesn't
            exist.
    """
    sel = _select_period(factors, period)
    scalar = sel["scalar"].to_numpy(dtype=float)
    offset = sel["offset"].to_numpy(dtype=float)

    if bins is None:
        bins = max(8, min(30, len(sel) // 5))

    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(
        2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
        hspace=0.02, wspace=0.02,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_hx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_hy = fig.add_subplot(gs[1, 1], sharey=ax)

    ax.scatter(
        scalar, offset, s=14, color=color, alpha=0.7,
        edgecolor="black", linewidths=0.3,
    )
    # Neutral cross: the correction is a no-op at (1, 0)
    ax.axvline(1.0, color="grey", linewidth=0.8, linestyle="--", zorder=0)
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlabel("Scalar")
    ax.set_ylabel("Offset")

    ax_hx.hist(scalar, bins=bins, color=color, alpha=0.8)
    ax_hy.hist(offset, bins=bins, color=color, alpha=0.8, orientation="horizontal")

    # Hide marginal decor without touching the shared main-axis tickers:
    # tick_params is per-axes, but set_*ticks on a shared axis is not.
    ax_hx.tick_params(axis="x", labelbottom=False, bottom=False)
    ax_hx.set_yticks([])
    ax_hy.tick_params(axis="y", labelleft=False, left=False)
    ax_hy.set_xticks([])
    for marginal in (ax_hx, ax_hy):
        for spine in ("top", "right", "left", "bottom"):
            marginal.spines[spine].set_visible(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    suffix = f" ({period})" if period is not None else ""
    n_clu = int(sel["cluster"].nunique())
    fig.suptitle(
        f"Learned correction factors, {n_clu} clusters{suffix}", fontsize=10
    )
    return fig
