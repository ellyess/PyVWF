#!/usr/bin/env python3
"""Generate Chapter 4 grid interpolation figures.

Reproduces and extends the ch4_grid_interpolation figures using the
latest unified corrections data (1,729 centroids, 14 countries/modes).

Figures:
    1. Correction control point map (scalar + offset)
    2. Per-country correction impact (corrected vs uncorrected MAE)
    3. Correction factor distributions by region (box plots)
    4. Interpolation surfaces (Nearest / IDW / RBF / Kriging)
    5. Spatial coverage (distance to nearest control point)
    6. Interpolation cross-validation scores
    7. Scalar vs offset correlation
    8. Per-country correction summary statistics
    9. Grid vs cluster MAE comparison (IDW + Kriging)
   10. Grid vs cluster improvement percentage (IDW + Kriging)

Usage:
    python scripts/pyvwf_to_grid/generate_ch4_grid_plots.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from plotting_style import thesis_plot_style, format_axes_standard, savefig_thesis
from thesis_colors import OKABE_ITO, METHOD_COLOURS, COUNTRY_COLOURS

# ===========================================================================
# Style -- apply rcParams globally, then use them everywhere
# ===========================================================================
STYLE = thesis_plot_style()
cm = STYLE["cm"]
FULL_WIDTH = STYLE["FULL_WIDTH"]
HALF_WIDTH = STYLE["HALF_WIDTH"]
MAP_WIDTH = STYLE["MAP_WIDTH"]

# ===========================================================================
# Paths
# ===========================================================================
OUTPUT_DIR = PROJECT_ROOT / "output" / "pyvwf_to_grid" / "analysis_plots" / "ch4_grid_interpolation"
CENTROIDS_CSV = PROJECT_ROOT / "output" / "pyvwf_to_grid" / "all_corrections_centroids.csv"
GRID_DIR = PROJECT_ROOT / "output" / "pyvwf_to_grid" / "grid_comparison"
EVAL_DIR = PROJECT_ROOT / "output" / "pyvwf_to_grid" / "grid_evaluation"
EVAL_METRICS_CSV = PROJECT_ROOT / "output" / "runs" / "turbine_grid" / "pyvwf_evaluation_metrics.csv"

# Map extent
EUROPE_EXTENT = [-12, 32, 34, 73]

# Interpolation methods
INTERP_METHODS = ["nearest", "idw", "rbf", "kriging"]
INTERP_LABELS = {
    "nearest": "Nearest Neighbour",
    "idw": "IDW",
    "rbf": "RBF",
    "kriging": "Kriging",
}

# METHOD_COLOURS and COUNTRY_COLOURS imported from thesis_colors


# ===========================================================================
# Data Loading
# ===========================================================================
def load_corrections():
    """Load correction centroids with label column."""
    df = pd.read_csv(CENTROIDS_CSV)
    df["label"] = df.apply(
        lambda r: f"{r['country_code']}-{r['cluster_mode']}"
        if r["cluster_mode"] != "all" else r["country_code"],
        axis=1,
    )
    return df


def load_grids():
    """Load interpolation grid NetCDF files."""
    grids = {}
    for method in INTERP_METHODS:
        nc_path = GRID_DIR / f"europe_corrections_{method}.nc"
        if nc_path.exists():
            grids[method] = xr.open_dataset(nc_path)
        else:
            print(f"  Warning: {nc_path.name} not found, skipping {method}")
    return grids


def load_cv_scores():
    """Load cross-validation scores."""
    csv_path = GRID_DIR / "cv_scores.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path, index_col=0)


def load_eval_metrics():
    """Load evaluation metrics (corrected vs uncorrected)."""
    if not EVAL_METRICS_CSV.exists():
        return None
    return pd.read_csv(EVAL_METRICS_CSV)


def load_grid_eval_metrics():
    """Load grid evaluation metrics (grid vs cluster comparison)."""
    csv_path = EVAL_DIR / "grid_evaluation_metrics.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


# ===========================================================================
# Map helpers
# ===========================================================================
def make_map_ax(fig, nrows, ncols, index, extent=None):
    """Create a map-like axis."""
    if extent is None:
        extent = EUROPE_EXTENT
    ax = fig.add_subplot(nrows, ncols, index)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    return ax


# ===========================================================================
# Figure 1: Correction Control Point Map
# ===========================================================================
def fig1_correction_map(centroids_df):
    """European correction control point map."""
    print("\n--- Figure 1: Correction Map ---")

    fig = plt.figure(figsize=(FULL_WIDTH, 8 * cm))

    for i, (var, vcenter, label) in enumerate([
        ("scalar", 1.0, "Scalar correction factor"),
        ("offset", 0.0, "Offset correction (m/s)"),
    ]):
        ax = make_map_ax(fig, 1, 2, i + 1)
        vals = centroids_df[var].values
        vmin = np.nanpercentile(vals, 2)
        vmax = np.nanpercentile(vals, 98)
        vmin = min(vmin, vcenter - 0.01)
        vmax = max(vmax, vcenter + 0.01)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        sc = ax.scatter(
            centroids_df["lon"], centroids_df["lat"],
            c=vals, cmap="RdBu_r", norm=norm,
            s=3, edgecolors="none", alpha=0.8,
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(label)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(var.capitalize(), fontweight="bold")

    fig.tight_layout()
    path = OUTPUT_DIR / "ch4_fig1_correction_map.png"
    format_axes_standard(fig)

    savefig_thesis(fig, path)
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 2: Per-country correction impact
# ===========================================================================
def fig2_country_performance(eval_df):
    """Per-country corrected vs uncorrected MAE."""
    print("\n--- Figure 2: Country Performance ---")
    if eval_df is None:
        print("  No evaluation data, skipping.")
        return

    corrected = eval_df[eval_df["correction_type"] == "corrected"].copy()
    uncorrected = eval_df[eval_df["correction_type"] == "uncorrected"].copy()

    def make_label(row):
        mode = row.get("mode", "all")
        country = row["country"]
        return f"{country}-{mode}" if mode != "all" else country

    corrected["label"] = corrected.apply(make_label, axis=1)
    uncorrected["label"] = uncorrected.apply(make_label, axis=1)

    merged = corrected[["label", "mae"]].rename(columns={"mae": "corrected_mae"}).merge(
        uncorrected[["label", "mae"]].rename(columns={"mae": "uncorrected_mae"}),
        on="label", how="inner",
    )
    merged["improvement_pct"] = (
        (merged["uncorrected_mae"] - merged["corrected_mae"]) / merged["uncorrected_mae"] * 100
    )
    merged = merged.sort_values("improvement_pct", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 6 * cm))
    x = np.arange(len(merged))
    width = 0.35

    ax.bar(x - width / 2, merged["uncorrected_mae"], width,
           color=OKABE_ITO[5], edgecolor="black", linewidth=0.3,
           label="Uncorrected", alpha=0.85)
    ax.bar(x + width / 2, merged["corrected_mae"], width,
           color=OKABE_ITO[2], edgecolor="black", linewidth=0.3,
           label="Corrected", alpha=0.85)

    for i, row in merged.iterrows():
        y_max = max(row["uncorrected_mae"], row["corrected_mae"])
        colour = OKABE_ITO[2] if row["improvement_pct"] > 0 else OKABE_ITO[5]
        ax.text(i, y_max + 0.005,
                f"{row['improvement_pct']:+.0f}%",
                ha="center", va="bottom", fontsize=5, fontweight="bold",
                color=colour)

    ax.set_xticks(x)
    ax.set_xticklabels(merged["label"], rotation=35, ha="right")
    ax.set_ylabel("MAE (capacity factor)")
    ax.legend(frameon=False, loc="upper right")
    sns.despine(ax=ax)
    fig.tight_layout()

    path = OUTPUT_DIR / "ch4_fig2_country_performance.png"
    format_axes_standard(fig)

    savefig_thesis(fig, path)
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 3: Correction distributions
# ===========================================================================
def fig3_correction_distributions(centroids_df):
    """Correction factor distributions by region (box plots)."""
    print("\n--- Figure 3: Correction Distributions ---")

    order = (centroids_df.groupby("label")["scalar"]
             .median().sort_values().index.tolist())

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 7 * cm))

    for ax, (var, title, ref_line) in zip(axes, [
        ("scalar", "Scalar Correction", 1.0),
        ("offset", "Offset Correction (m/s)", 0.0),
    ]):
        data_groups = [centroids_df.loc[centroids_df["label"] == lab, var].dropna().values
                       for lab in order]
        bp = ax.boxplot(
            data_groups, vert=True, patch_artist=True,
            showfliers=True, flierprops=dict(marker=".", markersize=1.5, alpha=0.4),
            medianprops=dict(color=OKABE_ITO[7], linewidth=0.8),
            boxprops=dict(linewidth=0.5),
            whiskerprops=dict(linewidth=0.5),
            capprops=dict(linewidth=0.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(OKABE_ITO[1])
            patch.set_alpha(0.6)
        ax.axhline(ref_line, color=OKABE_ITO[5], linewidth=0.6, linestyle="--", alpha=0.6,
                    label=f"Identity ({ref_line})")
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=6)
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=False)
        sns.despine(ax=ax)

    fig.tight_layout()
    path = OUTPUT_DIR / "ch4_fig3_correction_distributions.png"
    format_axes_standard(fig)

    savefig_thesis(fig, path)
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 4: Interpolation surfaces
# ===========================================================================
def _diverging_norm_cmap(vmin, vcenter, vmax, cmap_name="RdBu_r"):
    """Return (cmap, norm) using plain Normalize but a colormap that shifts the
    white point to vcenter.  Visually identical to TwoSlopeNorm + RdBu_r but
    avoids the matplotlib 3.9.2 bug where TwoSlopeNorm + colorbar tick placement
    produces NaN positions when the range is highly asymmetric."""
    white_frac = (vcenter - vmin) / (vmax - vmin)
    base = plt.cm.get_cmap(cmap_name)
    n = 512
    vals = np.linspace(0, 1, n)
    colors = []
    for v in vals:
        if v <= white_frac:
            colors.append(base(0.5 * v / white_frac if white_frac > 0 else 0.0))
        else:
            colors.append(base(0.5 + 0.5 * (v - white_frac) / (1.0 - white_frac)
                               if white_frac < 1 else 1.0))
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmap = LinearSegmentedColormap.from_list("_div_shifted", colors, N=n)
    norm = Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm


def fig4_interpolation_surfaces(grids):
    """2x4 grid of interpolated surfaces."""
    print("\n--- Figure 4: Interpolation Surfaces ---")
    methods = [m for m in INTERP_METHODS if m in grids]
    ncols = len(methods)
    if ncols == 0:
        print("  No grids found, skipping.")
        return

    fig = plt.figure(figsize=(FULL_WIDTH, 9 * cm))

    row_axes = {0: [], 1: []}
    row_cbar_info = {}  # (pcm_artist, vmin, vcenter, vmax)
    for row, (var, vcenter, label) in enumerate([
        ("scalar", 1.0, "Scalar"),
        ("offset", 0.0, "Offset (m/s)"),
    ]):
        all_vals = []
        for m in methods:
            ds_m = grids[m]
            if var in ds_m:
                data = ds_m[var].values
                all_vals.append(data[np.isfinite(data)])
        if not all_vals:
            continue
        all_vals = np.concatenate(all_vals)
        vmin = np.nanpercentile(all_vals, 2)
        vmax = np.nanpercentile(all_vals, 98)
        vmin = min(vmin, vcenter - 0.01)
        vmax = max(vmax, vcenter + 0.01)
        cmap, norm = _diverging_norm_cmap(vmin, vcenter, vmax)

        last_pcm = None
        for col, method in enumerate(methods):
            idx = row * ncols + col + 1
            ax = make_map_ax(fig, 2, ncols, idx)
            row_axes[row].append(ax)
            ds = grids[method]
            if var not in ds:
                continue
            lon_key = "lon" if "lon" in ds else "x"
            lat_key = "lat" if "lat" in ds else "y"
            last_pcm = ax.pcolormesh(ds[lon_key].values, ds[lat_key].values, ds[var].values,
                                     cmap=cmap, norm=norm, shading="auto")
            if row == 1:
                ax.set_xlabel("Longitude")
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(label, fontweight="bold")
            else:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_title(INTERP_LABELS.get(method, method), fontweight="bold")

        if last_pcm is not None:
            row_cbar_info[row] = (last_pcm, vmin, vcenter, vmax)

    fig.subplots_adjust(wspace=0.02, hspace=0.08)

    # Add colorbars — plain Normalize (via _diverging_norm_cmap) avoids the
    # matplotlib 3.9.2 TwoSlopeNorm + colorbar tick-placement NaN bug.
    for row in row_cbar_info:
        pcm, vmin, vcenter, vmax = row_cbar_info[row]
        cbar = fig.colorbar(pcm, ax=row_axes[row], shrink=0.7, pad=0.01, aspect=25)
        tick_vals = [round(vmin, 2), vcenter, round(vmax, 2)]
        cbar.set_ticks(tick_vals)
        cbar.ax.tick_params(labelsize=6)

    path = OUTPUT_DIR / "ch4_fig4_interpolation_surfaces.png"
    format_axes_standard(fig)

    savefig_thesis(fig, path)
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 5: Spatial coverage
# ===========================================================================
def fig5_spatial_coverage(grids, centroids_df):
    """Distance-to-nearest-control heatmap."""
    print("\n--- Figure 5: Spatial Coverage ---")
    ds = grids.get("idw") or next(iter(grids.values()), None)
    if ds is None or "distance_to_nearest_control" not in ds:
        print("  No distance data, skipping.")
        return

    lon_key = "lon" if "lon" in ds else "x"
    lat_key = "lat" if "lat" in ds else "y"

    fig = plt.figure(figsize=(HALF_WIDTH + 2 * cm, 8 * cm))
    ax = make_map_ax(fig, 1, 1, 1)
    dist = ds["distance_to_nearest_control"].values

    pcm = ax.pcolormesh(ds[lon_key].values, ds[lat_key].values, dist,
                        cmap="YlOrRd", shading="auto", vmin=0, vmax=6)
    cbar = plt.colorbar(pcm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Distance to nearest control point (\u00b0)")

    ax.scatter(centroids_df["lon"], centroids_df["lat"],
               s=1, c=OKABE_ITO[7], alpha=0.5, zorder=5)
    ax.contour(ds[lon_key].values, ds[lat_key].values, dist, levels=[5.0],
               colors=[OKABE_ITO[5]], linewidths=[0.8], linestyles=["--"])

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    path = OUTPUT_DIR / "ch4_fig5_spatial_coverage.png"
    format_axes_standard(fig)

    savefig_thesis(fig, path)
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 6: Interpolation cross-validation
# ===========================================================================
def fig6_interpolation_cv(cv_df):
    """Interpolation cross-validation scores (grouped bar chart)."""
    print("\n--- Figure 6: Interpolation CV ---")
    if cv_df is None:
        print("  No CV data, skipping.")
        return

    methods = cv_df.index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 5.5 * cm), sharey=False)

    for ax_idx, (prefix, title) in enumerate([("scalar", "Scalar"), ("offset", "Offset")]):
        ax = axes[ax_idx]
        x = np.arange(len(methods))
        width = 0.35

        mae_mean = cv_df[f"{prefix}_mae_mean"].values
        mae_std = cv_df[f"{prefix}_mae_std"].values
        rmse_mean = cv_df[f"{prefix}_rmse_mean"].values
        rmse_std = cv_df[f"{prefix}_rmse_std"].values
        colours = [METHOD_COLOURS.get(m, OKABE_ITO[7]) for m in methods]

        bars1 = ax.bar(x - width / 2, mae_mean, width, yerr=mae_std,
                       color=colours, edgecolor="black", linewidth=0.3,
                       capsize=2, error_kw={"linewidth": 0.5}, label="MAE")
        ax.bar(x + width / 2, rmse_mean, width, yerr=rmse_std,
               color=colours, edgecolor="black", linewidth=0.3,
               capsize=2, error_kw={"linewidth": 0.5}, alpha=0.5, label="RMSE")

        ax.set_xticks(x)
        ax.set_xticklabels([INTERP_LABELS.get(m, m) for m in methods])
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Error" if ax_idx == 0 else "")
        ax.legend(frameon=False)
        sns.despine(ax=ax)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=5)

    fig.tight_layout()
    path = OUTPUT_DIR / "ch4_fig6_interpolation_cv.png"
    format_axes_standard(fig)

    savefig_thesis(fig, path)
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 7: Scalar vs Offset correlation
# ===========================================================================
def fig7_scalar_vs_offset(centroids_df):
    """Scatter showing relationship between scalar and offset corrections."""
    print("\n--- Figure 7: Scalar vs Offset ---")

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 7 * cm))

    # Panel A: scatter coloured by country
    ax = axes[0]
    countries = centroids_df["country_code"].unique()

    def base_country(code):
        return code.split("-")[0]

    for country in sorted(countries):
        mask = centroids_df["country_code"] == country
        base = base_country(country)
        colour = COUNTRY_COLOURS.get(base, OKABE_ITO[7])
        n = mask.sum()
        label = country if n >= 3 else None
        ax.scatter(centroids_df.loc[mask, "scalar"],
                   centroids_df.loc[mask, "offset"],
                   s=3, alpha=0.5, c=colour, label=label,
                   edgecolors="none")

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(1, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Scalar correction")
    ax.set_ylabel("Offset correction (m/s)")
    ax.set_title("(a) By country", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="lower left",
              markerscale=2, handletextpad=0.2, columnspacing=0.5)
    sns.despine(ax=ax)

    # Correlation
    valid = centroids_df[["scalar", "offset"]].dropna()
    corr = valid["scalar"].corr(valid["offset"])
    ax.text(0.98, 0.98, f"r = {corr:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey",
                      alpha=0.8, linewidth=0.5))

    # Panel B: 2D histogram (density)
    ax2 = axes[1]
    h = ax2.hist2d(centroids_df["scalar"].values, centroids_df["offset"].values,
                   bins=40, cmap="Blues", cmin=1)
    plt.colorbar(h[3], ax=ax2, shrink=0.8, pad=0.02)
    ax2.axhline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
    ax2.axvline(1, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Scalar correction")
    ax2.set_ylabel("Offset correction (m/s)")
    ax2.set_title("(b) Density", fontweight="bold")

    fig.tight_layout()
    path = OUTPUT_DIR / "ch4_fig7_scalar_vs_offset.png"
    format_axes_standard(fig)

    savefig_thesis(fig, path)
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 8: Per-country correction summary
# ===========================================================================
def fig8_country_summary(centroids_df):
    """Per-country summary: cluster counts, mean scalar/offset."""
    print("\n--- Figure 8: Country Summary ---")

    summary = centroids_df.groupby("country_code").agg(
        n_clusters=("cluster", "count"),
        scalar_mean=("scalar", "mean"),
        scalar_std=("scalar", "std"),
        offset_mean=("offset", "mean"),
        offset_std=("offset", "std"),
    ).reset_index()

    summary = summary.sort_values("n_clusters", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 6 * cm),
                             gridspec_kw={"width_ratios": [1.2, 1, 1]})

    # Panel A: Cluster counts
    ax = axes[0]
    base_colours = [COUNTRY_COLOURS.get(c.split("-")[0], OKABE_ITO[7])
                    for c in summary["country_code"]]
    bars = ax.barh(range(len(summary)), summary["n_clusters"],
                   color=base_colours, edgecolor="black", linewidth=0.3, alpha=0.85)
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(summary["country_code"])
    ax.set_xlabel("Number of clusters")
    ax.set_title("(a) Cluster count", fontweight="bold")
    ax.invert_yaxis()
    for bar, val in zip(bars, summary["n_clusters"]):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=5)
    sns.despine(ax=ax)

    # Panel B: Mean scalar
    ax2 = axes[1]
    ax2.barh(range(len(summary)), summary["scalar_mean"],
             xerr=summary["scalar_std"].fillna(0),
             color=OKABE_ITO[4], edgecolor="black", linewidth=0.3,
             alpha=0.75, capsize=2, error_kw={"linewidth": 0.5})
    ax2.axvline(1.0, color=OKABE_ITO[5], linewidth=0.6, linestyle="--", alpha=0.6)
    ax2.set_yticks(range(len(summary)))
    ax2.set_yticklabels([])
    ax2.set_xlabel("Mean scalar")
    ax2.set_title("(b) Mean scalar", fontweight="bold")
    ax2.invert_yaxis()
    sns.despine(ax=ax2)

    # Panel C: Mean offset
    ax3 = axes[2]
    ax3.barh(range(len(summary)), summary["offset_mean"],
             xerr=summary["offset_std"].fillna(0),
             color=OKABE_ITO[0], edgecolor="black", linewidth=0.3,
             alpha=0.75, capsize=2, error_kw={"linewidth": 0.5})
    ax3.axvline(0.0, color=OKABE_ITO[5], linewidth=0.6, linestyle="--", alpha=0.6)
    ax3.set_yticks(range(len(summary)))
    ax3.set_yticklabels([])
    ax3.set_xlabel("Mean offset (m/s)")
    ax3.set_title("(c) Mean offset", fontweight="bold")
    ax3.invert_yaxis()
    sns.despine(ax=ax3)

    fig.tight_layout()
    path = OUTPUT_DIR / "ch4_fig8_country_summary.png"
    format_axes_standard(fig)

    savefig_thesis(fig, path)
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 9: Grid vs Cluster MAE comparison
# ===========================================================================
def fig9_grid_vs_cluster_mae(grid_eval_df):
    """Per-region MAE comparison: uncorrected, cluster, grid IDW, grid Kriging."""
    print("\n--- Figure 9: Grid vs Cluster MAE ---")
    if grid_eval_df is None:
        print("  No grid evaluation data, skipping.")
        return

    # Build region labels
    def make_label(row):
        mode = row.get("mode", "all")
        level = row.get("obs_level", "")
        suffix = f" ({level})" if level == "country" else ""
        return f"{row['country']}-{mode}{suffix}" if mode != "all" else f"{row['country']}{suffix}"

    grid_eval_df["label"] = grid_eval_df.apply(make_label, axis=1)

    # Pivot by correction source
    uncorr = grid_eval_df[grid_eval_df["correction_source"] == "uncorrected"].set_index("label")
    cluster = grid_eval_df[grid_eval_df["correction_source"] == "cluster_fixed"].set_index("label")

    # Separate IDW and kriging grid sources via 'method' column
    grid_all = grid_eval_df[grid_eval_df["correction_source"] == "grid"]
    method_col = "grid_method" if "grid_method" in grid_all.columns else "method"
    has_both = set(grid_all[method_col].unique()) >= {"idw", "kriging"}
    if has_both:
        grid_idw = grid_all[grid_all[method_col] == "idw"].set_index("label")
        grid_krig = grid_all[grid_all[method_col] == "kriging"].set_index("label")
    else:
        grid_idw = grid_all.set_index("label")
        grid_krig = None

    labels = uncorr.index.tolist()

    if grid_krig is not None and len(grid_krig) > 0:
        # 4-bar layout: uncorrected, cluster, IDW, kriging
        fig, ax = plt.subplots(figsize=(FULL_WIDTH, 7 * cm))
        x = np.arange(len(labels))
        width = 0.2

        ax.bar(x - 1.5 * width, uncorr.loc[labels, "mae"].values, width,
               color=OKABE_ITO[5], edgecolor="black", linewidth=0.3,
               label="Uncorrected", alpha=0.85)
        ax.bar(x - 0.5 * width, cluster.loc[labels, "mae"].values, width,
               color=OKABE_ITO[2], edgecolor="black", linewidth=0.3,
               label="Cluster", alpha=0.85)
        ax.bar(x + 0.5 * width, grid_idw.loc[labels, "mae"].values, width,
               color=OKABE_ITO[4], edgecolor="black", linewidth=0.3,
               label="Grid (IDW)", alpha=0.85)
        ax.bar(x + 1.5 * width, grid_krig.loc[labels, "mae"].values, width,
               color=OKABE_ITO[0], edgecolor="black", linewidth=0.3,
               label="Grid (Kriging)", alpha=0.85)
    else:
        # 3-bar layout fallback
        fig, ax = plt.subplots(figsize=(FULL_WIDTH, 6.5 * cm))
        x = np.arange(len(labels))
        width = 0.25

        ax.bar(x - width, uncorr.loc[labels, "mae"].values, width,
               color=OKABE_ITO[5], edgecolor="black", linewidth=0.3,
               label="Uncorrected", alpha=0.85)
        ax.bar(x, cluster.loc[labels, "mae"].values, width,
               color=OKABE_ITO[2], edgecolor="black", linewidth=0.3,
               label="Cluster", alpha=0.85)
        ax.bar(x + width, grid_idw.loc[labels, "mae"].values, width,
               color=OKABE_ITO[4], edgecolor="black", linewidth=0.3,
               label="Grid (IDW)", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("MAE (capacity factor)")
    ax.legend(frameon=False, loc="upper right")
    sns.despine(ax=ax)
    fig.tight_layout()

    path = OUTPUT_DIR / "ch4_fig9_grid_vs_cluster_mae.png"
    format_axes_standard(fig)

    savefig_thesis(fig, path)
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 10: Grid vs Cluster improvement percentage
# ===========================================================================
def fig10_grid_vs_cluster_improvement(grid_eval_df):
    """Per-region MAE improvement (%) over uncorrected: cluster vs grid IDW vs grid Kriging."""
    print("\n--- Figure 10: Grid vs Cluster Improvement ---")
    if grid_eval_df is None:
        print("  No grid evaluation data, skipping.")
        return

    # Build region labels
    def make_label(row):
        mode = row.get("mode", "all")
        level = row.get("obs_level", "")
        suffix = f" ({level})" if level == "country" else ""
        return f"{row['country']}-{mode}{suffix}" if mode != "all" else f"{row['country']}{suffix}"

    grid_eval_df["label"] = grid_eval_df.apply(make_label, axis=1)

    uncorr = grid_eval_df[grid_eval_df["correction_source"] == "uncorrected"].set_index("label")
    cluster = grid_eval_df[grid_eval_df["correction_source"] == "cluster_fixed"].set_index("label")

    grid_all = grid_eval_df[grid_eval_df["correction_source"] == "grid"]
    method_col = "grid_method" if "grid_method" in grid_all.columns else "method"
    has_both = set(grid_all[method_col].unique()) >= {"idw", "kriging"}
    if has_both:
        grid_idw = grid_all[grid_all[method_col] == "idw"].set_index("label")
        grid_krig = grid_all[grid_all[method_col] == "kriging"].set_index("label")
    else:
        grid_idw = grid_all.set_index("label")
        grid_krig = None

    labels = uncorr.index.tolist()

    # Compute improvement percentages
    cluster_imp, idw_imp, krig_imp = [], [], []
    for lab in labels:
        u = uncorr.loc[lab, "mae"]
        cluster_imp.append((u - cluster.loc[lab, "mae"]) / u * 100)
        idw_imp.append((u - grid_idw.loc[lab, "mae"]) / u * 100 if lab in grid_idw.index else 0)
        if grid_krig is not None and lab in grid_krig.index:
            krig_imp.append((u - grid_krig.loc[lab, "mae"]) / u * 100)

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 7 * cm))
    x = np.arange(len(labels))

    if grid_krig is not None and len(krig_imp) == len(labels):
        # 3-group layout: cluster, IDW, kriging
        width = 0.25
        bars_c = ax.bar(x - width, cluster_imp, width,
                        color=OKABE_ITO[2], edgecolor="black", linewidth=0.3,
                        label="Cluster", alpha=0.85)
        bars_i = ax.bar(x, idw_imp, width,
                        color=OKABE_ITO[4], edgecolor="black", linewidth=0.3,
                        label="Grid (IDW)", alpha=0.85)
        bars_k = ax.bar(x + width, krig_imp, width,
                        color=OKABE_ITO[0], edgecolor="black", linewidth=0.3,
                        label="Grid (Kriging)", alpha=0.85)
        all_bars = [bars_c, bars_i, bars_k]
    else:
        # 2-group fallback
        width = 0.35
        bars_c = ax.bar(x - width / 2, cluster_imp, width,
                        color=OKABE_ITO[2], edgecolor="black", linewidth=0.3,
                        label="Cluster", alpha=0.85)
        bars_i = ax.bar(x + width / 2, idw_imp, width,
                        color=OKABE_ITO[4], edgecolor="black", linewidth=0.3,
                        label="Grid (IDW)", alpha=0.85)
        all_bars = [bars_c, bars_i]

    # Add percentage labels
    for bars in all_bars:
        for bar in bars:
            val = bar.get_height()
            y_pos = val + 1 if val >= 0 else val - 1
            va = "bottom" if val >= 0 else "top"
            colour = OKABE_ITO[2] if val > 0 else OKABE_ITO[5]
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{val:.0f}%", ha="center", va=va,
                    fontsize=5, fontweight="bold", color=colour)

    ax.axhline(0, color=OKABE_ITO[7], linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("MAE Improvement over Uncorrected (%)")
    ax.legend(frameon=False, loc="upper right")
    sns.despine(ax=ax)
    fig.tight_layout()

    path = OUTPUT_DIR / "ch4_fig10_grid_vs_cluster_improvement.png"
    format_axes_standard(fig)

    savefig_thesis(fig, path)
    print(f"  Saved: {path}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 70)
    print("Generating Chapter 4 Grid Interpolation Figures")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    centroids_df = load_corrections()
    print(f"  Centroids: {len(centroids_df)} rows, "
          f"{centroids_df['country_code'].nunique()} country/mode groups")

    grids = load_grids()
    print(f"  Grids loaded: {list(grids.keys())}")

    cv_df = load_cv_scores()
    eval_df = load_eval_metrics()
    grid_eval_df = load_grid_eval_metrics()
    if grid_eval_df is not None:
        print(f"  Grid evaluation: {len(grid_eval_df)} rows")

    # Generate figures
    # fig1_correction_map(centroids_df)
    # fig2_country_performance(eval_df)
    # fig3_correction_distributions(centroids_df)
    fig4_interpolation_surfaces(grids)
    # fig5_spatial_coverage(grids, centroids_df)
    # fig6_interpolation_cv(cv_df)
    # fig7_scalar_vs_offset(centroids_df)
    # fig8_country_summary(centroids_df)
    # fig9_grid_vs_cluster_mae(grid_eval_df)
    # fig10_grid_vs_cluster_improvement(grid_eval_df)

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Control points: {len(centroids_df)}")
    print(f"Countries/modes: {centroids_df['country_code'].nunique()}")
    print(f"Scalar: mean={centroids_df['scalar'].mean():.3f}, "
          f"std={centroids_df['scalar'].std():.3f}, "
          f"range=[{centroids_df['scalar'].min():.3f}, {centroids_df['scalar'].max():.3f}]")
    print(f"Offset: mean={centroids_df['offset'].mean():.3f}, "
          f"std={centroids_df['offset'].std():.3f}, "
          f"range=[{centroids_df['offset'].min():.3f}, {centroids_df['offset'].max():.3f}]")

    if cv_df is not None:
        print("\nInterpolation CV (spatial folds):")
        for method in cv_df.index:
            print(f"  {INTERP_LABELS.get(method, method):20s}  "
                  f"scalar MAE={cv_df.loc[method, 'scalar_mae_mean']:.4f} "
                  f"\u00b1 {cv_df.loc[method, 'scalar_mae_std']:.4f}  "
                  f"offset MAE={cv_df.loc[method, 'offset_mae_mean']:.4f} "
                  f"\u00b1 {cv_df.loc[method, 'offset_mae_std']:.4f}")

    print("\n" + "=" * 70)
    print("ALL FIGURES GENERATED")
    print("=" * 70)
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
