#!/usr/bin/env python3
"""Generate Chapter 3 plots: DK onshore turbine-level results.

Plots generated:
  1. cluster_analysis.png          – Elbow + silhouette score for cluster count
  2. offset_vs_scalar.png          – Joint distribution of scalar vs offset
  3. voronoi_scalar_map.png        – Voronoi map coloured by scalar
  4. voronoi_offset_map.png        – Voronoi map coloured by offset
  5. train_turbine_locations.png   – Training and validation turbine locations
  6. temporal_slicing.pdf          – Temporal resolution scheme diagram
  7. observed_vs_simulated_cf.png  – Observed vs simulated CF at turbine locations
  8. era5_DK_train_bias.png        – Sim vs obs scatter (training)
  9. era5_DK_train_monthly.png     – Monthly boxplot sim vs obs (training)
 10. era5_DK_train_error_vs_clusters.png – Training MAE/RMSE vs clusters
"""
import warnings
warnings.filterwarnings("ignore")

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from scipy.spatial import Voronoi
import scipy as sp
import matplotlib.patheffects as pe
import shapely
import shapely.ops
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Add project root to path for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from plotting_style import thesis_plot_style, format_axes_standard, savefig_thesis
from thesis_colors import OKABE_ITO, TIME_RES_COLOURS, EXISTING_NEW_COLOURS
from scripts.evaluate_all_pyvwf_runs import evaluate_run
from vwf.datasets.era5 import prep_era5
from vwf.data import load_power_curves
from vwf.clustering import cluster_turbines
from vwf.metrics import overall_error
from vwf.metrics import calculate_error
import vwf.wind as wind

# =============================================================================
# CONSTANTS
# =============================================================================
STYLE = thesis_plot_style()
cm = STYLE["cm"]
FULL_WIDTH = STYLE["FULL_WIDTH"]
HALF_WIDTH = STYLE["HALF_WIDTH"]
THIRD_WIDTH = STYLE["THIRD_WIDTH"]
MAP_WIDTH = STYLE["MAP_WIDTH"]

COUNTRY = "DK"
YEAR_TEST = 2020

RUN_DIR = PROJECT_ROOT / "output" / "runs" / "turbine_dk_research" / "DK-onshore-obs_turbine-corrected-calc_z0"

TIME_RES_LABELS = {
    "fixed": "Fixed",
    "season": "Seasonal",
    "bimonth": "Bimonthly",
    "month": "Monthly",
}
TIME_RES_ORDER = {"fixed": 0, "season": 1, "bimonth": 2, "month": 3}
# TIME_RES_COLOURS imported from thesis_colors
TIME_RES_LINESTYLES = {
    "fixed": "-.",
    "season": ":",
    "bimonth": "--",
    "month": "-",
}

# OKABE_ITO and EXISTING_NEW_COLOURS imported from thesis_colors
EXISTING_NEW_LABELS = {
    "Yes": "Existing turbines",
    "No": "New turbines",
}


def ordered_time_res(values=None):
    """Return temporal-resolution values in canonical plotting order."""
    if values is None:
        values = list(TIME_RES_ORDER.keys())
    return sorted(values, key=lambda x: TIME_RES_ORDER.get(x, 99))


# =============================================================================
# HELPERS
# =============================================================================
def load_dk_shape():
    """Load Denmark country boundary from GeoJSON."""
    shape_path = PROJECT_ROOT / "input" / "regions" / "dk.json"
    if shape_path.exists():
        dk = gpd.read_file(shape_path)
        if len(dk) > 0:
            return dk
    return None


def scatter_mercator(ax, df, **kwargs):
    """Scatter lon/lat points projected to EPSG:3395."""
    if df is None or len(df) == 0:
        return None
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3395)
    return ax.scatter(gdf.geometry.x, gdf.geometry.y, **kwargs)


def to_long_cf(df, value_name):
    """Convert a wide CF DataFrame (time × turbine IDs) to long format."""
    df = df.copy()
    if "ID" in df.columns:
        if value_name not in df.columns and "cf" in df.columns:
            df = df.rename(columns={"cf": value_name})
        keep = [c for c in ["time", "year", "month", "ID", value_name] if c in df.columns]
        return df[keep]
    id_vars = [c for c in ["time", "year", "month"] if c in df.columns]
    return df.melt(id_vars=id_vars, var_name="ID", value_name=value_name)


def fit_clusters(num_clu, turb_train):
    """Fit KMeans and return the fitted model + labelled DataFrame."""
    kmeans = KMeans(
        init="random", n_clusters=num_clu,
        n_init=10, max_iter=300, random_state=42,
    )
    kmeans.fit(turb_train[["lat", "lon"]])
    turb_train = turb_train.copy()
    turb_train["cluster"] = kmeans.predict(turb_train[["lat", "lon"]])
    return kmeans, turb_train


def simulate_corrected_training_cf(base_df, factors, time_res):
    """Simulate corrected training CF from uncorrected sim CF + correction factors.

    Args:
        base_df: DataFrame with columns ['ID', 'sim', 'obs', 'month', 'cluster'].
        factors: Correction factors table for one (time_res, n_clu) configuration.
        time_res: One of {'fixed', 'month', 'season', 'bimonth'}.

    Returns:
        DataFrame containing at least ['ID', 'month', 'obs', 'sim', 'sim_cor'] for
        rows where correction factors were successfully applied.
    """
    month_to_season = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn",
    }

    cur = base_df.copy()
    if time_res == "fixed":
        cur = cur.merge(factors[["cluster", "scalar", "offset"]], on="cluster", how="left")
    elif time_res == "month":
        factors = factors.copy()
        factors["month"] = factors["month"].astype(int)
        cur = cur.merge(factors[["cluster", "month", "scalar", "offset"]],
                        on=["cluster", "month"], how="left")
    elif time_res == "season":
        cur["season"] = cur["month"].map(month_to_season)
        cur = cur.merge(factors[["cluster", "season", "scalar", "offset"]],
                        on=["cluster", "season"], how="left")
    elif time_res == "bimonth":
        cur["bimonth"] = ((cur["month"].astype(int) - 1) // 2 + 1).astype(str) + "/6"
        cur = cur.merge(factors[["cluster", "bimonth", "scalar", "offset"]],
                        on=["cluster", "bimonth"], how="left")
    else:
        return pd.DataFrame(columns=["ID", "month", "obs", "sim", "sim_cor"])

    # Keep only points where factors are available and simulate corrected CF.
    cur = cur[cur["scalar"].notna() & cur["offset"].notna()].copy()
    if len(cur) == 0:
        return pd.DataFrame(columns=["ID", "month", "obs", "sim", "sim_cor"])

    cur["sim_cor"] = cur["sim"] * cur["scalar"] + cur["offset"]
    return cur


# =============================================================================
# PLOT 1 – Cluster analysis (elbow + silhouette)
# =============================================================================
def plot_cluster_analysis(turb_info_train, output_dir):
    """Elbow and silhouette score vs number of clusters."""
    print("\n[1/9] Plotting cluster analysis (this may take a few minutes)...")

    clu_range = list(range(100, 3400, 100))
    sil_scores, sse_scores = [], []

    for k in clu_range:
        kmeans = KMeans(init="random", n_clusters=k, n_init=10, max_iter=300, random_state=42)
        kmeans.fit(turb_info_train[["lat", "lon"]])
        labels = kmeans.predict(turb_info_train[["lat", "lon"]])
        sil_scores.append(silhouette_score(turb_info_train[["lat", "lon"]], labels))
        sse_scores.append(kmeans.inertia_)
        if k % 500 == 0:
            print(f"  k={k} done")

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 6.5 * cm), dpi=STYLE["dpi"])
    tick_list = list(range(100, 3400, 400))

    # SSE (elbow)
    axes[0].plot(clu_range, sse_scores, "o", markersize=2.5)
    axes[0].set_xlabel("Number of Clusters ($n_{clu}$)")
    axes[0].set_ylabel("Sum of Squared Distance (SSE)")
    # Annotate the elbow around k=500 (index 4)
    elbow_idx = 4
    axes[0].annotate(
        "Elbow", xy=(clu_range[elbow_idx], sse_scores[elbow_idx]),
        xytext=(clu_range[elbow_idx] + 200, sse_scores[elbow_idx] + (sse_scores[0] - sse_scores[-1]) * 0.12),
        arrowprops=dict(arrowstyle="->", color="r", lw=1, ls="--"),
    )
    axes[0].set_xticks(tick_list)
    axes[0].set_xticklabels(tick_list, rotation=45, fontsize=6)

    # Silhouette
    axes[1].plot(clu_range, sil_scores, "o", markersize=2.5)
    axes[1].set_xlabel("Number of Clusters ($n_{clu}$)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_xticks(tick_list)
    axes[1].set_xticklabels(tick_list, rotation=45, fontsize=6)

    plt.tight_layout()
    out = output_dir / "cluster_analysis.png"
    format_axes_standard(fig)
    savefig_thesis(fig, out)


# =============================================================================
# PLOT 2 – Offset vs scalar joint plot
# =============================================================================
def plot_offset_vs_scalar(turb_info_train, run_dir, output_dir,
                          num_clu=700, time_res="fixed"):
    """Seaborn joint plot of scalar vs offset for a given cluster config."""
    print("\n[2/9] Plotting offset vs scalar...")

    factors_path = (
        run_dir / "training" / "correction-factors"
        / f"{COUNTRY}_factors_{time_res}_{num_clu}.csv"
    )
    factors = pd.read_csv(factors_path)

    onshore = turb_info_train[turb_info_train["type"] == "onshore"].copy()
    kmeans, df = fit_clusters(num_clu, onshore)
    df = df.merge(factors[["cluster", "scalar", "offset"]], on="cluster")

    g = sns.jointplot(
        x="scalar", y="offset", data=df,
        s=10, marginal_kws=dict(bins=10), color="#471164",
    )
    g.fig.set_figwidth(HALF_WIDTH)
    g.fig.set_figheight(HALF_WIDTH)
    g.set_axis_labels("Scalar", "Offset")
    plt.tight_layout()
    out = output_dir / "offset_vs_scalar.png"
    format_axes_standard(g.fig)
    savefig_thesis(g.fig, out)
    print(f"  Saved: {out}")
    plt.close()

    return kmeans, df


# =============================================================================
# PLOTS 3–4 – Voronoi choropleth maps (scalar, offset)
# =============================================================================
def plot_voronoi_maps(kmeans, df_with_factors, turb_info_train, output_dir):
    """Voronoi maps of scalar and offset clipped to Denmark boundary."""
    print("\n[3/9] Plotting Voronoi correction factor maps...")

    dk_shape = load_dk_shape()
    if dk_shape is None:
        print("  ! Skipping – dk.json not found")
        return

    # Voronoi from cluster centres (KMeans fitted on lat,lon → flip to lon,lat)
    centres_lonlat = kmeans.cluster_centers_[:, ::-1]
    padding = np.array([
        [turb_info_train.lon.min() - 1, turb_info_train.lat.min() - 1],
        [turb_info_train.lon.min() - 1, turb_info_train.lat.max() + 1],
        [turb_info_train.lon.max() + 1, turb_info_train.lat.min() - 1],
        [turb_info_train.lon.max() + 1, turb_info_train.lat.max() + 1],
    ])
    all_points = np.concatenate([centres_lonlat, padding])

    vor = Voronoi(all_points)
    lines = [
        shapely.geometry.LineString(vor.vertices[line])
        for line in vor.ridge_vertices
        if -1 not in line
    ]
    voronoi_polygons = list(shapely.ops.polygonize(lines))

    # Clip to Denmark outline
    dk_union = shapely.ops.unary_union(dk_shape.geometry)
    voronoi_polygons = [p.intersection(dk_union) for p in voronoi_polygons]

    # Assign each Voronoi cell to its cluster ID
    final_polygons, polygon_cluster_ids = [], []
    for p in voronoi_polygons:
        try:
            cx, cy = p.centroid.x, p.centroid.y
            clu = kmeans.predict(pd.DataFrame({"lat": [cy], "lon": [cx]}))[0]
            final_polygons.append(p)
            polygon_cluster_ids.append(clu)
        except Exception:
            pass

    # Dissolve turbine-level factors to one value per cluster
    gdf_pts = gpd.GeoDataFrame(
        df_with_factors,
        geometry=gpd.points_from_xy(df_with_factors["lon"], df_with_factors["lat"], crs="EPSG:4326"),
    ).dissolve("cluster").reset_index()

    gdf_vor = gpd.GeoDataFrame(
        {"cluster": polygon_cluster_ids},
        geometry=final_polygons,
        crs="EPSG:4326",
    )
    maps = gdf_vor.merge(gdf_pts[["scalar", "cluster", "offset"]], on="cluster")

    plot_idx = 3
    for col, midpoint, label in [("scalar", 1.0, "Scalar"), ("offset", 0.0, "Offset")]:
        fig, ax = plt.subplots(figsize=(THIRD_WIDTH, 8 * cm), dpi=STYLE["dpi"])

        vmin, vmax = maps[col].min(), maps[col].max()
        norm = TwoSlopeNorm(vcenter=midpoint, vmin=vmin, vmax=vmax)
        maps.plot(column=col, cmap="RdBu_r", norm=norm, ax=ax, edgecolor="none", linewidth=0.1)
        dk_shape.boundary.plot(ax=ax, edgecolor="black", linewidth=0.3)

        ax.set_axis_off()
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04, label=label)

        plt.tight_layout()
        fname = f"voronoi_{col}_map.png"
        out = output_dir / fname
        format_axes_standard(fig)
        savefig_thesis(fig, out)
        print(f"  [{plot_idx}/9] Saved: {out}")
        plot_idx += 1
        plt.close()


# =============================================================================
# PLOT 5 – Turbine locations (training + validation)
# =============================================================================
def plot_turbine_locations(turb_info_train, turb_info_val, output_dir):
    """Side-by-side map of training and validation turbine locations."""
    print("\n[5/9] Plotting turbine locations...")

    dk_shape = load_dk_shape()
    if dk_shape is not None:
        dk_shape = dk_shape.to_crs("EPSG:3395")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 7 * cm))
    ms, mlw, marker = 5, 0.5, "2"

    # --- Training ---
    onshore = turb_info_train[turb_info_train["type"] == "onshore"]
    offshore = turb_info_train[turb_info_train["type"] == "offshore"]
    if dk_shape is not None:
        dk_shape.plot(ax=ax1, facecolor="gray", edgecolor="gray", linewidth=0.2, alpha=0.2)
    scatter_mercator(ax1, onshore, s=ms, c="#2E86AB", alpha=0.8, label="Onshore",
                     marker=marker, linewidths=mlw)
    scatter_mercator(ax1, offshore, s=ms, c="#A23B72", alpha=0.8, label="Offshore",
                     marker=marker, linewidths=mlw)
    ax1.set_title(f"Training Turbines (2015-2019)\n(n={len(turb_info_train)})")
    ax1.legend(loc="upper right", markerscale=3, frameon=False)
    ax1.set_axis_off()

    # --- Validation ---
    train_ids = set(turb_info_train["ID"])
    val = turb_info_val.copy()
    val["is_new"] = ~val["ID"].isin(train_ids)
    val["offshore"] = val["type"] == "offshore"
    if dk_shape is not None:
        dk_shape.plot(ax=ax2, facecolor="gray", edgecolor="gray", linewidth=0.2, alpha=0.2)

    scatter_mercator(ax2, val[(~val["is_new"]) & (~val["offshore"])],
                     s=ms, c=EXISTING_NEW_COLOURS["Yes"], alpha=0.8,
                     label="Existing turbines (onshore)", marker=marker, linewidths=mlw)
    scatter_mercator(ax2, val[(val["is_new"]) & (~val["offshore"])],
                     s=ms, c=EXISTING_NEW_COLOURS["No"], alpha=0.8,
                     label="New turbines (onshore)", marker=marker, linewidths=mlw)
    scatter_mercator(ax2, val[(~val["is_new"]) & (val["offshore"])],
                     s=ms, c=EXISTING_NEW_COLOURS["Yes"], alpha=0.8,
                     label="Existing turbines (offshore)", marker="^", linewidths=mlw)
    scatter_mercator(ax2, val[(val["is_new"]) & (val["offshore"])],
                     s=ms, c=EXISTING_NEW_COLOURS["No"], alpha=0.8,
                     label="New turbines (offshore)", marker="^", linewidths=mlw)

    n_new = int(val["is_new"].sum())
    ax2.set_title(f"Validation Turbines ({YEAR_TEST})\n(n={len(val)}, new={n_new})")
    ax2.legend(loc="upper right", markerscale=3, frameon=False)
    ax2.set_axis_off()

    plt.tight_layout()
    out = output_dir / "train_turbine_locations.png"
    format_axes_standard(fig)
    savefig_thesis(fig, out)
    print(f"  Saved: {out}")
    plt.close()


# =============================================================================
# PLOT 6 – Temporal slicing scheme
# =============================================================================
def plot_temporal_slicing(output_dir):
    """Diagram of the four temporal slicing schemes."""
    print("\n[6/9] Plotting temporal slicing overview...")

    schemes = {
        "fixed": [list(range(1, 13))],
        "season": [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        "bimonth": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],
        "month": [[i] for i in range(1, 13)],
    }
    palette = ["#2E86AB", "#A23B72", "#06A77D", "#D84A05", "#F18F01", "#C73E1D"]

    fig, axes = plt.subplots(4, 1, figsize=(FULL_WIDTH, 7 * cm))
    for idx, (tres, groups) in enumerate(schemes.items()):
        ax = axes[idx]
        for gi, months in enumerate(groups):
            for m in months:
                ax.barh(0, 1, left=m - 1, height=0.5,
                        color=palette[gi % len(palette)], alpha=0.7,
                        edgecolor="black", linewidth=0.5)
        ax.set_xlim(0, 12)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xticks(np.arange(0.5, 12, 1))
        ax.set_yticks([])
        ax.set_ylabel(TIME_RES_LABELS[tres], rotation=0, ha="right", va="center")
        if idx == 3:
            ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
            ax.set_xlabel("Month")
        else:
            ax.set_xticklabels([])

    plt.suptitle("Temporal Slicing Schemes", fontweight="bold")
    plt.tight_layout()
    out = output_dir / "temporal_slicing.png"
    format_axes_standard(fig)
    savefig_thesis(fig, out)
    print(f"  Saved: {out}")
    plt.close()


# =============================================================================
# PLOT 7 – Observed vs simulated CF at turbine locations
# =============================================================================
def plot_cf_at_turbine_locations(sim_cf, obs_cf, turb_info_train, output_dir):
    """Side-by-side map of mean observed and simulated CF per turbine."""
    print("\n[7/9] Plotting observed vs simulated CF at turbine locations...")

    sim_long = to_long_cf(sim_cf, "simulated_cf")
    obs_long = to_long_cf(obs_cf, "observed_cf")
    sim_long["ID"] = sim_long["ID"].astype(str)
    obs_long["ID"] = obs_long["ID"].astype(str)

    merge_keys = [c for c in ["time", "year", "month", "ID"]
                  if c in sim_long.columns and c in obs_long.columns]
    cf_long = sim_long.merge(obs_long, on=merge_keys, how="inner")

    cf_df = cf_long.groupby("ID", as_index=False)[["observed_cf", "simulated_cf"]].mean()
    coords = turb_info_train[["ID", "lon", "lat"]].copy()
    coords["ID"] = coords["ID"].astype(str)
    cf_df = cf_df.merge(coords, on="ID", how="left").dropna(subset=["lon", "lat"])

    dk_shape = load_dk_shape()
    if dk_shape is not None:
        dk_shape = dk_shape.to_crs("EPSG:3395")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 7 * cm))

    vals = pd.concat([cf_df["observed_cf"], cf_df["simulated_cf"]]).dropna()
    vmin, vmax = float(vals.min()), float(vals.max())
    kw = dict(s=7, cmap="viridis", vmin=vmin, vmax=vmax, linewidths=0, alpha=0.9)

    if dk_shape is not None:
        dk_shape.plot(ax=ax1, facecolor="lightgray", edgecolor="gray", linewidth=0.2, alpha=0.3)
        dk_shape.plot(ax=ax2, facecolor="lightgray", edgecolor="gray", linewidth=0.2, alpha=0.3)

    for ax, col, title in [
        (ax1, "observed_cf", "Observed CF"),
        (ax2, "simulated_cf", "Simulated CF"),
    ]:
        use = cf_df[["lon", "lat", col]].dropna()
        gdf = gpd.GeoDataFrame(
            use, geometry=gpd.points_from_xy(use["lon"], use["lat"]), crs="EPSG:4326",
        ).to_crs(epsg=3395)
        sc = ax.scatter(gdf.geometry.x, gdf.geometry.y, c=gdf[col].to_numpy(), **kw)
        ax.set_title(f"{title}\n(n = {len(cf_df)})")
        ax.set_axis_off()


    cbar = fig.colorbar(
        sc,
        ax=[ax1, ax2],
        orientation="vertical",
        fraction=0.1,
        pad=0.5,
        aspect=10,
        anchor=(5, 0.5),
    )
    cbar.set_label("Capacity factor")

    plt.tight_layout()
    out = output_dir / "observed_vs_simulated_cf.png"
    format_axes_standard(fig)
    savefig_thesis(fig, out)
    print(f"  Saved: {out}")
    plt.close()


# =============================================================================
# PLOTS 8–9 – Training bias scatter + monthly boxplot
# =============================================================================
def plot_train_bias(gen_train, output_dir):
    """Sim vs obs scatter and monthly boxplot for the training period."""

    # --- 8: overall scatter ---
    print("\n[8/9] Plotting training bias scatter...")
    comp = (
        gen_train
        .groupby(["ID", "type"], dropna=False)
        .agg(sim=("sim", "mean"), obs=("obs", "mean"))
        .reset_index()
    )
    comp[["obs", "sim"]] = comp[["obs", "sim"]] * 100

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, 5 * cm), dpi=STYLE["dpi"])
    sns.scatterplot(
        data=comp, x="obs", y="sim",
        hue="type", hue_order=["onshore"],
        palette={"onshore": OKABE_ITO[4], "offshore": OKABE_ITO[5]},
        s=22, marker="o", edgecolor="black", linewidth=0.25, ax=ax,
    )
    ax.set_ylabel("Simulated, %")
    ax.set_xlabel("Observed, %")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    lims = [max(ax.get_xlim()[0], ax.get_ylim()[0]),
            min(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, color=OKABE_ITO[7], lw=STYLE["lw"])
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["Onshore"], frameon=False, title=None)
    sns.despine(ax=ax)
    plt.tight_layout()
    out = output_dir / f"era5_{COUNTRY}_train_bias.png"
    format_axes_standard(fig)
    savefig_thesis(fig, out)
    print(f"  Saved: {out}")
    plt.close()

    # --- 9: monthly boxplot ---
    print("\n[9/9] Plotting monthly distribution...")
    comp_m = (
        gen_train
        .groupby(["ID", "month"])
        .agg(sim=("sim", "mean"), obs=("obs", "mean"))
        .reset_index()
    )
    comp_m = comp_m.melt(id_vars=["ID", "month"], var_name="model", value_name="cf")
    comp_m["cf"] = comp_m["cf"] * 100

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 5.5 * cm), dpi=STYLE["dpi"])
    sns.boxplot(
        data=comp_m, x="month", y="cf",
        hue="model", hue_order=["sim", "obs"],
        palette={"sim": OKABE_ITO[4], "obs": OKABE_ITO[5]},
        showfliers=False, linewidth=STYLE["lw"] * 0.8, ax=ax,
    )
    ax.set_ylabel("Capacity factor, %")
    ax.set_xlabel("Month")
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(bottom=0)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["Simulated", "Observed"], loc="upper right",
              frameon=False, title=None)
    sns.despine(ax=ax)
    plt.tight_layout()
    out = output_dir / f"era5_{COUNTRY}_train_monthly.png"
    format_axes_standard(fig)
    savefig_thesis(fig, out)
    print(f"  Saved: {out}")
    plt.close()


def plot_train_error_vs_clusters(gen_train, turb_info, run_dir, output_dir, country=COUNTRY):
    """Training MAE/RMSE vs cluster count across temporal resolutions."""
    print("\n[10/10] Plotting training error vs clusters...")

    results_dir = run_dir / "results" / "capacity-factor"
    if not results_dir.exists():
        print(f"  ! Results directory not found: {results_dir}")
        return

    factors_dir = run_dir / "training" / "correction-factors"
    if not factors_dir.exists():
        print(f"  ! Correction factors directory not found: {factors_dir}")
        return

    cluster_set = set()
    time_res_set = set()
    for fac_file in factors_dir.glob(f"{country}_factors_*_*.csv"):
        match = re.search(rf"{country}_factors_(\w+)_(\d+)\.csv", fac_file.name)
        if match:
            time_res, n_clusters = match.groups()
            cluster_set.add(int(n_clusters))
            time_res_set.add(time_res)

    if not cluster_set or not time_res_set:
        print("  ! No training corrected files found")
        return

    cluster_list = sorted(cluster_set)
    time_res_list = sorted(time_res_set, key=lambda x: TIME_RES_ORDER.get(x, 99))

    if ("year" not in gen_train.columns or "month" not in gen_train.columns) and "time" in gen_train.columns:
        gen_train = gen_train.copy()
        gen_train["time"] = pd.to_datetime(gen_train["time"])
        if "year" not in gen_train.columns:
            gen_train["year"] = gen_train["time"].dt.year
        if "month" not in gen_train.columns:
            gen_train["month"] = gen_train["time"].dt.month
    obs_long = gen_train[["ID", "year", "month", "obs"]].copy()
    obs_long["ID"] = obs_long["ID"].astype(str)

    onshore_info = turb_info[turb_info["type"] == "onshore"].copy()
    onshore_info["ID"] = onshore_info["ID"].astype(str)
    obs_long = obs_long[obs_long["ID"].isin(set(onshore_info["ID"]))].copy()

    print("  Loading ERA5 training wind fields for correction simulation...")
    reanalysis = prep_era5(country, True, True)
    if "year" in obs_long.columns:
        y0, y1 = int(obs_long["year"].min()), int(obs_long["year"].max())
        reanalysis = reanalysis.sel(time=slice(str(y0), str(y1)))

    power_curves = load_power_curves()

    print("  Interpolating uncorrected ERA5 wind speeds to training turbines...")
    unc_ws = wind.interpolate_wind(reanalysis, onshore_info)

    _, curve_by_model = wind._get_power_curve_cache(power_curves)

    def ws_to_cf_df(cor_ws):
        """Convert corrected wind-speed DataArray to long monthly CF DataFrame."""
        ws_vals = cor_ws.values
        models = np.asarray(cor_ws.model.values)
        cf_vals = np.empty_like(ws_vals, dtype=float)
        for model_name in np.unique(models):
            mask = models == model_name
            cf_vals[:, mask] = curve_by_model[model_name](ws_vals[:, mask])

        cor_cf = pd.DataFrame(cf_vals, index=cor_ws.time.values, columns=cor_ws.turbine.values)
        cor_cf.index.name = "time"
        cor_cf = cor_cf.reset_index()
        cor_cf["time"] = pd.to_datetime(cor_cf["time"])
        cor_cf = cor_cf.groupby(pd.Grouper(key="time", freq="ME")).mean().reset_index()
        cor_cf = cor_cf.melt(id_vars=["time"], var_name="ID", value_name="sim_cor")
        cor_cf["ID"] = cor_cf["ID"].astype(str)
        cor_cf["year"] = cor_cf["time"].dt.year.astype(int)
        cor_cf["month"] = cor_cf["time"].dt.month.astype(int)
        return cor_cf[["ID", "year", "month", "sim_cor"]]

    rows = []
    for num_clu in cluster_list:
        clus_info = cluster_turbines(num_clu, onshore_info.copy(), True)

        for tres in time_res_list:
            fac_path = factors_dir / f"{country}_factors_{tres}_{num_clu}.csv"
            if not fac_path.exists():
                continue
            factors = pd.read_csv(fac_path)

            cor_ws = wind.correct_wind_speed(unc_ws.copy(), tres, factors, clus_info)
            cor_cf_long = ws_to_cf_df(cor_ws)
            cur = cor_cf_long.merge(obs_long, on=["ID", "year", "month"], how="inner")
            if len(cur) == 0:
                continue

            diff = cur["sim_cor"] - cur["obs"]
            rows.append({
                "num_clu": num_clu,
                "time_res": tres,
                "mae": float(np.abs(diff).mean()),
                "rmse": float(np.sqrt((diff ** 2).mean())),
            })

    corrected = pd.DataFrame(rows)

    if len(corrected) == 0:
        print("  ! No corrected training metrics available")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(HALF_WIDTH, 6.8 * cm), dpi=STYLE["dpi"])

    for tres in time_res_list:
        subset = corrected[corrected["time_res"] == tres].sort_values("num_clu")
        if len(subset) == 0:
            continue
        colour = TIME_RES_COLOURS.get(tres, "#999999")
        label = TIME_RES_LABELS.get(tres, tres)
        ax1.plot(subset["num_clu"], subset["mae"], "o-",
                 color=colour, linewidth=STYLE["lw"], markersize=3,
                 label=label, alpha=0.85)
        ax2.plot(subset["num_clu"], subset["rmse"], "o-",
                 color=colour, linewidth=STYLE["lw"], markersize=3,
                 label=label, alpha=0.85)

    for ax, metric in [(ax1, "MAE"), (ax2, "RMSE")]:
        ax.set_xscale("log")
        ax.set_xlabel(r"Number of Clusters ($n_{\mathrm{clu}}$)")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(labels),
        title=r"Temporal Frequency ($t_{freq}$)",
        frameon=False,
    )

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    out = output_dir / f"era5_{country}_train_error_vs_clusters.png"
    format_axes_standard(fig)
    savefig_thesis(fig, out)
    print(f"  Saved: {out}")
    plt.close()


def load_sensitivity_metrics(runs_dir):
    """Load sensitivity experiment metrics into a standard plotting structure."""
    runs_dir = Path(runs_dir)

    def _to_plot_df(df):
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["time_res", "n_clu", "rmse", "mae"])
        use = df.copy()
        if "n_clusters" in use.columns:
            use = use.rename(columns={"n_clusters": "n_clu"})
        use = use[
            (use["country"] == COUNTRY)
            & (use["mode"] == "onshore")
            & (use["correction_type"] == "corrected")
            & (use["time_res"].notna())
            & (use["n_clu"].notna())
            & (use["time_res"].astype(str).str.lower() != "uncorrected")
        ].copy()
        use["n_clu"] = use["n_clu"].astype(int)
        use = use[use["n_clu"] > 1]
        return use[["time_res", "n_clu", "rmse", "mae"]]

    sensitivity_map = {
        "missing_30pct": {
            "path": runs_dir / "sensitivity_missing_30pct" / "pyvwf_evaluation_metrics.csv",
            "label": "(a) 30% missing",
        },
        "missing_50pct": {
            "path": runs_dir / "sensitivity_missing_50pct" / "pyvwf_evaluation_metrics.csv",
            "label": "(b) 50% missing",
        },
        "fix_train_ge15se": {
            "path": runs_dir / "sensitivity_fix_train_ge15se" / "pyvwf_evaluation_metrics.csv",
            "label": "(c) Train GE.1.5se",
        },
        "fix_train_vestas": {
            "path": runs_dir / "sensitivity_fix_train_vestas_v66" / "pyvwf_evaluation_metrics.csv",
            "label": "(d) Train Vestas.V66.2000",
        },
        "fix_test_ge15se": {
            "path": runs_dir / "sensitivity_fix_test_ge15se" / "pyvwf_evaluation_metrics.csv",
            "label": "(e) Val GE.1.5se",
        },
        "fix_test_vestas": {
            "path": runs_dir / "sensitivity_fix_test_vestas_v66" / "pyvwf_evaluation_metrics.csv",
            "label": "(f) Val Vestas.V66.2000",
        },
    }

    out = {}

    # Standard model: evaluate directly from the baseline run
    try:
        std_rows = evaluate_run(Path(RUN_DIR), metric_type="total")
        std_df = _to_plot_df(pd.DataFrame(std_rows))
        if len(std_df) > 0:
            out["standard"] = {"data": std_df, "label": "Standard model"}
    except Exception as exc:
        print(f"  ! Could not load standard sensitivity baseline: {exc}")

    for key, cfg in sensitivity_map.items():
        csv_path = cfg["path"]
        if not csv_path.exists():
            continue
        try:
            raw = pd.read_csv(csv_path)
            plot_df = _to_plot_df(raw)
            if len(plot_df) > 0:
                out[key] = {"data": plot_df, "label": cfg["label"]}
        except Exception as exc:
            print(f"  ! Failed loading sensitivity file {csv_path}: {exc}")

    return out

def plot_sensitivity_analysis(data, output_dir, runs_dir):
    """Figure 10: Sensitivity analysis of data quality on model performance.

    Exports one file per scenario plus a standalone legend figure,
    so panels can be assembled in LaTeX with subcaptions.
    """
    print("\n[10/11] Plotting sensitivity analysis...")

    # Load sensitivity metrics from run directories
    print("  Loading sensitivity metrics...")
    sensitivity_data = load_sensitivity_metrics(runs_dir)

    if len(sensitivity_data) == 0:
        print("  ! No sensitivity data found, skipping")
        return

    scenario_order = [
        "standard",
        "missing_30pct",
        "missing_50pct",
        "fix_train_ge15se",
        "fix_train_vestas",
        "fix_test_ge15se",
        "fix_test_vestas",
    ]

    # Determine consistent y-axis limits across all scenarios
    all_rmse = []
    for key in scenario_order:
        if key in sensitivity_data:
            all_rmse.extend(sensitivity_data[key]['data']['rmse'].dropna().tolist())
    if all_rmse:
        y_min = max(0, min(all_rmse) - 0.005)
        y_max = max(all_rmse) + 0.005
    else:
        y_min, y_max = 0.08, 0.17

    # Filter standard model to sensitivity cluster range for comparison consistency
    sensitivity_clusters = [1, 10, 100, 500, 1000]
    if 'standard' in sensitivity_data:
        std_df = sensitivity_data['standard']['data']
        sensitivity_data['standard']['data'] = std_df[std_df['n_clu'].isin(sensitivity_clusters)]

    sensitivity_dir = output_dir / "sensitivity_analysis"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)

    panel_names = {
        "standard": "original",
        "missing_30pct": "miss30",
        "missing_50pct": "miss50",
        "fix_train_ge15se": "trainge",
        "fix_train_vestas": "trainvestas",
        "fix_test_ge15se": "testge",
        "fix_test_vestas": "testvestas",
    }

    for key in scenario_order:
        fig, ax = plt.subplots(1, 1, figsize=(THIRD_WIDTH, 5.0 * cm), dpi=STYLE["dpi"])

        if key not in sensitivity_data:
            ax.text(
                0.5,
                0.5,
                "Data not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=7,
            )
            ax.set_xscale("log")
            ax.set_xlim(0.8, 1200)
        else:
            df = sensitivity_data[key]["data"]

            for tres in ordered_time_res():
                subset = df[df["time_res"] == tres].sort_values("n_clu")
                if len(subset) > 0:
                    ax.plot(
                        subset["n_clu"],
                        subset["rmse"],
                        color=TIME_RES_COLOURS.get(tres, "#999999"),
                        linestyle=TIME_RES_LINESTYLES.get(tres, "-"),
                        linewidth=STYLE["lw"],
                        markersize=2,
                        marker="o",
                        label=TIME_RES_LABELS.get(tres, tres),
                    )

            ax.set_xscale("log")

        ax.set_xlabel(r"Number of Clusters ($n_{\mathrm{clu}}$)")
        ax.set_ylabel("RMSE")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        plt.tight_layout()
        panel_name = panel_names.get(key, key)
        panel_path = sensitivity_dir / f"{COUNTRY}_{panel_name}_test_rmse.png"
        format_axes_standard(fig)
        savefig_thesis(fig, panel_path)
        print(f"  Saved: {panel_path}")
        plt.close(fig)

    legend_fig, legend_ax = plt.subplots(1, 1, figsize=(THIRD_WIDTH, 2.2 * cm), dpi=STYLE["dpi"])
    legend_ax.set_axis_off()
    lines = []
    labels = []
    for tres in ordered_time_res():
        line, = legend_ax.plot(
            [],
            [],
            color=TIME_RES_COLOURS.get(tres, "#999999"),
            linestyle=TIME_RES_LINESTYLES.get(tres, "-"),
            linewidth=STYLE["lw"] * 1.5,
            marker="o",
            markersize=2.5,
        )
        lines.append(line)
        labels.append(TIME_RES_LABELS.get(tres, tres))

    legend_ax.legend(
        lines,
        labels,
        loc="center",
        frameon=False,
        ncol=1,
        title=r"Temporal Frequency ($t_{freq}$)",
    )

    plt.tight_layout()
    legend_path = sensitivity_dir / f"{COUNTRY}_legend_test_rmse.png"
    savefig_thesis(legend_fig, legend_path)
    print(f"  Saved: {legend_path}")
    plt.close(legend_fig)

def plot_error_vs_clusters(run_dir, turb_info, output_dir,
                           country=COUNTRY, year_test=YEAR_TEST):
    """MAE and RMSE vs number of clusters for overall, temporal and spatial modes.

        Uses scripts.evaluate_all_pyvwf_runs.evaluate_run for three error types:
            - 'total'           (overall)
            - 'temporal-focus'  (temporal)
            - 'spatial-focus'   (spatial)
        and produces one plot per mode.
    """
    print("\n[7] Plotting error vs clusters (overall / temporal / spatial)...")

    results_dir = run_dir / "results" / "capacity-factor"
    if not results_dir.exists():
        print(f"  ! Results directory not found: {results_dir}")
        return

    modes = [
        ("total", "overall", "Overall"),
        ("temporal-focus", "temporal", "Temporal"),
        ("spatial-focus", "spatial", "Spatial"),
    ]

    for metric_type, mode_slug, mode_label in modes:
        print(f"  Computing {mode_label} metrics via evaluator...")
        try:
            rows = evaluate_run(Path(run_dir), metric_type=metric_type)
        except Exception as e:
            print(f"  ! Error evaluating {mode_label} metrics: {e}")
            continue

        summary = pd.DataFrame(rows)
        if len(summary) == 0:
            print(f"  ! No results returned for {mode_label}")
            continue

        # DK onshore corrected rows only
        corrected = summary[
            (summary["country"] == country)
            & (summary["mode"] == "onshore")
            & (summary["correction_type"] == "corrected")
            & (summary["time_res"].notna())
            & (summary["n_clusters"].notna())
            & (summary["time_res"].astype(str).str.lower() != "uncorrected")
        ].copy()
        corrected["n_clu"] = corrected["n_clusters"].astype(int)
        corrected = corrected[corrected["n_clu"] > 1]

        if len(corrected) == 0:
            print(f"  ! No corrected results for {mode_label}")
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(HALF_WIDTH, 6.5 * cm), dpi=STYLE["dpi"])

        time_res_vals = sorted(
            corrected["time_res"].unique(),
            key=lambda x: TIME_RES_ORDER.get(x, 99),
        )

        for tres in time_res_vals:
            subset = corrected[corrected["time_res"] == tres].sort_values("n_clu")
            colour = TIME_RES_COLOURS.get(tres, "#999999")
            label = TIME_RES_LABELS.get(tres, tres)
            line_style = TIME_RES_LINESTYLES.get(tres, "-")

            ax1.plot(subset["n_clu"], subset["rmse"],
                     marker="o", linestyle=line_style,
                     color=colour, linewidth=STYLE["lw"], markersize=3,
                     label=label, alpha=0.85)
            ax2.plot(subset["n_clu"], subset["mae"],
                     marker="o", linestyle=line_style,
                     color=colour, linewidth=STYLE["lw"], markersize=3,
                     label=label, alpha=0.85)

        for ax, metric in [(ax1, "RMSE"), (ax2, "MAE")]:
            ax.set_xscale("log")
            ax.set_xlabel(r"Number of Clusters ($n_{\mathrm{clu}}$)")
            ax.set_ylabel(metric)
            # ax.legend(fontsize=6, frameon=False)
            ax.grid(True, alpha=0.3, linewidth=0.5)

        plt.tight_layout()
        fname = f"error_vs_clusters_{mode_slug}.png"
        output_path = output_dir / fname
        format_axes_standard(fig)
        savefig_thesis(fig, output_path)
        print(f"  Saved: {output_path}")
        plt.close()


def plot_spatial_split_error_vs_clusters(
    run_dir,
    turb_info,
    output_dir,
    country=COUNTRY,
    year_test=YEAR_TEST,
    cluster_list=None,
    time_res_list=None,
):
    """Spatial RMSE vs clusters split by all / existing / new turbines."""
    print("\n[8] Plotting spatial split error vs clusters (all/existing/new)...")

    if cluster_list is None:
        cluster_list = [1, 2, 5, 7, 10, 20, 50, 70, 100, 200, 500, 700, 800, 900, 1000, 2000, 3000, 3300]
    if time_res_list is None:
        time_res_list = ["fixed", "season", "bimonth", "month"]

    if "In training?" not in turb_info.columns:
        print("  ! Missing 'In training?' column in turbine metadata; skipping spatial split plot")
        return

    spatial_metrics = overall_error(
        "spatial-focus", str(run_dir), country, turb_info.copy(),
        cluster_list, time_res_list, False, year_test
    )

    turb_old = turb_info.loc[turb_info["In training?"] == "Yes"].copy().reset_index(drop=True)
    total_old_metrics = overall_error(
        "spatial-focus", str(run_dir), country, turb_old,
        cluster_list, time_res_list, False, year_test
    )

    turb_new = turb_info.loc[turb_info["In training?"] == "No"].copy().reset_index(drop=True)
    total_new_metrics = overall_error(
        "spatial-focus", str(run_dir), country, turb_new,
        cluster_list, time_res_list, False, year_test
    )

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 6.2 * cm), dpi=STYLE["dpi"])

    plot_specs = [
        (spatial_metrics, "All turbines", axes[0], False),
        (total_old_metrics, "Existing turbines", axes[1], False),
        (total_new_metrics, "New turbines", axes[2], True),
    ]

    for df_metrics, title, ax, show_legend in plot_specs:
        df_plot = df_metrics[df_metrics["time_res"] != "uncorrected"].copy()
        for tres in sorted(time_res_list, key=lambda x: TIME_RES_ORDER.get(x, 99), reverse=True):
            subset = df_plot[df_plot["time_res"] == tres].sort_values("num_clu")
            if len(subset) == 0:
                continue
            ax.plot(
                subset["num_clu"],
                subset["rmse"],
                marker="o",
                linestyle=TIME_RES_LINESTYLES.get(tres, "-"),
                color=TIME_RES_COLOURS.get(tres, "#999999"),
                label=TIME_RES_LABELS.get(tres, tres),
                linewidth=STYLE["lw"],
                markersize=2,
                alpha=0.9,
            )

        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_ylabel("RMSE")
        ax.set_xlabel(r"Number of Clusters ($n_{clu}$)")
        ax.set_xticks([1, 10, 100, 1000])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, alpha=0.3, linewidth=0.5)
        if not show_legend and ax.get_legend() is not None:
            ax.get_legend().remove()

    handles, labels = axes[2].get_legend_handles_labels()
    if axes[2].get_legend() is not None:
        axes[2].get_legend().remove()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=4,
        title=r"Temporal Frequency ($t_{freq}$)",
        frameon=False,
    )

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    out = output_dir / f"{country}_spatial_separate_test_error.png"
    format_axes_standard(fig)
    savefig_thesis(fig, out)
    print(f"  Saved: {out}")
    plt.close()


def plot_spatial_error_correlation(
    run_dir,
    turb_info_train,
    turb_info_val,
    output_dir,
    country=COUNTRY,
    year_test=YEAR_TEST,
    regions=8,
    number_clu=2,
    time_res="fixed",
):
    """Correlate turbine-level absolute MBE with feature mismatch and distance."""
    print("\n[9] Plotting spatial error correlation...")

    def _safe_pct_diff(a, b):
        denom = (np.abs(a) + np.abs(b)) / 2.0
        out = np.where(denom > 0, (np.abs(a - b) / denom) * 100.0, np.nan)
        return out

    def _assign_clusters_with_distance(train_df, target_df, n_clu):
        kmeans = KMeans(
            init="random", n_clusters=n_clu,
            n_init=10, max_iter=300, random_state=42,
        )
        kmeans.fit(train_df[["lat", "lon"]])

        train_out = train_df.copy()
        train_out["cluster"] = kmeans.predict(train_df[["lat", "lon"]])

        test_out = target_df.copy()
        test_out["cluster"] = kmeans.predict(target_df[["lat", "lon"]])
        test_out["distance"] = kmeans.transform(target_df[["lat", "lon"]]).min(axis=1)
        return train_out, test_out

    def _regional_info(n_regions, turb_info, turb_base):
        kmeans = KMeans(
            init="random", n_clusters=n_regions,
            n_init=10, max_iter=300, random_state=42,
        )
        kmeans.fit(turb_base[["lat", "lon"]])
        out = turb_info.copy()
        out["region_delete"] = kmeans.predict(out[["lat", "lon"]])

        reg = out.groupby("region_delete", as_index=False)[["lat", "lon"]].mean()
        reg = reg.sort_values("lon").reset_index(drop=True)
        reg["region"] = reg.index

        out = out.merge(reg[["region_delete", "region"]], on="region_delete", how="left")
        out = out.drop(columns=["region_delete"])
        return out

    obs_path = run_dir / "results" / "capacity-factor" / f"{country}_{year_test}_obs_cf.csv"
    cor_path = run_dir / "results" / "capacity-factor" / f"{country}_{year_test}_{time_res}_{number_clu}_cor_cf.csv"
    if not obs_path.exists() or not cor_path.exists():
        print(f"  ! Missing required files for spatial correlation: {obs_path.name}, {cor_path.name}")
        return

    turb_train = turb_info_train.copy()
    turb_test = turb_info_val.copy()
    turb_train["ID"] = turb_train["ID"].astype(str)
    turb_test["ID"] = turb_test["ID"].astype(str)

    obs_cf = pd.read_csv(obs_path, parse_dates=["time"])
    cor_cf = pd.read_csv(cor_path, parse_dates=["time"])

    obs_long = to_long_cf(obs_cf, "obs")
    sim_long = to_long_cf(cor_cf, "sim")
    for df in (obs_long, sim_long):
        df["ID"] = df["ID"].astype(str)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            if "year" not in df.columns:
                df["year"] = df["time"].dt.year
            if "month" not in df.columns:
                df["month"] = df["time"].dt.month

    merge_keys = [c for c in ["ID", "year", "month"] if c in obs_long.columns and c in sim_long.columns]
    merged = sim_long.merge(obs_long, on=merge_keys, how="inner")
    if len(merged) == 0:
        print("  ! No overlapping corrected/observed points for spatial correlation")
        return

    clus_info_train, clus_info_test = _assign_clusters_with_distance(turb_train, turb_test, number_clu)
    clus_info_test["ID"] = clus_info_test["ID"].astype(str)

    turb_error_test = (
        merged.groupby("ID", as_index=False)
        .agg(diff=("sim", lambda s: (s - merged.loc[s.index, "obs"]).mean()))
    )
    turb_error_test = turb_error_test.merge(
        clus_info_test[["ID", "distance"]], on="ID", how="left"
    )
    turb_error_test["turb_err"] = np.abs(turb_error_test["diff"])

    region_info_test = _regional_info(regions, turb_test.copy(), turb_train.copy())
    region_info_test["ID"] = region_info_test["ID"].astype(str)

    plot_error = turb_error_test.merge(
        region_info_test[["ID", "capacity", "region"]], on="ID", how="left"
    )
    region_sum_cap = region_info_test.groupby("region", as_index=False)["capacity"].sum()
    region_sum_cap = region_sum_cap.rename(columns={"capacity": "max_capacity_region"})
    plot_error = plot_error.merge(region_sum_cap, on="region", how="left")

    numeric_cols = [c for c in ["capacity", "height", "p_density", "diameter"] if c in clus_info_train.columns]
    if len(numeric_cols) == 0:
        print("  ! Missing turbine feature columns for spatial correlation plot")
        return

    weighted_clusters = clus_info_train.groupby("cluster", as_index=False)[numeric_cols].mean()
    rename_map = {
        "capacity": "clu_capacity",
        "height": "clu_height",
        "p_density": "clu_density",
        "diameter": "clu_diameter",
    }
    weighted_clusters = weighted_clusters.rename(columns={k: v for k, v in rename_map.items() if k in weighted_clusters.columns})

    merge_cols = [c for c in ["ID", "cluster", "lat", "lon", "In training?", "height", "p_density", "diameter"] if c in clus_info_test.columns]
    plot_error = plot_error.merge(clus_info_test[merge_cols], on="ID", how="left")
    plot_error = plot_error.merge(weighted_clusters, on="cluster", how="left")

    capacity_col = None
    for candidate in ["capacity", "capacity_x", "capacity_y"]:
        if candidate in plot_error.columns:
            capacity_col = candidate
            break

    if capacity_col is not None and "clu_capacity" in plot_error.columns:
        plot_error["err_capacity"] = _safe_pct_diff(plot_error["clu_capacity"], plot_error[capacity_col])
    if {"height", "clu_height"}.issubset(plot_error.columns):
        plot_error["err_height"] = _safe_pct_diff(plot_error["clu_height"], plot_error["height"])
    if {"p_density", "clu_density"}.issubset(plot_error.columns):
        plot_error["err_density"] = _safe_pct_diff(plot_error["clu_density"], plot_error["p_density"])
    if {"diameter", "clu_diameter"}.issubset(plot_error.columns):
        plot_error["err_diameter"] = _safe_pct_diff(plot_error["clu_diameter"], plot_error["diameter"])

    feature_order = [c for c in ["err_capacity", "err_diameter", "err_height", "distance"] if c in plot_error.columns]
    if len(feature_order) == 0:
        print("  ! No feature mismatch columns available for plotting")
        return

    plot_melt = plot_error[["ID", "turb_err", "In training?"] + feature_order].melt(
        id_vars=["ID", "turb_err", "In training?"],
        var_name="error_type",
        value_name="percs",
    ).dropna(subset=["percs", "turb_err"])

    if len(plot_melt) == 0:
        print("  ! No valid points to plot for spatial error correlation")
        return

    g = sns.lmplot(
        data=plot_melt,
        x="percs",
        y="turb_err",
        hue="In training?",
        hue_order=["Yes", "No"],
        palette=EXISTING_NEW_COLOURS,
        col="error_type",
        col_order=feature_order,
        col_wrap=2,
        markers="o",
        truncate=True,
        facet_kws=dict(sharex=False, sharey=True),
        line_kws=dict(
            linewidth=2,
            path_effects=[pe.withStroke(linewidth=3, foreground="black")],
        ),
        scatter_kws=dict(linewidths=0.1, edgecolor="black", s=18),
        height=2.1,
        aspect=2,
        n_boot=1000,
    )

    # --- Increase font sizes for readability at this figure size ---
    _fs_label = 12
    _fs_tick = 10
    _fs_annot = 11
    _fs_legend = 12

    g.axes.flat[0].set_ylabel("Absolute of the MBE", fontsize=_fs_label)
    if len(g.axes.flat) > 2:
        g.axes.flat[2].set_ylabel("Absolute of the MBE", fontsize=_fs_label)
    xlabel_map = {
        "err_capacity": "Capacity Diff, %",
        "err_diameter": "Diameter Diff, %",
        "err_height": "Height Diff, %",
        "distance": "Euclidean Distance, °",
    }
    for ax, feat in zip(g.axes.flat, g.col_names):
        ax.set_xlabel(xlabel_map.get(feat, feat), fontsize=_fs_label)
        ax.tick_params(labelsize=_fs_tick)
    g.set_titles(col_template="")

    if g._legend is not None:
        g._legend.set_title(None)
        if len(g._legend.texts) >= 2:
            g._legend.texts[0].set_text(EXISTING_NEW_LABELS["Yes"])
            g._legend.texts[1].set_text(EXISTING_NEW_LABELS["No"])
    sns.move_legend(g, "center right", bbox_to_anchor=(1.15, 0.5), frameon=False,
                    fontsize=_fs_legend, markerscale=2.5)

    for ax, feature in zip(g.axes.flat, g.col_names):
        feat_points = plot_melt[plot_melt["error_type"] == feature]
        exist = feat_points[feat_points["In training?"] == "Yes"][["percs", "turb_err"]]
        new = feat_points[feat_points["In training?"] == "No"][["percs", "turb_err"]]

        r2_exist = np.nan
        r2_new = np.nan
        if len(exist) > 1:
            _, _, r_val, _, _ = sp.stats.linregress(exist["percs"], exist["turb_err"])
            r2_exist = r_val ** 2
        if len(new) > 1:
            _, _, r_val2, _, _ = sp.stats.linregress(new["percs"], new["turb_err"])
            r2_new = r_val2 ** 2

        ax.text(
            0.35,
            0.7,
            f"$R_e^2$={r2_exist:.2f}\n$R_n^2$={r2_new:.2f}",
            transform=ax.transAxes,
            fontsize=_fs_annot,
        )

    out = output_dir / f"{country}_spatial_error_correlation.png"
    format_axes_standard(g.fig)
    savefig_thesis(g.fig, out)
    print(f"  Saved: {out}")
    plt.close(g.fig)


def regional_info(regions, turb_info, turb_base, remove=None):
    """Assign ordered regional labels using KMeans on lat/lon."""
    kmeans = KMeans(
        init="random",
        n_clusters=regions,
        n_init=10,
        max_iter=300,
        random_state=42,
    )
    kmeans.fit(turb_base[["lat", "lon"]])

    turb_info = turb_info.copy()
    turb_info["region_delete"] = kmeans.predict(turb_info[["lat", "lon"]])

    region_df = turb_info.groupby("region_delete", as_index=False)[["lat", "lon"]].mean()
    region_df = region_df.sort_values("lon").reset_index(drop=True)
    region_df["region"] = region_df.index
    turb_info = turb_info.merge(region_df[["region_delete", "region"]], on="region_delete", how="left")
    turb_info = turb_info.drop("region_delete", axis=1)

    if remove == "new_offshore":
        turb_info = turb_info.drop(
            turb_info[
                (turb_info["type"] == "offshore")
                & (turb_info["In training?"] == "No")
            ].index
        ).reset_index(drop=True)
    elif remove == "all_offshore":
        turb_info = turb_info.drop(
            turb_info[(turb_info["type"] == "offshore")].index
        ).reset_index(drop=True)

    return turb_info


def _regional_error_metrics(run_dir, country, turb_info, cluster_list, time_res_list, year_test):
    """Build regional error table similar to legacy uncor_error('regional-error')."""
    obs_cf = pd.read_csv(
        run_dir / "results" / "capacity-factor" / f"{country}_{year_test}_obs_cf.csv",
        parse_dates=["time"],
    )
    unc_cf = pd.read_csv(
        run_dir / "results" / "capacity-factor" / f"{country}_{year_test}_unc_cf.csv",
        parse_dates=["time"],
    )

    all_rows = []

    unc = calculate_error("regional-error", unc_cf, obs_cf, turb_info, train=False).reset_index()
    unc = unc.rename(columns={"index": "region"})
    unc["num_clu"] = 1
    unc["time_res"] = "uncorrected"
    all_rows.append(unc)

    for num_clu in cluster_list:
        for time_res in time_res_list:
            cor_path = run_dir / "results" / "capacity-factor" / f"{country}_{year_test}_{time_res}_{num_clu}_cor_cf.csv"
            if not cor_path.exists():
                continue
            cor_cf = pd.read_csv(cor_path, parse_dates=["time"])
            cur = calculate_error("regional-error", cor_cf, obs_cf, turb_info, train=False).reset_index()
            cur = cur.rename(columns={"index": "region"})
            cur["num_clu"] = num_clu
            cur["time_res"] = time_res
            all_rows.append(cur)

    if not all_rows:
        return pd.DataFrame(columns=["In training?", "region", "ID", "num_clu", "time_res", "cf_obs", "cf_sim", "diff"])

    df_metrics = pd.concat(all_rows, ignore_index=True)
    return df_metrics


def plot_regional_error(run_dir, country, df_metrics):
    """Plot regional error by region for selected cluster counts (fixed temporal mode)."""
    base_clu = 1
    choice_clu = 1
    choice_clu2 = 700
    choice_clu3 = 3300

    fig, axes = plt.subplots(1, figsize=(FULL_WIDTH, 4.5 * cm), sharey="all", dpi=STYLE["dpi"])
    subset = df_metrics[
        (df_metrics["time_res"] == "fixed")
        & (df_metrics["In training?"] == "Both")
        & (
            (df_metrics["num_clu"] == base_clu)
            | (df_metrics["num_clu"] == choice_clu)
            | (df_metrics["num_clu"] == choice_clu2)
            | (df_metrics["num_clu"] == choice_clu3)
        )
    ].copy()

    if len(subset) == 0:
        print("  ! No regional error rows available for fixed mode")
        plt.close(fig)
        return

    sns.lineplot(
        x="region",
        y="diff",
        data=subset,
        hue="num_clu",
        style="num_clu",
        palette="tab10",
        ax=axes,
        legend=True,
    )

    axes.set_xticks(range(0, 8))
    axes.set_ylabel("MBE")
    axes.set_xlabel("Region")

    axes.legend(
        title="No. of Clusters\n($n_{clu}$)",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
    )

    plt.tight_layout(rect=[0.0, 0.0, 0.88, 1.0])
    out = run_dir / "plots" / f"{country}_regional_difference.png"
    format_axes_standard(fig)
    savefig_thesis(fig, out)
    print(f"  Saved: {out}")
    plt.close(fig)


def plot_regional_map(region_info_test, output_dir):
    """Regional label map styled consistently with other matplotlib figures."""
    if len(region_info_test) == 0:
        print("  ! Empty region_info input, skipping regional map")
        return

    df = region_info_test.copy()
    df["region"] = df["region"].astype(str)

    dk_shape = load_dk_shape()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 6 * cm), dpi=STYLE["dpi"])

    if dk_shape is not None:
        dk_merc = dk_shape.to_crs("EPSG:3395")
        dk_merc.plot(ax=ax, facecolor="gray", edgecolor="gray", linewidth=0.2, alpha=0.2)

    gdf_pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3395)

    region_values = sorted(gdf_pts["region"].unique(), key=lambda x: int(x))
    cmap = plt.get_cmap("tab10")
    colour_map = {reg: cmap(i % 10) for i, reg in enumerate(region_values)}

    for reg in region_values:
        subset = gdf_pts[gdf_pts["region"] == reg]
        ax.scatter(
            subset.geometry.x,
            subset.geometry.y,
            s=5,
            color=colour_map[reg],
            alpha=0.9,
            label=reg,
            linewidths=0.0,
        )

    ax.set_axis_off()
    ax.legend(
        title="Region",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        ncol=1,
    )

    plt.tight_layout(rect=[0.0, 0.0, 0.88, 1.0])
    out = output_dir / f"{COUNTRY}_regional_labels.png"
    format_axes_standard(fig)
    savefig_thesis(fig, out)
    print(f"  Saved: {out}")
    plt.close(fig)


def run_regional_analysis(run_dir, country, turb_info, turb_info_train, output_dir, year_test=YEAR_TEST):
    """Generate regional label map and regional error difference plot."""
    print("\n[10] Plotting regional analysis (labels + regional difference)...")

    region_info_test = regional_info(
        8,
        turb_info[turb_info["type"] == "onshore"].copy(),
        turb_info_train[turb_info_train["type"] == "onshore"].copy(),
    )
    region_info_test["region"] = region_info_test["region"].astype(str)
    print(region_info_test.region.value_counts())

    plot_regional_map(region_info_test, output_dir)

    region_error = _regional_error_metrics(
        run_dir,
        country,
        region_info_test,
        [1, 700, 3300],
        ["fixed"],
        year_test,
    )
    plot_regional_error(run_dir, country, region_error)


def wavg_cf_monthly(df, turb_info):
    """Calculate weighted monthly capacity factor by turbine capacity."""
    df = df.groupby(pd.Grouper(key="time", freq="ME")).mean(numeric_only=True).reset_index()
    df = df.melt(id_vars=["time"], var_name="ID", value_name="cf")

    df["ID"] = df["ID"].astype(str)
    turb_info = turb_info.copy()
    turb_info["ID"] = turb_info["ID"].astype(str)
    df = pd.merge(df, turb_info[["ID", "capacity"]], on=["ID"], how="left")

    def weighted_avg(group, values, weights):
        return (group[values] * group[weights]).sum() / group[weights].sum()

    wavg = lambda x: weighted_avg(df.loc[x.index], "cf", "capacity")
    df = df.groupby(pd.Grouper(key="time", freq="ME")).agg({"cf": wavg}).reset_index()
    return df


def wavg_cf_monthly_train(df, turb_info):
    """Calculate weighted monthly CF from long-format monthly table."""
    df = df.copy()
    df["ID"] = df["ID"].astype(str)
    turb_info = turb_info.copy()
    turb_info["ID"] = turb_info["ID"].astype(str)
    df = pd.merge(df, turb_info[["ID", "capacity"]], on=["ID"], how="left")

    def weighted_avg(group, values, weights):
        return (group[values] * group[weights]).sum() / group[weights].sum()

    wavg = lambda x: weighted_avg(df.loc[x.index], "cf", "capacity")
    df = df.groupby(["month"]).agg({"cf": wavg})
    return df.reset_index()


def plot_obs_vs_unc_monthly_cf(run_dir, country, year_test, turb_info, turb_info_train, output_dir):
    """Plot monthly observed CF (train years + 2020) using weighted averages."""
    print("\n[11] Plotting observed vs uncorrected monthly CF...")

    obs_path = run_dir / "results" / "capacity-factor" / f"{country}_{year_test}_obs_cf.csv"
    train_gen_path = run_dir / "training" / f"{country}_train_gen_cf.csv"
    train_sim_path = run_dir / "results" / "capacity-factor" / f"{country}_train_sim_cf.csv"
    train_obs_path = run_dir / "results" / "capacity-factor" / f"{country}_train_obs_cf.csv"
    sim_path = run_dir / "results" / "capacity-factor" / f"{country}_{year_test}_unc_cf.csv"
    if not obs_path.exists() or not sim_path.exists():
        print(f"  ! Missing monthly CF inputs: {obs_path.name}, {sim_path.name}")
        return

    test_obs = pd.read_csv(obs_path, parse_dates=["time"])
    obs_cf_test = wavg_cf_monthly(test_obs, turb_info)
    obs_cf_test["year"] = pd.DatetimeIndex(obs_cf_test["time"]).year
    obs_cf_test["month"] = pd.DatetimeIndex(obs_cf_test["time"]).month
    obs_cf_test = obs_cf_test.drop("time", axis=1)[["year", "month", "cf"]]
    obs_cf_test["cf"] = obs_cf_test["cf"] * 100

    turb_info_train = turb_info_train.copy()
    turb_info_train["ID"] = turb_info_train["ID"].astype(str)

    if train_gen_path.exists():
        gen_train = pd.read_csv(train_gen_path)
        gen_train["ID"] = gen_train["ID"].astype(str)
    elif train_sim_path.exists() and train_obs_path.exists():
        sim_train = pd.read_csv(train_sim_path, parse_dates=["time"])
        obs_train = pd.read_csv(train_obs_path, parse_dates=["time"])
        sim_long = to_long_cf(sim_train, "sim")
        obs_long = to_long_cf(obs_train, "obs")
        sim_long["ID"] = sim_long["ID"].astype(str)
        obs_long["ID"] = obs_long["ID"].astype(str)
        if "year" not in sim_long.columns and "time" in sim_long.columns:
            sim_long["year"] = pd.to_datetime(sim_long["time"]).dt.year
            sim_long["month"] = pd.to_datetime(sim_long["time"]).dt.month
        if "year" not in obs_long.columns and "time" in obs_long.columns:
            obs_long["year"] = pd.to_datetime(obs_long["time"]).dt.year
            obs_long["month"] = pd.to_datetime(obs_long["time"]).dt.month
        merge_keys = [c for c in ["ID", "year", "month"] if c in sim_long.columns and c in obs_long.columns]
        gen_train = sim_long.merge(obs_long, on=merge_keys, how="inner")[["ID", "year", "month", "sim", "obs"]]
    else:
        print(f"  ! Missing training monthly inputs: {train_gen_path.name} (or fallback {train_sim_path.name}/{train_obs_path.name})")
        return

    gen_train = pd.merge(gen_train, turb_info_train[["ID", "capacity"]], on=["ID"], how="left")

    def weighted_avg(group, values, weights):
        return (group[values] * group[weights]).sum() / group[weights].sum()

    wavg = lambda x: weighted_avg(gen_train.loc[x.index], "sim", "capacity")
    wavg2 = lambda x: weighted_avg(gen_train.loc[x.index], "obs", "capacity")
    comp_month = gen_train.groupby(["year", "month"]).agg({"sim": wavg, "obs": wavg2}).reset_index()
    comp_month["sim"] = comp_month["sim"] * 100
    comp_month["obs"] = comp_month["obs"] * 100

    df_sim = pd.read_csv(sim_path, parse_dates=["time"])
    df_sim["time"] = pd.to_datetime(df_sim["time"])
    df_sim["month"] = df_sim.time.dt.month
    df_sim_monthly = df_sim.drop(columns=["time"]).groupby(["month"]).mean(numeric_only=True).reset_index()
    df_sim_monthly = df_sim_monthly.melt(id_vars=["month"], var_name="ID", value_name="cf")

    turb_info_train = turb_info_train.copy()
    turb_info_train["ID"] = turb_info_train["ID"].astype(str)
    df_sim_monthly = pd.merge(df_sim_monthly, turb_info_train[["ID", "capacity"]], on=["ID"], how="left")

    def weighted_avg(group, values, weights):
        return (group[values] * group[weights]).sum() / group[weights].sum() * 100

    wavg = lambda x: weighted_avg(df_sim_monthly.loc[x.index], "cf", "capacity")
    df_sim_monthly = df_sim_monthly.groupby("month").agg({"cf": wavg}).reset_index()

    fig, ax = plt.subplots(1, figsize=(HALF_WIDTH, 6.5 * cm), dpi=STYLE["dpi"])
    sns.lineplot(
        x="month", y="obs", data=comp_month,
        hue="year", estimator="mean", errorbar="sd", legend=True, palette="tab10", ax=ax,
    )
    sns.lineplot(
        x="month", y="cf", data=obs_cf_test,
        legend=True, color="black", label=str(year_test), linestyle="--", ax=ax,
    )
    # Intentionally computed but not plotted by default (kept to match notebook workflow)
    _ = df_sim_monthly

    handles, labels = ax.get_legend_handles_labels()
    ax.set_ylabel("Capacity Factors, %")
    ax.set_xlabel("Month")
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(range(1, 13))
    ax.set_title(" ")
    ax.legend(handles, labels, ncol=1, loc="center right", frameon=False, bbox_to_anchor=(1.3, 0.5))
    ax.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()

    out = output_dir / f"{country}_obs_cf.png"
    format_axes_standard(fig)
    savefig_thesis(fig, out)
    print(f"  Saved: {out}")
    plt.close(fig)


def _monthly_error_metrics(run_dir, country, turb_info, cluster_list, time_res_list, year_test):
    """Build monthly error table similar to legacy uncor_error('monthly-error')."""
    obs_cf = pd.read_csv(
        run_dir / "results" / "capacity-factor" / f"{country}_{year_test}_obs_cf.csv",
        parse_dates=["time"],
    )
    unc_cf = pd.read_csv(
        run_dir / "results" / "capacity-factor" / f"{country}_{year_test}_unc_cf.csv",
        parse_dates=["time"],
    )

    rows = []

    unc = calculate_error("monthly-error", unc_cf, obs_cf, turb_info, train=False).reset_index()
    unc["num_clu"] = 1
    unc["time_res"] = "uncorrected"
    rows.append(unc)

    for num_clu in cluster_list:
        for time_res in time_res_list:
            cor_path = run_dir / "results" / "capacity-factor" / f"{country}_{year_test}_{time_res}_{num_clu}_cor_cf.csv"
            if not cor_path.exists():
                continue
            cor_cf = pd.read_csv(cor_path, parse_dates=["time"])
            cur = calculate_error("monthly-error", cor_cf, obs_cf, turb_info, train=False).reset_index()
            cur["num_clu"] = num_clu
            cur["time_res"] = time_res
            rows.append(cur)

    if not rows:
        return pd.DataFrame(columns=["In training?", "month", "ID", "num_clu", "time_res", "diff"])

    return pd.concat(rows, ignore_index=True)


def plot_monthly_error(run_dir, country, df_metrics, output_dir):
    """Plot monthly MBE for n_clu=1 and n_clu=3300 across temporal resolutions."""
    print("\n[12] Plotting monthly error profiles...")
    base_clu = 1
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 6.5 * cm), sharey="all", dpi=STYLE["dpi"])

    left_df = df_metrics[
        ((df_metrics["num_clu"] == base_clu) & (df_metrics["time_res"] != "uncorrected"))
        & (df_metrics["In training?"] == "Both")
    ].copy()
    right_df = df_metrics[
        (df_metrics["num_clu"] == 3300)
        & (df_metrics["In training?"] == "Both")
    ].copy()

    for tres in ordered_time_res(["month", "bimonth", "season", "fixed"]):
        left_sub = left_df[left_df["time_res"] == tres].sort_values("month")
        right_sub = right_df[right_df["time_res"] == tres].sort_values("month")

        if len(left_sub) > 0:
            axes[0].plot(
                left_sub["month"], left_sub["diff"],
                color=TIME_RES_COLOURS.get(tres, "#999999"),
                linestyle=TIME_RES_LINESTYLES.get(tres, "-"),
                marker="o",
                linewidth=STYLE["lw"],
                markersize=2.5,
            )

        if len(right_sub) > 0:
            axes[1].plot(
                right_sub["month"], right_sub["diff"],
                color=TIME_RES_COLOURS.get(tres, "#999999"),
                linestyle=TIME_RES_LINESTYLES.get(tres, "-"),
                marker="o",
                linewidth=STYLE["lw"],
                markersize=2.5,
                label=TIME_RES_LABELS.get(tres, tres),
            )

    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    axes[0].set_ylabel("MBE")
    axes[0].set_xlabel("Month")
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(month_labels)
    axes[0].set_title(r"No. of Clusters ($n_{clu}$) = 1")

    axes[1].set_ylabel("MBE")
    axes[1].set_xlabel("Month")
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(month_labels)
    axes[1].set_title(r"No. of Clusters ($n_{clu}$) = 3300")

    for ax in axes:
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=4,
        title=r"Temporal Frequency ($t_{freq}$)",
        frameon=False,
    )

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    out = output_dir / f"{country}_monthly_difference.png"
    format_axes_standard(fig)
    savefig_thesis(fig, out)
    print(f"  Saved: {out}")
    plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================
def main():
    output_dir = RUN_DIR / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory : {RUN_DIR}")
    print(f"Output directory: {output_dir}")

    # --- Load common data ---
    turb_info_train = pd.read_csv(
        RUN_DIR / "training" / "simulated-turbines" / f"{COUNTRY}_train_turb_info.csv"
    )
    turb_info_val = pd.read_csv(
        RUN_DIR / "training" / "simulated-turbines" / f"{COUNTRY}_{YEAR_TEST}_turb_info.csv"
    )
    train_ids = set(turb_info_train["ID"].astype(str))
    turb_info_val["In training?"] = np.where(
        turb_info_val["ID"].astype(str).isin(train_ids), "Yes", "No"
    )
    sim_cf = pd.read_csv(RUN_DIR / "results" / "capacity-factor" / f"{COUNTRY}_train_sim_cf.csv")
    obs_cf = pd.read_csv(RUN_DIR / "results" / "capacity-factor" / f"{COUNTRY}_train_obs_cf.csv")

    print(f"Training turbines : {len(turb_info_train)}")
    print(f"Validation turbines: {len(turb_info_val)}")

    # 1. Cluster analysis
    # plot_cluster_analysis(turb_info_train, output_dir)

    # # 2–4. Offset vs scalar + Voronoi maps
    # kmeans, df_factors = plot_offset_vs_scalar(turb_info_train, RUN_DIR, output_dir)
    # plot_voronoi_maps(kmeans, df_factors, turb_info_train, output_dir)

    # # 5. Turbine locations
    # plot_turbine_locations(turb_info_train, turb_info_val, output_dir)

    # # 6. Temporal slicing
    # plot_temporal_slicing(output_dir)

    # # 7. CF at turbine locations
    # plot_cf_at_turbine_locations(sim_cf, obs_cf, turb_info_train, output_dir)

    # # 8. Error vs clusters from evaluator outputs (overall / temporal / spatial)
    # plot_error_vs_clusters(RUN_DIR, turb_info_train, output_dir)

    # # 8a. Spatial split (all / existing / new turbines)
    # plot_spatial_split_error_vs_clusters(RUN_DIR, turb_info_val, output_dir)

    # # 8b. Sensitivity analysis
    # runs_dir = PROJECT_ROOT / "output" / "runs"
    # plot_sensitivity_analysis(None, output_dir, runs_dir)

    # # 9–11. Training bias + training error-vs-clusters
    # sim_long = to_long_cf(sim_cf, "sim")
    # obs_long = to_long_cf(obs_cf, "obs")
    # sim_long["ID"] = sim_long["ID"].astype(str)
    # obs_long["ID"] = obs_long["ID"].astype(str)
    # turb_info_train["ID"] = turb_info_train["ID"].astype(str)

    # merge_keys = [c for c in ["time", "year", "month", "ID"]
    #               if c in sim_long.columns and c in obs_long.columns]
    # gen_train = sim_long.merge(obs_long, on=merge_keys, how="inner")
    # gen_train = gen_train.merge(
    #     turb_info_train[["ID", "type"]].drop_duplicates(), on="ID", how="left"
    # )
    # if "month" not in gen_train.columns and "time" in gen_train.columns:
    #     gen_train["time"] = pd.to_datetime(gen_train["time"])
    #     gen_train["month"] = gen_train["time"].dt.month

    # plot_train_bias(gen_train, output_dir)
    # plot_train_error_vs_clusters(gen_train, turb_info_train, RUN_DIR, output_dir)

    # 12. Spatial turbine error correlation plot
    plot_spatial_error_correlation(RUN_DIR, turb_info_train, turb_info_val, output_dir)

    # # 13. Regional analysis plots
    # run_regional_analysis(RUN_DIR, COUNTRY, turb_info_val, turb_info_train, output_dir, YEAR_TEST)

    # # 14. Monthly observed vs uncorrected CF and monthly error profiles
    # plot_obs_vs_unc_monthly_cf(RUN_DIR, COUNTRY, YEAR_TEST, turb_info_val, turb_info_train, output_dir)
    # monthly_diff = _monthly_error_metrics(
    #     RUN_DIR,
    #     COUNTRY,
    #     turb_info_val,
    #     [1, 700, 3300],
    #     ["fixed", "season", "bimonth", "month"],
    #     YEAR_TEST,
    # )
    # plot_monthly_error(RUN_DIR, COUNTRY, monthly_diff, output_dir)

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
