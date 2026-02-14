#!/usr/bin/env python3
"""Analyse grid interpolation and ML model results for PyVWF bias corrections.

Produces figures organised by thesis chapter:

Chapter 1 — Grid Interpolation for PyPSA-Eur:
  ch1_fig1  Correction control point map (1,474 centroids)
  ch1_fig2  Per-country correction impact (corrected vs uncorrected)
  ch1_fig3  Correction factor distributions by region
  ch1_fig4  Interpolation surfaces (Nearest / IDW / RBF / Kriging)
  ch1_fig5  Spatial coverage map
  ch1_fig6  Interpolation cross-validation scores

Chapter 2 — ML Spatial Trend Learning:
  ch2_fig1  ML model comparison (5 models, random CV)
  ch2_fig2  Feature importance (Random Forest)
  ch2_fig3  RF predicted vs actual (scatter + residuals)
  ch2_fig4  Random CV vs Spatial CV performance
  ch2_fig5  ML vs Interpolation under spatial CV

Usage:
    python scripts/analyse_grid_and_ml_results.py
"""

import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from sklearn.ensemble import RandomForestRegressor

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from plotting_style import thesis_plot_style

# Try importing cartopy for coastlines
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# ============================================================================
# Configuration
# ============================================================================

STYLE = thesis_plot_style()
cm = STYLE["cm"]

# Data paths (relative to project root)
CENTROIDS_CSV = PROJECT_ROOT / "output/grid_run/turbine_grid/all_corrections_centroids.csv"
GRID_DIR = PROJECT_ROOT / "output/grid_run/turbine_grid/grid_comparison"
ML_DIR = PROJECT_ROOT / "output/grid_run/turbine_grid/ml_results"
EVAL_METRICS_CSV = PROJECT_ROOT / "output/runs/turbine_grid/pyvwf_evaluation_metrics.csv"
TERRAIN_NC = PROJECT_ROOT / "input/terrain/terrain_europe_full.nc"

# Output — separate subdirectories per chapter
CH1_DIR = PROJECT_ROOT / "output/grid_run/turbine_grid/analysis_plots/ch1_grid_interpolation"
CH2_DIR = PROJECT_ROOT / "output/grid_run/turbine_grid/analysis_plots/ch2_ml_models"

# Map extent
EUROPE_EXTENT = [-12, 32, 34, 73]

# Interpolation methods
INTERP_METHODS = ["nearest", "idw", "rbf", "kriging"]
INTERP_LABELS = {"nearest": "Nearest Neighbour", "idw": "IDW", "rbf": "RBF", "kriging": "Kriging"}

# ML models
ML_LABELS = {
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elastic_net": "ElasticNet",
}

# Colour palettes
METHOD_COLOURS = {"nearest": "#636363", "idw": "#3182bd", "rbf": "#e6550d", "kriging": "#31a354"}
ML_COLOURS = {
    "random_forest": "#3182bd", "gradient_boosting": "#e6550d",
    "ridge": "#756bb1", "lasso": "#31a354", "elastic_net": "#636363",
}


# ============================================================================
# Data Loading
# ============================================================================

def load_corrections_centroids():
    df = pd.read_csv(CENTROIDS_CSV)
    df["label"] = df.apply(
        lambda r: f"{r['country_code']}-{r['cluster_mode']}"
        if r["cluster_mode"] != "all" else r["country_code"],
        axis=1,
    )
    return df


def load_interpolation_grids():
    grids = {}
    for method in INTERP_METHODS:
        nc_path = GRID_DIR / f"europe_corrections_{method}.nc"
        if nc_path.exists():
            grids[method] = xr.open_dataset(nc_path)
        else:
            print(f"  Warning: {nc_path.name} not found, skipping {method}")
    return grids


def load_cv_scores():
    csv_path = GRID_DIR / "cv_scores.csv"
    if not csv_path.exists():
        print(f"  Warning: {csv_path} not found")
        return None
    return pd.read_csv(csv_path, index_col=0)


def load_ml_comparison():
    csv_path = ML_DIR / "model_comparison.csv"
    if not csv_path.exists():
        print(f"  Warning: {csv_path} not found")
        return None
    return pd.read_csv(csv_path)


def parse_ml_training_summary():
    txt_path = ML_DIR / "training_summary.txt"
    if not txt_path.exists():
        return None
    text = txt_path.read_text()
    summary = {}
    for target in ["SCALAR", "OFFSET"]:
        block = text.split(target)[1] if target in text else ""
        for metric in ["R²", "MAE", "RMSE"]:
            pattern = rf"CV {metric}:\s*([\d.]+)\s*±\s*([\d.]+)"
            match = re.search(pattern, block)
            if match:
                key = f"{target.lower()}_{metric.lower().replace('²', '2')}"
                summary[key] = (float(match.group(1)), float(match.group(2)))
    return summary


def load_evaluation_metrics():
    if not EVAL_METRICS_CSV.exists():
        print(f"  Warning: {EVAL_METRICS_CSV} not found")
        return None
    return pd.read_csv(EVAL_METRICS_CSV)


# ============================================================================
# Spatial CV for ML Models
# ============================================================================

def spatial_cv_split(df, n_splits=5):
    """Create spatial CV splits by sorting points by longitude."""
    sorted_idx = df["lon"].argsort().values
    n = len(df)
    fold_size = n // n_splits
    splits = []
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_splits - 1 else n
        test_indices = sorted_idx[test_start:test_end]
        train_indices = np.concatenate([sorted_idx[:test_start], sorted_idx[test_end:]])
        splits.append((train_indices, test_indices))
    return splits


def extract_ml_features(centroids_df):
    """Extract terrain + spatial features for ML. Returns (features_df, feature_cols)."""
    if not TERRAIN_NC.exists():
        print(f"    Terrain data not found: {TERRAIN_NC}")
        return None, None

    terrain_ds = xr.open_dataset(TERRAIN_NC)
    features_df = centroids_df.copy()
    for var_name in terrain_ds.data_vars:
        lons = xr.DataArray(features_df["lon"].values, dims="points")
        lats = xr.DataArray(features_df["lat"].values, dims="points")
        values = terrain_ds[var_name].interp(lon=lons, lat=lats, method="linear").values
        features_df[f"terrain_{var_name}"] = values

    features_df["abs_lat"] = np.abs(features_df["lat"])

    feature_cols = [c for c in features_df.columns if c.startswith("terrain_")]
    feature_cols += ["lon", "lat", "abs_lat"]

    for col in feature_cols:
        if features_df[col].isna().any():
            features_df[col] = features_df[col].fillna(features_df[col].median())

    return features_df, feature_cols


def run_spatial_cv_ml(features_df, feature_cols, n_splits=5):
    """Run spatial-fold CV for Random Forest."""
    print("\n  Running spatial-fold CV for ML (Random Forest)...")

    X = features_df[feature_cols].values
    y_scalar = features_df["scalar"].values
    y_offset = features_df["offset"].values

    print(f"    Features: {len(feature_cols)}, Samples: {len(X)}, Folds: {n_splits}")

    splits = spatial_cv_split(features_df, n_splits=n_splits)
    results = {k: [] for k in ["scalar_mae", "scalar_rmse", "scalar_r2",
                                "offset_mae", "offset_rmse", "offset_r2"]}

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        for target_name, y_all in [("scalar", y_scalar), ("offset", y_offset)]:
            y_train, y_test = y_all[train_idx], y_all[test_idx]
            model = RandomForestRegressor(
                n_estimators=100, max_depth=15,
                min_samples_split=10, min_samples_leaf=5,
                random_state=42, n_jobs=-1,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = np.abs(y_pred - y_test).mean()
            rmse = np.sqrt(((y_pred - y_test) ** 2).mean())
            ss_res = ((y_test - y_pred) ** 2).sum()
            ss_tot = ((y_test - y_test.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            results[f"{target_name}_mae"].append(mae)
            results[f"{target_name}_rmse"].append(rmse)
            results[f"{target_name}_r2"].append(r2)
        print(f"    Fold {fold_idx + 1}/{n_splits}: "
              f"scalar MAE={results['scalar_mae'][-1]:.4f}, "
              f"offset MAE={results['offset_mae'][-1]:.4f}")

    summary = {}
    for key, vals in results.items():
        summary[key] = (np.mean(vals), np.std(vals))

    print(f"    Spatial CV RF: scalar MAE={summary['scalar_mae'][0]:.4f} +/- {summary['scalar_mae'][1]:.4f}, "
          f"R2={summary['scalar_r2'][0]:.4f} +/- {summary['scalar_r2'][1]:.4f}")
    print(f"                   offset MAE={summary['offset_mae'][0]:.4f} +/- {summary['offset_mae'][1]:.4f}, "
          f"R2={summary['offset_r2'][0]:.4f} +/- {summary['offset_r2'][1]:.4f}")

    cv_path = CH2_DIR / "ml_spatial_cv_scores.csv"
    cv_row = {f"{k}_mean": v[0] for k, v in summary.items()}
    cv_row.update({f"{k}_std": v[1] for k, v in summary.items()})
    pd.DataFrame([cv_row]).to_csv(cv_path, index=False)
    print(f"    Saved: {cv_path}")

    return summary


def train_full_rf(features_df, feature_cols):
    """Train RF on all data, return models and predictions for diagnostic plots."""
    X = features_df[feature_cols].values
    models = {}
    predictions = {}
    for target in ["scalar", "offset"]:
        y = features_df[target].values
        model = RandomForestRegressor(
            n_estimators=100, max_depth=15,
            min_samples_split=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
        model.fit(X, y)
        models[target] = model
        predictions[target] = model.predict(X)
    return models, predictions


# ============================================================================
# Helper: Map Axes
# ============================================================================

def make_map_ax(fig, nrows, ncols, index, extent=None):
    if extent is None:
        extent = EUROPE_EXTENT
    if HAS_CARTOPY:
        ax = fig.add_subplot(nrows, ncols, index, projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines(resolution="50m", linewidth=0.4, color="0.3")
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":", color="0.5")
        return ax
    else:
        ax = fig.add_subplot(nrows, ncols, index)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2, linewidth=0.4)
        return ax


def set_map_labels(ax, xlabel=True, ylabel=True):
    if HAS_CARTOPY:
        ax.tick_params(labelsize=5)
    else:
        if xlabel:
            ax.set_xlabel("Longitude")
        if ylabel:
            ax.set_ylabel("Latitude")


# ============================================================================
# CHAPTER 1 FIGURES — Grid Interpolation
# ============================================================================

def ch1_fig1_correction_map(centroids_df):
    """European correction control point map."""
    print("  ch1_fig1: Correction map...")
    fig = plt.figure(figsize=(17 * cm, 9 * cm))

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

        kwargs = {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}
        sc = ax.scatter(
            centroids_df["lon"], centroids_df["lat"],
            c=vals, cmap="RdBu_r", norm=norm,
            s=4, edgecolors="none", alpha=0.8, **kwargs,
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cbar.ax.tick_params(labelsize=5)
        cbar.set_label(label, fontsize=6)
        set_map_labels(ax)
        ax.set_title(var.capitalize(), fontsize=7, fontweight="bold")

    fig.suptitle(f"PyVWF Correction Factors ({len(centroids_df)} Cluster Centroids)",
                 fontsize=7, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_path = CH1_DIR / "ch1_fig1_correction_map.png"
    fig.savefig(save_path, dpi=STYLE["dpi"])
    plt.close(fig)
    print(f"    Saved: {save_path}")


def ch1_fig2_country_performance(eval_df):
    """Per-country corrected vs uncorrected MAE."""
    print("  ch1_fig2: Country performance...")
    if eval_df is None:
        print("    No evaluation data, skipping.")
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

    fig, ax = plt.subplots(figsize=(14 * cm, 7 * cm))
    x = np.arange(len(merged))
    width = 0.35

    ax.bar(x - width / 2, merged["uncorrected_mae"], width,
           color="#d9534f", edgecolor="black", linewidth=0.4,
           label="Uncorrected", alpha=0.8)
    ax.bar(x + width / 2, merged["corrected_mae"], width,
           color="#5cb85c", edgecolor="black", linewidth=0.4,
           label="Corrected", alpha=0.8)

    for i, row in merged.iterrows():
        y_max = max(row["uncorrected_mae"], row["corrected_mae"])
        ax.text(i, y_max + 0.005,
                f"{row['improvement_pct']:+.0f}%",
                ha="center", va="bottom", fontsize=4.5, fontweight="bold",
                color="#2a6e2a" if row["improvement_pct"] > 0 else "#a02020")

    ax.set_xticks(x)
    ax.set_xticklabels(merged["label"], fontsize=5.5, rotation=35, ha="right")
    ax.set_ylabel("MAE (capacity factor)")
    ax.set_title("Correction Impact by Region", fontsize=7, fontweight="bold")
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    fig.tight_layout()

    save_path = CH1_DIR / "ch1_fig2_country_performance.png"
    fig.savefig(save_path, dpi=STYLE["dpi"])
    plt.close(fig)
    print(f"    Saved: {save_path}")


def ch1_fig3_correction_distributions(centroids_df):
    """Correction factor distributions by region (box plots)."""
    print("  ch1_fig3: Correction distributions...")
    order = (centroids_df.groupby("label")["scalar"]
             .median().sort_values().index.tolist())

    fig, axes = plt.subplots(1, 2, figsize=(17 * cm, 7 * cm))

    for ax, (var, title, ref_line) in zip(axes, [
        ("scalar", "Scalar Correction", 1.0),
        ("offset", "Offset Correction (m/s)", 0.0),
    ]):
        data_groups = [centroids_df.loc[centroids_df["label"] == lab, var].dropna().values
                       for lab in order]
        bp = ax.boxplot(
            data_groups, vert=True, patch_artist=True,
            showfliers=True, flierprops=dict(marker=".", markersize=1.5, alpha=0.4),
            medianprops=dict(color="black", linewidth=0.8),
            boxprops=dict(linewidth=0.5),
            whiskerprops=dict(linewidth=0.5),
            capprops=dict(linewidth=0.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#8da0cb")
            patch.set_alpha(0.6)
        ax.axhline(ref_line, color="red", linewidth=0.6, linestyle="--", alpha=0.6,
                    label=f"Identity ({ref_line})")
        ax.set_xticklabels(order, fontsize=4.5, rotation=45, ha="right")
        ax.set_title(title, fontsize=7, fontweight="bold")
        ax.legend(fontsize=5, frameon=False)

    fig.suptitle("Correction Factor Distributions by Region",
                 fontsize=7, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_path = CH1_DIR / "ch1_fig3_correction_distributions.png"
    fig.savefig(save_path, dpi=STYLE["dpi"])
    plt.close(fig)
    print(f"    Saved: {save_path}")


def ch1_fig4_interpolation_surfaces(grids):
    """2x4 grid of interpolated surfaces."""
    print("  ch1_fig4: Interpolation surfaces...")
    methods = [m for m in INTERP_METHODS if m in grids]
    ncols = len(methods)
    if ncols == 0:
        print("    No grids found, skipping.")
        return

    fig = plt.figure(figsize=(ncols * 5 * cm, 10 * cm))

    for row, (var, vcenter, label) in enumerate([
        ("scalar", 1.0, "Scalar"),
        ("offset", 0.0, "Offset (m/s)"),
    ]):
        all_vals = []
        for m in methods:
            if var in grids[m]:
                data = grids[m][var].values
                all_vals.append(data[np.isfinite(data)])
        if not all_vals:
            continue
        all_vals = np.concatenate(all_vals)
        vmin = np.nanpercentile(all_vals, 2)
        vmax = np.nanpercentile(all_vals, 98)
        vmin = min(vmin, vcenter - 0.01)
        vmax = max(vmax, vcenter + 0.01)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        for col, method in enumerate(methods):
            idx = row * ncols + col + 1
            ax = make_map_ax(fig, 2, ncols, idx)
            ds = grids[method]
            if var not in ds:
                continue
            kwargs = {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}
            ax.pcolormesh(ds["lon"].values, ds["lat"].values, ds[var].values,
                          cmap="RdBu_r", norm=norm, shading="auto", **kwargs)
            set_map_labels(ax, xlabel=(row == 1), ylabel=(col == 0))
            if row == 0:
                ax.set_title(INTERP_LABELS.get(method, method), fontsize=6, fontweight="bold")
            if col == 0:
                ax.text(-0.15, 0.5, label, transform=ax.transAxes,
                        fontsize=6, fontweight="bold", rotation=90, va="center", ha="center")

        cbar_ax = fig.add_axes([0.92, 0.55 - row * 0.45, 0.015, 0.35])
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
        fig.colorbar(sm, cax=cbar_ax).ax.tick_params(labelsize=5)

    fig.subplots_adjust(wspace=0.05, hspace=0.15, right=0.9)
    save_path = CH1_DIR / "ch1_fig4_interpolation_surfaces.png"
    fig.savefig(save_path, dpi=STYLE["dpi"])
    plt.close(fig)
    print(f"    Saved: {save_path}")


def ch1_fig5_spatial_coverage(grids, centroids_df):
    """Distance-to-nearest-control heatmap."""
    print("  ch1_fig5: Spatial coverage...")
    ds = grids.get("idw") or next(iter(grids.values()), None)
    if ds is None or "distance_to_nearest_control" not in ds:
        print("    No distance data, skipping.")
        return

    fig = plt.figure(figsize=(10 * cm, 10 * cm))
    ax = make_map_ax(fig, 1, 1, 1)
    dist = ds["distance_to_nearest_control"].values

    kwargs = {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    pcm = ax.pcolormesh(ds["lon"].values, ds["lat"].values, dist,
                        cmap="YlOrRd", shading="auto", vmin=0, vmax=6, **kwargs)
    cbar = plt.colorbar(pcm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Distance to nearest control point (deg)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    ax.scatter(centroids_df["lon"], centroids_df["lat"],
               s=1.5, c="black", alpha=0.5, zorder=5, **kwargs)
    ax.contour(ds["lon"].values, ds["lat"].values, dist, levels=[5.0],
               colors=["red"], linewidths=[0.8], linestyles=["--"], **kwargs)

    set_map_labels(ax)
    ax.set_title("Spatial Coverage: Distance to Nearest Control Point",
                 fontsize=6, fontweight="bold")
    fig.tight_layout()
    save_path = CH1_DIR / "ch1_fig5_spatial_coverage.png"
    fig.savefig(save_path, dpi=STYLE["dpi"])
    plt.close(fig)
    print(f"    Saved: {save_path}")


def ch1_fig6_interpolation_cv(cv_df):
    """Interpolation cross-validation scores (bar chart)."""
    print("  ch1_fig6: Interpolation CV...")
    if cv_df is None:
        print("    No CV data, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14 * cm, 6 * cm), sharey=False)
    methods = cv_df.index.tolist()
    x = np.arange(len(methods))
    width = 0.35

    for ax_idx, (prefix, title) in enumerate([("scalar", "Scalar"), ("offset", "Offset")]):
        ax = axes[ax_idx]
        mae_mean = cv_df[f"{prefix}_mae_mean"].values
        mae_std = cv_df[f"{prefix}_mae_std"].values
        rmse_mean = cv_df[f"{prefix}_rmse_mean"].values
        rmse_std = cv_df[f"{prefix}_rmse_std"].values
        colours = [METHOD_COLOURS.get(m, "#999") for m in methods]

        bars1 = ax.bar(x - width / 2, mae_mean, width, yerr=mae_std,
                       color=colours, edgecolor="black", linewidth=0.4,
                       capsize=2, label="MAE")
        ax.bar(x + width / 2, rmse_mean, width, yerr=rmse_std,
               color=colours, edgecolor="black", linewidth=0.4,
               capsize=2, alpha=0.5, label="RMSE")
        ax.set_xticks(x)
        ax.set_xticklabels([INTERP_LABELS.get(m, m) for m in methods], fontsize=6)
        ax.set_title(title, fontsize=7, fontweight="bold")
        ax.set_ylabel("Error" if ax_idx == 0 else "")
        ax.legend(fontsize=5, frameon=False)
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=4.5)

    fig.suptitle("Interpolation Cross-Validation (5-fold spatial)", fontsize=7, fontweight="bold")
    fig.tight_layout()
    save_path = CH1_DIR / "ch1_fig6_interpolation_cv.png"
    fig.savefig(save_path, dpi=STYLE["dpi"])
    plt.close(fig)
    print(f"    Saved: {save_path}")


# ============================================================================
# CHAPTER 2 FIGURES — ML Models
# ============================================================================

def ch2_fig1_ml_model_comparison(ml_df):
    """5-model comparison bar chart (random CV)."""
    print("  ch2_fig1: ML model comparison...")
    if ml_df is None:
        print("    No ML data, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14 * cm, 6 * cm))
    models = ml_df["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    for ax_idx, (prefix, title) in enumerate([("scalar", "Scalar"), ("offset", "Offset")]):
        ax = axes[ax_idx]
        r2_vals = ml_df[f"{prefix}_r2"].values
        mae_vals = ml_df[f"{prefix}_mae"].values
        colours = [ML_COLOURS.get(m, "#999") for m in models]

        bars_r2 = ax.bar(x - width / 2, r2_vals, width, color=colours,
                         edgecolor="black", linewidth=0.4, label="R$^2$")
        ax.bar(x + width / 2, mae_vals, width, color=colours,
               edgecolor="black", linewidth=0.4, alpha=0.5, label="MAE")

        best_idx = np.argmax(r2_vals)
        bars_r2[best_idx].set_edgecolor("red")
        bars_r2[best_idx].set_linewidth(1.2)

        ax.set_xticks(x)
        ax.set_xticklabels([ML_LABELS.get(m, m) for m in models],
                           fontsize=5, rotation=30, ha="right")
        ax.set_title(title, fontsize=7, fontweight="bold")
        ax.set_ylabel("Score" if ax_idx == 0 else "")
        ax.legend(fontsize=5, frameon=False)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
        for bar in bars_r2:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    max(yval, 0) + 0.01,
                    f"{yval:.3f}", ha="center", va="bottom", fontsize=4.5)

    fig.suptitle("ML Model Comparison (5-fold random CV)", fontsize=7, fontweight="bold")
    fig.tight_layout()
    save_path = CH2_DIR / "ch2_fig1_ml_model_comparison.png"
    fig.savefig(save_path, dpi=STYLE["dpi"])
    plt.close(fig)
    print(f"    Saved: {save_path}")


def ch2_fig2_feature_importance(models, feature_cols):
    """Feature importance bar charts for scalar and offset RF models."""
    print("  ch2_fig2: Feature importance...")
    if models is None:
        print("    No models, skipping.")
        return

    # Clean feature names for display
    display_names = []
    for f in feature_cols:
        name = f.replace("terrain_", "").replace("_", " ").title()
        if f == "abs_lat":
            name = "|Latitude|"
        display_names.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(17 * cm, 7 * cm))

    for ax, (target, title) in zip(axes, [("scalar", "Scalar"), ("offset", "Offset")]):
        importances = models[target].feature_importances_
        sorted_idx = np.argsort(importances)

        ax.barh(np.arange(len(feature_cols)), importances[sorted_idx],
                color="#3182bd", edgecolor="black", linewidth=0.3, alpha=0.85)
        ax.set_yticks(np.arange(len(feature_cols)))
        ax.set_yticklabels([display_names[i] for i in sorted_idx], fontsize=5.5)
        ax.set_xlabel("Importance", fontsize=6)
        ax.set_title(f"{title} Prediction", fontsize=7, fontweight="bold")

        # Label top 3
        for rank in range(1, 4):
            idx = sorted_idx[-rank]
            val = importances[idx]
            ax.text(val + 0.005, len(feature_cols) - rank, f"{val:.3f}",
                    va="center", fontsize=4.5)

    fig.suptitle("Random Forest Feature Importance", fontsize=7, fontweight="bold")
    fig.tight_layout()
    save_path = CH2_DIR / "ch2_fig2_feature_importance.png"
    fig.savefig(save_path, dpi=STYLE["dpi"])
    plt.close(fig)
    print(f"    Saved: {save_path}")


def ch2_fig3_predictions_scatter(features_df, predictions):
    """Predicted vs actual scatter + residual plots for RF."""
    print("  ch2_fig3: Predictions scatter...")
    if predictions is None:
        print("    No predictions, skipping.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14 * cm, 12 * cm))

    for row, (target, title) in enumerate([("scalar", "Scalar"), ("offset", "Offset (m/s)")]):
        y_true = features_df[target].values
        y_pred = predictions[target]
        residuals = y_pred - y_true

        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        # Scatter
        ax_scatter = axes[row, 0]
        ax_scatter.scatter(y_true, y_pred, s=2, alpha=0.4, c="#3182bd", edgecolors="none")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax_scatter.plot(lims, lims, "--", color="red", linewidth=0.8, label="1:1")
        ax_scatter.set_xlabel(f"True {target}", fontsize=6)
        ax_scatter.set_ylabel(f"Predicted {target}", fontsize=6)
        ax_scatter.set_title(f"{title}: R$^2$={r2:.3f}, MAE={mae:.3f}",
                             fontsize=6, fontweight="bold")
        ax_scatter.legend(fontsize=5, frameon=False)

        # Residuals
        ax_resid = axes[row, 1]
        ax_resid.scatter(y_true, residuals, s=2, alpha=0.4, c="#3182bd", edgecolors="none")
        ax_resid.axhline(0, color="red", linewidth=0.8, linestyle="--")
        ax_resid.set_xlabel(f"True {target}", fontsize=6)
        ax_resid.set_ylabel("Residual (Pred - True)", fontsize=6)
        bias = residuals.mean()
        std = residuals.std()
        ax_resid.set_title(f"Residuals: Bias={bias:.3f}, Std={std:.3f}",
                           fontsize=6, fontweight="bold")

    fig.suptitle("Random Forest Predictions (trained on all data)",
                 fontsize=7, fontweight="bold")
    fig.tight_layout()
    save_path = CH2_DIR / "ch2_fig3_predictions_scatter.png"
    fig.savefig(save_path, dpi=STYLE["dpi"])
    plt.close(fig)
    print(f"    Saved: {save_path}")


def ch2_fig4_random_vs_spatial_cv(ml_summary, ml_spatial_cv):
    """Side-by-side comparison of random vs spatial CV for RF."""
    print("  ch2_fig4: Random vs Spatial CV...")
    if ml_summary is None or ml_spatial_cv is None:
        print("    Missing data, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14 * cm, 6 * cm))
    metrics = ["mae", "r2"]
    metric_labels = {"mae": "MAE", "r2": "R$^2$"}

    for ax_idx, (prefix, title) in enumerate([("scalar", "Scalar"), ("offset", "Offset")]):
        ax = axes[ax_idx]
        x = np.arange(len(metrics))
        width = 0.35

        random_vals = []
        random_errs = []
        spatial_vals = []
        spatial_errs = []

        for metric in metrics:
            rkey = f"{prefix}_{metric}"
            skey = f"{prefix}_{metric}"
            r_mean, r_std = ml_summary.get(rkey, (np.nan, np.nan))
            s_mean, s_std = ml_spatial_cv.get(skey, (np.nan, np.nan))
            random_vals.append(r_mean)
            random_errs.append(r_std)
            spatial_vals.append(s_mean)
            spatial_errs.append(s_std)

        bars1 = ax.bar(x - width / 2, random_vals, width, yerr=random_errs,
                       color="#3182bd", edgecolor="black", linewidth=0.4,
                       capsize=3, label="Random CV", alpha=0.85)
        bars2 = ax.bar(x + width / 2, spatial_vals, width, yerr=spatial_errs,
                       color="#e6550d", edgecolor="black", linewidth=0.4,
                       capsize=3, label="Spatial CV", alpha=0.85, hatch="//")

        for bar in list(bars1) + list(bars2):
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    max(yval, 0) + 0.01,
                    f"{yval:.3f}", ha="center", va="bottom", fontsize=4.5)

        ax.set_xticks(x)
        ax.set_xticklabels([metric_labels[m] for m in metrics], fontsize=6)
        ax.set_title(title, fontsize=7, fontweight="bold")
        ax.legend(fontsize=5, frameon=False)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)

    fig.suptitle("Random Forest: Random vs Spatial Cross-Validation",
                 fontsize=7, fontweight="bold")
    fig.tight_layout()
    save_path = CH2_DIR / "ch2_fig4_random_vs_spatial_cv.png"
    fig.savefig(save_path, dpi=STYLE["dpi"])
    plt.close(fig)
    print(f"    Saved: {save_path}")


def ch2_fig5_ml_vs_interpolation(cv_df, ml_summary, ml_spatial_cv):
    """ML vs Interpolation under fair spatial CV."""
    print("  ch2_fig5: ML vs Interpolation (spatial CV)...")
    if cv_df is None or (ml_summary is None and ml_spatial_cv is None):
        print("    Missing data, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(17 * cm, 7 * cm))

    for ax_idx, (prefix, title) in enumerate([("scalar", "Scalar MAE"), ("offset", "Offset MAE")]):
        ax = axes[ax_idx]

        methods = cv_df.index.tolist()
        interp_mae = cv_df[f"{prefix}_mae_mean"].values
        interp_std = cv_df[f"{prefix}_mae_std"].values

        labels = [INTERP_LABELS.get(m, m) for m in methods]
        mae_vals = list(interp_mae)
        std_vals = list(interp_std)
        colours = [METHOD_COLOURS.get(m, "#999") for m in methods]
        hatches = [""] * len(methods)

        if ml_summary:
            ml_key = f"{prefix}_mae"
            ml_mae, ml_std = ml_summary.get(ml_key, (np.nan, np.nan))
            labels.append("RF\n(random CV)")
            mae_vals.append(ml_mae)
            std_vals.append(ml_std)
            colours.append(ML_COLOURS["random_forest"])
            hatches.append("")

        if ml_spatial_cv:
            sp_key = f"{prefix}_mae"
            sp_mae, sp_std = ml_spatial_cv.get(sp_key, (np.nan, np.nan))
            labels.append("RF\n(spatial CV)")
            mae_vals.append(sp_mae)
            std_vals.append(sp_std)
            colours.append("#e6550d")
            hatches.append("//")

        x = np.arange(len(labels))
        bars = ax.bar(x, mae_vals, 0.6, yerr=std_vals,
                      color=colours, edgecolor="black", linewidth=0.4,
                      capsize=3, alpha=0.85)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        best_idx = np.nanargmin(mae_vals)
        bars[best_idx].set_edgecolor("red")
        bars[best_idx].set_linewidth(1.2)

        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=4.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=5.5)
        ax.set_title(title, fontsize=7, fontweight="bold")
        ax.set_ylabel("MAE" if ax_idx == 0 else "")

    fig.suptitle("Interpolation vs ML: Cross-Validation MAE",
                 fontsize=7, fontweight="bold")
    fig.tight_layout()
    save_path = CH2_DIR / "ch2_fig5_ml_vs_interpolation.png"
    fig.savefig(save_path, dpi=STYLE["dpi"])
    plt.close(fig)
    print(f"    Saved: {save_path}")


# ============================================================================
# Summary Statistics
# ============================================================================

def print_summary(centroids_df, cv_df, ml_df, ml_summary, eval_df, ml_spatial_cv=None):
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\nControl points: {len(centroids_df)}")
    print(f"Countries: {centroids_df['country_code'].nunique()}")
    print(f"Regions: {centroids_df['label'].nunique()}")

    print(f"\nScalar corrections:  mean={centroids_df['scalar'].mean():.3f}, "
          f"std={centroids_df['scalar'].std():.3f}, "
          f"range=[{centroids_df['scalar'].min():.3f}, {centroids_df['scalar'].max():.3f}]")
    print(f"Offset corrections:  mean={centroids_df['offset'].mean():.3f}, "
          f"std={centroids_df['offset'].std():.3f}, "
          f"range=[{centroids_df['offset'].min():.3f}, {centroids_df['offset'].max():.3f}]")

    if cv_df is not None:
        print("\nInterpolation CV (spatial folds, MAE):")
        for method in cv_df.index:
            print(f"  {INTERP_LABELS.get(method, method):20s}  "
                  f"scalar={cv_df.loc[method, 'scalar_mae_mean']:.4f} +/- {cv_df.loc[method, 'scalar_mae_std']:.4f}  "
                  f"offset={cv_df.loc[method, 'offset_mae_mean']:.4f} +/- {cv_df.loc[method, 'offset_mae_std']:.4f}")

    if ml_df is not None:
        print("\nML Model Comparison (random CV):")
        for _, row in ml_df.iterrows():
            print(f"  {ML_LABELS.get(row['model'], row['model']):20s}  "
                  f"scalar R2={row['scalar_r2']:.3f}, MAE={row['scalar_mae']:.4f}  |  "
                  f"offset R2={row['offset_r2']:.3f}, MAE={row['offset_mae']:.4f}")

    if ml_summary:
        print("\nRandom Forest (random CV):")
        for key, (mean, std) in ml_summary.items():
            print(f"  {key:15s}  {mean:.4f} +/- {std:.4f}")

    if ml_spatial_cv:
        print("\nRandom Forest (spatial CV):")
        for key, (mean, std) in ml_spatial_cv.items():
            print(f"  {key:15s}  {mean:.4f} +/- {std:.4f}")

    if eval_df is not None:
        corrected = eval_df[eval_df["correction_type"] == "corrected"]
        uncorrected = eval_df[eval_df["correction_type"] == "uncorrected"]
        if len(corrected) > 0 and len(uncorrected) > 0:
            avg_cor = corrected["mae"].mean()
            avg_unc = uncorrected["mae"].mean()
            improvement = (avg_unc - avg_cor) / avg_unc * 100
            print(f"\nOverall correction improvement: {improvement:.1f}%")
            print(f"  Mean corrected MAE:   {avg_cor:.4f}")
            print(f"  Mean uncorrected MAE: {avg_unc:.4f}")

    print("=" * 70)


# ============================================================================
# Main
# ============================================================================

def main():
    CH1_DIR.mkdir(parents=True, exist_ok=True)
    CH2_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Chapter 1 output: {CH1_DIR}")
    print(f"Chapter 2 output: {CH2_DIR}")

    # Load all data
    print("\nLoading data...")
    centroids_df = load_corrections_centroids()
    print(f"  Centroids: {len(centroids_df)} rows")

    grids = load_interpolation_grids()
    print(f"  Grids loaded: {list(grids.keys())}")

    cv_df = load_cv_scores()
    ml_df = load_ml_comparison()
    ml_summary = parse_ml_training_summary()
    eval_df = load_evaluation_metrics()

    # Extract ML features and train full model (needed for ch2 figs)
    features_df, feature_cols = extract_ml_features(centroids_df)
    ml_spatial_cv = None
    rf_models = None
    rf_predictions = None
    if features_df is not None:
        ml_spatial_cv = run_spatial_cv_ml(features_df, feature_cols, n_splits=5)
        rf_models, rf_predictions = train_full_rf(features_df, feature_cols)

    # ── Chapter 1: Grid Interpolation ──
    print("\n" + "=" * 70)
    print("CHAPTER 1: Grid Interpolation for PyPSA-Eur")
    print("=" * 70)
    ch1_fig1_correction_map(centroids_df)
    ch1_fig2_country_performance(eval_df)
    ch1_fig3_correction_distributions(centroids_df)
    ch1_fig4_interpolation_surfaces(grids)
    ch1_fig5_spatial_coverage(grids, centroids_df)
    ch1_fig6_interpolation_cv(cv_df)

    # ── Chapter 2: ML Models ──
    print("\n" + "=" * 70)
    print("CHAPTER 2: ML Spatial Trend Learning")
    print("=" * 70)
    ch2_fig1_ml_model_comparison(ml_df)
    ch2_fig2_feature_importance(rf_models, feature_cols)
    ch2_fig3_predictions_scatter(features_df, rf_predictions)
    ch2_fig4_random_vs_spatial_cv(ml_summary, ml_spatial_cv)
    ch2_fig5_ml_vs_interpolation(cv_df, ml_summary, ml_spatial_cv)

    # Print summary
    print_summary(centroids_df, cv_df, ml_df, ml_summary, eval_df, ml_spatial_cv)

    print(f"\nChapter 1 figures: {CH1_DIR}")
    print(f"Chapter 2 figures: {CH2_DIR}")


if __name__ == "__main__":
    main()
