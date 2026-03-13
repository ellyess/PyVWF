#!/usr/bin/env python3
"""Generate Chapter 5 ML model figures.

Reproduces and extends the ML correction figures using the latest
turbine-level training results (feature selection, tuned models,
spatial CV, 8 models).

Figures:
    1. Model comparison bar chart (R² and MAE for all 8 models)
    2. Feature importance (Random Forest, selected features)
    3. Predictions scatter with residuals (Elastic Net best model)
    4. Feature ablation study
    5. Results progression across enhancement phases
    6. Per-fold spatial CV performance
    7. Random vs spatial CV comparison
    8. Lasso feature selection (coefficients for both targets)
    9. ML vs interpolation comparison

Usage:
    python scripts/pyvwf_ml/generate_ch5_ml_plots.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from plotting_style import thesis_plot_style
from vwf.ml_correction import (
    build_turbine_level_dataset,
    create_feature_matrix,
    train_correction_model,
    get_cv_splitter,
)

# ===========================================================================
# Style -- apply rcParams globally via thesis_plot_style
# ===========================================================================
STYLE = thesis_plot_style()
cm = STYLE["cm"]

# Thesis page width constants
FULL_WIDTH = 16 * cm    # ~6.3 in
HALF_WIDTH = 8 * cm     # ~3.15 in

# Okabe-Ito colourblind-safe palette (consistent with Chapters 3 & 4)
OKABE_ITO = [
    "#E69F00",  # 0 orange
    "#56B4E9",  # 1 sky blue
    "#009E73",  # 2 bluish green
    "#F0E442",  # 3 yellow
    "#0072B2",  # 4 blue
    "#D55E00",  # 5 vermillion
    "#CC79A7",  # 6 reddish purple
    "#000000",  # 7 black
]

# ===========================================================================
# Paths
# ===========================================================================
OUTPUT_DIR = PROJECT_ROOT / "output" / "pyvwf_ml" / "analysis_plots" / "ch5_ml_models"
ML_DIR = PROJECT_ROOT / "output" / "pyvwf_ml" / "turbine_level_corine"
UNIFIED_DIR = PROJECT_ROOT / "output" / "pyvwf_ml" / "unified_ml"
TURBINE_7FEAT_DIR = PROJECT_ROOT / "output" / "pyvwf_ml" / "turbine_7feat_tuned"
CORRECTIONS_CSV = PROJECT_ROOT / "output" / "pyvwf_to_grid" / "all_corrections_centroids.csv"
TERRAIN_NC = PROJECT_ROOT / "input" / "terrain" / "terrain_europe_enhanced.nc"
COASTLINE = PROJECT_ROOT / "input" / "terrain" / "coastlines.geojson"
INVARIANT_NC = PROJECT_ROOT / "input" / "era5" / "invariant" / "era5_invariant_europe.nc"
CORINE_NC = PROJECT_ROOT / "input" / "terrain" / "corine_europe.nc"
TURBINE_DIR = PROJECT_ROOT / "input" / "turbine_level_data"
ERA5_WIND_DIR = PROJECT_ROOT / "input" / "era5" / "EU"
INTERP_CV_CSV = PROJECT_ROOT / "output" / "pyvwf_to_grid" / "grid_comparison" / "cv_scores.csv"

# ===========================================================================
# Feature groups
# ===========================================================================
FEATURE_GROUPS = {
    "terrain": [
        "elevation", "slope", "aspect", "roughness", "curvature",
        "distance_to_coast", "is_coastal", "subgrid_variance",
        "complexity", "aspect_category", "distance_to_coast_km",
    ],
    "era5": [
        "era5_elevation", "era5_lsm", "era5_roughness_length",
        "elevation_mismatch", "era5_wind_mean", "era5_wind_std",
        "era5_wind_shear", "era5_wind_winter_mean", "era5_wind_summer_mean",
        "era5_wind_seasonal_range", "era5_weibull_k", "era5_weibull_a",
        "era5_diurnal_amplitude", "era5_wind_night_mean",
    ],
    "turbine": [
        "hub_height", "rotor_diameter", "capacity", "specific_power",
    ],
    "fleet": [
        "mean_hub_height", "mean_rotor_diameter", "mean_capacity",
    ],
    "corine": [
        "is_urban", "is_agricultural", "is_forest", "is_bare",
        "is_water", "roughness_from_lc",
    ],
    "spatial": [
        "lat_norm", "lon_norm",
    ],
}

# Readable feature names
FEATURE_LABELS = {
    "hub_height": "Hub height",
    "rotor_diameter": "Rotor diameter",
    "capacity": "Capacity",
    "specific_power": "Specific power",
    "elevation": "Elevation",
    "slope": "Slope",
    "aspect": "Aspect",
    "roughness": "Terrain roughness",
    "curvature": "Curvature",
    "distance_to_coast": "Dist. to coast (grid)",
    "is_coastal": "Is coastal",
    "subgrid_variance": "Sub-grid variance",
    "complexity": "Terrain complexity",
    "aspect_category": "Aspect category",
    "distance_to_coast_km": "Dist. to coast (km)",
    "era5_elevation": "ERA5 elevation",
    "era5_lsm": "ERA5 land-sea mask",
    "era5_roughness_length": "ERA5 roughness length",
    "elevation_mismatch": "Elevation mismatch",
    "era5_wind_mean": "ERA5 wind mean",
    "era5_wind_std": "ERA5 wind std",
    "era5_wind_shear": "ERA5 wind shear",
    "era5_wind_winter_mean": "ERA5 winter wind",
    "era5_wind_summer_mean": "ERA5 summer wind",
    "era5_wind_seasonal_range": "ERA5 seasonal range",
    "era5_weibull_k": "ERA5 Weibull k",
    "era5_weibull_a": "ERA5 Weibull a",
    "era5_diurnal_amplitude": "ERA5 diurnal amp.",
    "era5_wind_night_mean": "ERA5 night wind",
    "is_urban": "Urban",
    "is_agricultural": "Agricultural",
    "is_forest": "Forest",
    "is_bare": "Bare",
    "is_water": "Water",
    "roughness_from_lc": "Roughness (land cover)",
    "lat_norm": "Latitude",
    "lon_norm": "Longitude",
    "mean_hub_height": "Mean hub height",
    "mean_rotor_diameter": "Mean rotor diameter",
    "mean_capacity": "Mean capacity",
}

# Feature group colours (Okabe-Ito)
GROUP_COLOURS = {
    "terrain": OKABE_ITO[1],   # sky blue
    "era5": OKABE_ITO[0],      # orange
    "turbine": OKABE_ITO[2],   # bluish green
    "fleet": OKABE_ITO[2],     # bluish green (same as turbine)
    "corine": OKABE_ITO[6],    # reddish purple
    "spatial": OKABE_ITO[3],   # yellow
}

# Model display names
MODEL_NAMES = {
    "ridge": "Ridge",
    "gradient_boosting": "Grad. Boost.",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "lasso": "Lasso",
    "elastic_net": "Elastic Net",
    "mlp": "MLP (ANN)",
}

# Model colours (Okabe-Ito)
MODEL_COLOURS = {
    "ridge": OKABE_ITO[4],              # blue
    "gradient_boosting": OKABE_ITO[5],  # vermillion
    "lightgbm": OKABE_ITO[2],           # bluish green
    "xgboost": OKABE_ITO[6],            # reddish purple
    "random_forest": OKABE_ITO[1],      # sky blue
    "lasso": OKABE_ITO[0],              # orange
    "elastic_net": OKABE_ITO[7],        # black
    "mlp": OKABE_ITO[3],                # yellow
}


def load_turbine_metadata(turbine_dir):
    """Load turbine metadata for DK, UK, DE."""
    metadata = {}
    dk_path = turbine_dir / "DK" / "dk_md.csv"
    if dk_path.exists():
        dk = pd.read_csv(dk_path)
        metadata["DK"] = pd.DataFrame({
            "height": pd.to_numeric(dk["height"], errors="coerce"),
            "diameter": pd.to_numeric(dk["diameter"], errors="coerce"),
            "capacity": pd.to_numeric(dk["capacity"], errors="coerce"),
            "lon": pd.to_numeric(dk["lon"], errors="coerce"),
            "lat": pd.to_numeric(dk["lat"], errors="coerce"),
        })

    uk_path = turbine_dir / "UK" / "uk_md.csv"
    if uk_path.exists():
        uk = pd.read_csv(uk_path)
        metadata["UK"] = pd.DataFrame({
            "height": pd.to_numeric(uk["height"], errors="coerce"),
            "diameter": pd.to_numeric(uk["diameter"], errors="coerce"),
            "capacity": pd.to_numeric(uk["capacity"], errors="coerce"),
            "lon": pd.to_numeric(uk["lon"], errors="coerce"),
            "lat": pd.to_numeric(uk["lat"], errors="coerce"),
        })

    de_path = turbine_dir / "DE" / "DE_md.csv"
    if de_path.exists():
        de = pd.read_csv(de_path)
        de_df = de[["V1", "kW", "Rotor..m.", "Tower..m."]].copy()
        de_df = de_df.dropna(subset=["kW", "Rotor..m.", "Tower..m."])
        for c in ["kW", "Rotor..m.", "Tower..m."]:
            de_df[c] = pd.to_numeric(de_df[c], errors="coerce")
        de_df = de_df.dropna()
        de_df = de_df[
            (de_df["kW"] > 0) & (de_df["kW"] < 20000)
            & (de_df["Rotor..m."] > 5) & (de_df["Rotor..m."] < 300)
            & (de_df["Tower..m."] > 10) & (de_df["Tower..m."] < 300)
        ]
        metadata["DE"] = de_df

    return metadata


def build_features():
    """Build the full turbine-level feature matrix."""
    print("Loading corrections...")
    corrections = pd.read_csv(CORRECTIONS_CSV)
    print(f"  {len(corrections)} centroids")

    print("Loading turbine metadata...")
    turbine_metadata = load_turbine_metadata(TURBINE_DIR)

    de_geo = TURBINE_DIR / "DE" / "geolocate.germany.csv"
    if not de_geo.exists():
        de_geo = None

    print("Building turbine-level dataset...")
    training_df = build_turbine_level_dataset(
        corrections, turbine_metadata, de_geolocate=de_geo,
    )

    print("Creating feature matrix...")
    features_df = create_feature_matrix(
        training_df,
        terrain_nc=TERRAIN_NC if TERRAIN_NC.exists() else None,
        coastline_geojson=COASTLINE if COASTLINE.exists() else None,
        invariant_nc=INVARIANT_NC if INVARIANT_NC.exists() else None,
        corine_nc=CORINE_NC if CORINE_NC.exists() else None,
        era5_wind_dir=ERA5_WIND_DIR if ERA5_WIND_DIR.exists() else None,
        turbine_metadata=None,
    )

    # Feature columns (exclude targets, coords, metadata)
    exclude = {
        "scalar", "offset", "lon", "lat", "country_code", "country_name",
        "cluster", "cluster_mode", "obs_level", "area_km2",
        "land_cover_class", "geometry", "ID", "turbine_id", "domain", "type",
        "lat_norm", "lon_norm",
    }
    feature_cols = [c for c in features_df.columns if c not in exclude]
    feature_cols = [c for c in feature_cols if c in features_df.columns]

    print(f"  Features: {len(feature_cols)}")
    return features_df, feature_cols


def get_feature_group_colour(feat):
    """Return colour for a feature based on its group."""
    for group, cols in FEATURE_GROUPS.items():
        if feat in cols:
            return GROUP_COLOURS[group]
    return OKABE_ITO[7]  # black fallback


# ===========================================================================
# Figure 1: Model Comparison (from saved CSVs)
# ===========================================================================
def fig1_model_comparison():
    """Bar chart comparing all 8 models on both targets."""
    print("\n--- Figure 1: Model Comparison ---")

    # Use turbine_7feat_tuned (7 selected features, tuned) if available, else fallback
    if (TURBINE_7FEAT_DIR / "model_comparison_scalar.csv").exists():
        scalar_df = pd.read_csv(TURBINE_7FEAT_DIR / "model_comparison_scalar.csv")
        offset_df = pd.read_csv(TURBINE_7FEAT_DIR / "model_comparison_offset.csv")
        subtitle = "7 selected features, tuned, 5-fold spatial CV (turbine-level)"
    else:
        scalar_df = pd.read_csv(ML_DIR / "model_comparison_scalar.csv")
        offset_df = pd.read_csv(ML_DIR / "model_comparison_offset.csv")
        subtitle = "27 features, 5-fold spatial CV"

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 6 * cm))

    for ax, df, target in zip(axes, [scalar_df, offset_df], ["Scalar", "Offset"]):
        df = df.sort_values("r2_mean", ascending=False).reset_index(drop=True)
        x = np.arange(len(df))
        width = 0.35

        colours = [MODEL_COLOURS.get(m, "#666") for m in df["model"]]

        bars_r2 = ax.bar(x - width / 2, df["r2_mean"], width,
                         label=r"R$^2$", color=colours, edgecolor="black",
                         linewidth=0.3)
        ax.bar(x + width / 2, df["mae_mean"], width,
               label="MAE", color=colours, alpha=0.4,
               edgecolor="black", linewidth=0.3)

        # Error bars for R²
        if "r2_std" in df.columns:
            ax.errorbar(x - width / 2, df["r2_mean"], yerr=df["r2_std"],
                        fmt="none", color="black", capsize=2, linewidth=0.5)

        # Value labels
        for bar, val in zip(bars_r2, df["r2_mean"]):
            y = max(val, 0) + 0.008
            ax.text(bar.get_x() + bar.get_width() / 2, y,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=5,
                    fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_NAMES.get(m, m) for m in df["model"]],
                           rotation=35, ha="right")
        ax.set_ylabel("Score")
        ax.set_title(f"{target}", fontweight="bold")
        ax.axhline(0, color="grey", linewidth=0.4, linestyle="--", alpha=0.5)
        ax.legend(frameon=False, loc="upper right")
        sns.despine(ax=ax)

    fig.tight_layout()
    path = OUTPUT_DIR / "ch5_fig1_ml_model_comparison.png"
    plt.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 2: Feature Importance (Random Forest)
# ===========================================================================
def fig2_feature_importance(features_df, feature_cols):
    """Side-by-side feature importance for scalar and offset."""
    print("\n--- Figure 2: Feature Importance ---")

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 8 * cm))

    for ax, target in zip(axes, ["scalar", "offset"]):
        result = train_correction_model(
            features_df,
            target_col=target,
            feature_cols=feature_cols,
            model_type="random_forest",
            cv_folds=5,
            cv_strategy="spatial_lon",
        )

        fi = result["feature_importance"]
        if fi is None:
            continue

        fi = fi.sort_values("importance", ascending=True)

        colours = [get_feature_group_colour(f) for f in fi["feature"]]
        labels = [FEATURE_LABELS.get(f, f) for f in fi["feature"]]

        bars = ax.barh(range(len(fi)), fi["importance"], color=colours,
                       edgecolor="white", linewidth=0.2)

        ax.set_yticks(range(len(fi)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Importance (MDI)")
        ax.set_title(f"{target.capitalize()}", fontweight="bold")
        sns.despine(ax=ax)

        for bar, val in zip(bars, fi["importance"]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=4.5)

    # Legend for feature groups
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=g.capitalize())
                       for g, c in GROUP_COLOURS.items()]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.tight_layout(rect=[0, 0.04, 1, 1.0])
    path = OUTPUT_DIR / "ch5_fig2_feature_importance.png"
    plt.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    return result


# ===========================================================================
# Figure 3: Predictions Scatter (best model)
# ===========================================================================
def fig3_predictions_scatter(features_df, feature_cols):
    """Actual vs predicted scatter with residuals for best linear model."""
    print("\n--- Figure 3: Predictions Scatter (Elastic Net) ---")

    fig = plt.figure(figsize=(FULL_WIDTH, 10 * cm))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    for row, target in enumerate(["scalar", "offset"]):
        result = train_correction_model(
            features_df,
            target_col=target,
            feature_cols=feature_cols,
            model_type="elastic_net",
            cv_folds=5,
            cv_strategy="spatial_lon",
            tune_hyperparams=True,
        )

        y_true = features_df[target].values
        X = features_df[feature_cols].copy().fillna(features_df[feature_cols].median())
        mask = np.isfinite(y_true)
        y_true = y_true[mask]
        y_pred = result["model"].predict(X.loc[mask])

        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        residuals = y_pred - y_true

        # Scatter
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.scatter(y_true, y_pred, alpha=0.1, s=3, c=OKABE_ITO[4], rasterized=True)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax1.plot(lims, lims, "--", color=OKABE_ITO[5], lw=0.8, alpha=0.8, label="1:1")
        ax1.set_xlabel(f"True {target}")
        ax1.set_ylabel(f"Predicted {target}")
        unit = " (m/s)" if target == "offset" else ""
        ax1.set_title(f"{target.capitalize()}{unit}: "
                      f"R\u00b2={r2:.3f}, MAE={mae:.3f}", fontweight="bold")
        ax1.legend(frameon=False, loc="upper left")
        sns.despine(ax=ax1)

        # Residuals
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.scatter(y_true, residuals, alpha=0.1, s=3, c=OKABE_ITO[4], rasterized=True)
        ax2.axhline(0, color=OKABE_ITO[5], linestyle="--", lw=0.8, alpha=0.8)
        ax2.set_xlabel(f"True {target}")
        ax2.set_ylabel("Residual (Pred \u2212 True)")
        ax2.set_title(f"Residuals: Bias={residuals.mean():.3f}, "
                      f"Std={residuals.std():.3f}", fontweight="bold")
        sns.despine(ax=ax2)

    fig.tight_layout()
    path = OUTPUT_DIR / "ch5_fig3_predictions_scatter.png"
    plt.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 4: Feature Ablation
# ===========================================================================
def fig4_ablation():
    """Feature group ablation study."""
    print("\n--- Figure 4: Feature Ablation ---")

    ablation_df = pd.read_csv(ML_DIR / "ablation_results.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 5.5 * cm),
                                    gridspec_kw={"width_ratios": [1.3, 1]})

    # Panel A: R² by condition
    df = ablation_df.copy()
    group_labels = {
        "none (baseline)": "All features\n(baseline)",
        "terrain": "Drop terrain\n(11 feat.)",
        "era5": "Drop ERA5\n(6 feat.)",
        "turbine": "Drop turbine\n(4 feat.)",
        "corine": "Drop CORINE\n(6 feat.)",
    }
    df["label"] = df["dropped_group"].map(group_labels)

    colours = []
    for g in df["dropped_group"]:
        if g == "none (baseline)":
            colours.append(OKABE_ITO[4])   # blue
        elif g == "era5":
            colours.append(OKABE_ITO[5])   # vermillion
        else:
            colours.append(OKABE_ITO[7])   # black

    bars = ax1.bar(range(len(df)), df["r2"], color=colours,
                   edgecolor="black", linewidth=0.3)
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df["label"])
    ax1.set_ylabel(r"R$^2$ (spatial CV)")
    ax1.set_title("(a) R\u00b2 under feature ablation", fontweight="bold")
    ax1.axhline(0, color="grey", linewidth=0.4, linestyle="--")

    for bar, val in zip(bars, df["r2"]):
        y_pos = max(val, 0) + 0.008 if val >= 0 else val - 0.02
        va = "bottom" if val >= 0 else "top"
        ax1.text(bar.get_x() + bar.get_width() / 2, y_pos,
                 f"{val:.3f}", ha="center", va=va, fontsize=5, fontweight="bold")

    # Panel B: Delta R²
    drop_df = df[df["dropped_group"] != "none (baseline)"].copy()

    group_order = ["era5", "terrain", "turbine", "corine"]
    drop_df["sort_key"] = drop_df["dropped_group"].map(
        {g: i for i, g in enumerate(group_order)}
    )
    drop_df = drop_df.sort_values("sort_key")
    drop_colours = [GROUP_COLOURS.get(g, OKABE_ITO[7]) for g in drop_df["dropped_group"]]

    short_labels = {
        "terrain": "Terrain",
        "era5": "ERA5",
        "turbine": "Turbine",
        "corine": "CORINE",
    }
    bars2 = ax2.barh(range(len(drop_df)), drop_df["delta_r2"], color=drop_colours,
                     edgecolor="black", linewidth=0.3)
    ax2.set_yticks(range(len(drop_df)))
    ax2.set_yticklabels([short_labels.get(g, g) for g in drop_df["dropped_group"]])
    ax2.set_xlabel(r"$\Delta$R$^2$ (change from baseline)")
    ax2.set_title("(b) Impact of dropping group", fontweight="bold")
    ax2.axvline(0, color="grey", linewidth=0.4, linestyle="--")

    for bar, val in zip(bars2, drop_df["delta_r2"]):
        x_pos = val - 0.008 if val < 0 else val + 0.008
        ha = "right" if val < 0 else "left"
        ax2.text(x_pos, bar.get_y() + bar.get_height() / 2,
                 f"{val:+.3f}", ha=ha, va="center", fontsize=5, fontweight="bold")

    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    fig.tight_layout()
    path = OUTPUT_DIR / "ch5_fig4_feature_ablation.png"
    plt.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 5: Results Progression
# ===========================================================================
def fig5_results_progression():
    """Show R² improvement across enhancement phases."""
    print("\n--- Figure 5: Results Progression ---")

    phases = [
        "Centroid\n(1.5k pts,\n20 feat.)",
        "Turbine-level\n(23k pts,\n18 feat.)",
        "Turbine-level\n+ wind clim.\n(21 feat.)",
        "Turbine-level\n+ CORINE\n(27 feat.)",
        "35 feat.\n+ tuned",
        "Centroid\nselected\n(34 feat.)",
    ]
    # R² values (best model, spatial CV) across progressive enhancement
    scalar_r2 = [0.152, 0.240, 0.307, 0.314, 0.292, 0.253]
    offset_r2 = [0.196, 0.243, 0.309, 0.317, 0.285, 0.254]

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 6 * cm))

    x = np.arange(len(phases))
    width = 0.35

    bars_s = ax.bar(x - width / 2, scalar_r2, width, label="Scalar",
                    color=OKABE_ITO[4], edgecolor="black", linewidth=0.3)
    bars_o = ax.bar(x + width / 2, offset_r2, width, label="Offset",
                    color=OKABE_ITO[5], edgecolor="black", linewidth=0.3)

    for bars, vals in [(bars_s, scalar_r2), (bars_o, offset_r2)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.006,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=5,
                    fontweight="bold")

    # Annotate key improvements
    ax.annotate("", xy=(2, 0.24), xytext=(1, 0.155),
                arrowprops=dict(arrowstyle="->", color=OKABE_ITO[2], lw=1.2))
    ax.text(1.5, 0.20, "+58%", color=OKABE_ITO[2], fontweight="bold",
            ha="center")

    ax.annotate("", xy=(3, 0.307), xytext=(2, 0.243),
                arrowprops=dict(arrowstyle="->", color=OKABE_ITO[2], lw=1.2))
    ax.text(2.5, 0.28, "+28%", color=OKABE_ITO[2], fontweight="bold",
            ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel(r"R$^2$ (spatial CV)")
    ax.set_title("Model Performance Progression (best model, spatial CV)",
                 fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, 0.40)
    ax.axhline(0, color="grey", linewidth=0.4, linestyle="--", alpha=0.3)
    sns.despine(ax=ax)

    fig.tight_layout()
    path = OUTPUT_DIR / "ch5_fig5_results_progression.png"
    plt.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 6: Per-fold CV Performance
# ===========================================================================
def fig6_per_fold_cv(features_df, feature_cols):
    """Per-fold spatial CV performance for top models."""
    print("\n--- Figure 6: Per-fold Spatial CV ---")

    models_to_show = ["elastic_net", "ridge", "random_forest"]
    model_labels = {
        "elastic_net": "Elastic Net",
        "ridge": "Ridge",
        "random_forest": "Random Forest",
    }

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 6 * cm))

    for ax, target in zip(axes, ["scalar", "offset"]):
        fold_data = []
        for mtype in models_to_show:
            result = train_correction_model(
                features_df,
                target_col=target,
                feature_cols=feature_cols,
                model_type=mtype,
                cv_folds=5,
                cv_strategy="spatial_lon",
            )
            cv = result["cv_scores"]
            for fold_i, r2_val in enumerate(cv["test_r2"]):
                fold_data.append({
                    "Model": model_labels[mtype],
                    "Fold": fold_i + 1,
                    "R\u00b2": r2_val,
                })

        fold_df = pd.DataFrame(fold_data)

        model_order = [model_labels[m] for m in models_to_show]
        palette = {model_labels[m]: MODEL_COLOURS[m] for m in models_to_show}

        # Box + strip plot
        bp = ax.boxplot(
            [fold_df[fold_df["Model"] == m]["R\u00b2"].values for m in model_order],
            positions=range(len(model_order)),
            widths=0.45, patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=0.8),
            boxprops=dict(linewidth=0.5),
            whiskerprops=dict(linewidth=0.5),
            capprops=dict(linewidth=0.5),
        )
        for patch, m in zip(bp["boxes"], models_to_show):
            patch.set_facecolor(MODEL_COLOURS[m])
            patch.set_alpha(0.6)

        # Overlay individual fold points
        for i, m in enumerate(model_order):
            vals = fold_df[fold_df["Model"] == m]["R\u00b2"].values
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       color="black", s=12, alpha=0.7, zorder=5)

        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels(model_order)
        ax.set_title(f"{target.capitalize()}", fontweight="bold")
        ax.set_ylabel(r"R$^2$")
        ax.axhline(0, color="grey", linewidth=0.4, linestyle="--", alpha=0.5)
        sns.despine(ax=ax)

        # Mean annotation
        for i, m in enumerate(model_order):
            mean_val = fold_df[fold_df["Model"] == m]["R\u00b2"].mean()
            ax.text(i, ax.get_ylim()[1] * 0.95,
                    f"\u03bc={mean_val:.3f}", ha="center", fontsize=5,
                    fontweight="bold")

    fig.tight_layout()
    path = OUTPUT_DIR / "ch5_fig6_per_fold_cv.png"
    plt.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 7: Random vs Spatial CV
# ===========================================================================
def fig7_random_vs_spatial_cv(features_df, feature_cols):
    """Compare random vs spatial CV for Elastic Net."""
    print("\n--- Figure 7: Random vs Spatial CV ---")

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 5.5 * cm))

    for ax, target in zip(axes, ["scalar", "offset"]):
        cv_data = []
        for strategy, label in [("random", "Random CV"), ("spatial_lon", "Spatial CV")]:
            result = train_correction_model(
                features_df,
                target_col=target,
                feature_cols=feature_cols,
                model_type="elastic_net",
                cv_folds=5,
                cv_strategy=strategy,
                tune_hyperparams=True,
            )
            cv = result["cv_scores"]
            r2_mean = cv["test_r2"].mean()
            r2_std = cv["test_r2"].std()
            mae_mean = -cv["test_neg_mean_absolute_error"].mean()
            mae_std = (-cv["test_neg_mean_absolute_error"]).std()
            cv_data.append({
                "Strategy": label,
                "R\u00b2 mean": r2_mean, "R\u00b2 std": r2_std,
                "MAE mean": mae_mean, "MAE std": mae_std,
            })

        cv_df = pd.DataFrame(cv_data)
        x = np.arange(2)
        width = 0.3
        colours = [OKABE_ITO[4], OKABE_ITO[0]]  # blue, orange
        hatches = ["", "///"]

        for i, (_, row) in enumerate(cv_df.iterrows()):
            ax.bar(x[0] + i * width - width / 2, row["MAE mean"], width,
                   yerr=row["MAE std"], capsize=3,
                   color=colours[i], alpha=0.7, hatch=hatches[i],
                   label=row["Strategy"], edgecolor="black", linewidth=0.3,
                   error_kw={"linewidth": 0.5})
            ax.bar(x[1] + i * width - width / 2, row["R\u00b2 mean"], width,
                   yerr=row["R\u00b2 std"], capsize=3,
                   color=colours[i], alpha=0.7, hatch=hatches[i],
                   edgecolor="black", linewidth=0.3,
                   error_kw={"linewidth": 0.5})

            # Labels
            ax.text(x[0] + i * width - width / 2,
                    row["MAE mean"] + row["MAE std"] + 0.008,
                    f"{row["MAE mean"]:.3f}", ha="center", fontsize=5,
                    fontweight="bold")
            ax.text(x[1] + i * width - width / 2,
                    max(row["R\u00b2 mean"] + row["R\u00b2 std"] + 0.008, 0.008),
                    f"{row["R\u00b2 mean"]:.3f}", ha="center", fontsize=5,
                    fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(["MAE", r"R$^2$"])
        ax.set_title(f"{target.capitalize()}", fontweight="bold")
        ax.axhline(0, color="grey", linewidth=0.4, linestyle="--", alpha=0.3)
        ax.legend(frameon=False, loc="upper right")
        sns.despine(ax=ax)

    fig.tight_layout()
    path = OUTPUT_DIR / "ch5_fig7_random_vs_spatial_cv.png"
    plt.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 8: Lasso Feature Selection Coefficients
# ===========================================================================
def _plot_lasso_panel(ax, fi_df, top_features, title):
    """Plot a single Lasso coefficient panel on the given axis."""
    fi_sel = fi_df[fi_df["feature"].isin(top_features)].copy()
    fi_sel = fi_sel.sort_values("importance", ascending=True)

    colours = [get_feature_group_colour(f) for f in fi_sel["feature"]]
    labels = [FEATURE_LABELS.get(f, f) for f in fi_sel["feature"]]

    coefs = fi_sel["coefficient"].values
    bars = ax.barh(range(len(fi_sel)), coefs, color=colours,
                   edgecolor="black", linewidth=0.3, alpha=0.85)

    ax.set_yticks(range(len(fi_sel)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Lasso Coefficient")
    ax.set_title(title, fontweight="bold")
    ax.axvline(0, color="grey", linewidth=0.4, linestyle="--")
    sns.despine(ax=ax)

    x_range = max(abs(coefs.min()), abs(coefs.max())) or 0.01
    pad = x_range * 0.04
    for bar, val in zip(bars, coefs):
        x_pos = val + pad if val >= 0 else val - pad
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", ha=ha, va="center", fontsize=4.5)

    lo, hi = ax.get_xlim()
    ax.set_xlim(lo - x_range * 0.15, hi + x_range * 0.15)


def _get_top_features(scalar_fi, offset_fi, n=15):
    """Return the top-n features by max absolute importance across targets."""
    scalar_nz = scalar_fi[scalar_fi["importance"] > 0]
    offset_nz = offset_fi[offset_fi["importance"] > 0]
    all_feats = sorted(
        set(scalar_nz["feature"].tolist() + offset_nz["feature"].tolist()),
        key=lambda f: max(
            scalar_fi.loc[scalar_fi["feature"] == f, "importance"].values[0],
            offset_fi.loc[offset_fi["feature"] == f, "importance"].values[0],
        ),
        reverse=True,
    )
    return all_feats[:n]


def fig8_feature_selection():
    """Lasso feature selection: 2x2 panel (centroid + turbine x scalar + offset)."""
    print("\n--- Figure 8: Feature Selection ---")

    # Centroid-level (unified_ml)
    c_scalar = UNIFIED_DIR / "feature_importance_scalar.csv"
    c_offset = UNIFIED_DIR / "feature_importance_offset.csv"
    # Turbine-level
    t_scalar = ML_DIR / "feature_importance_scalar.csv"
    t_offset = ML_DIR / "feature_importance_offset.csv"

    has_centroid = c_scalar.exists() and c_offset.exists()
    has_turbine = t_scalar.exists() and t_offset.exists()

    if not has_centroid and not has_turbine:
        print("  No feature importance CSVs found, skipping.")
        return

    # Load data
    datasets = {}
    if has_centroid:
        datasets["centroid"] = {
            "scalar": pd.read_csv(c_scalar),
            "offset": pd.read_csv(c_offset),
        }
    if has_turbine:
        datasets["turbine"] = {
            "scalar": pd.read_csv(t_scalar),
            "offset": pd.read_csv(t_offset),
        }

    n_rows = len(datasets)
    level_names = list(datasets.keys())

    # Compute top features per level
    top_per_level = {}
    for lvl in level_names:
        top_per_level[lvl] = _get_top_features(
            datasets[lvl]["scalar"], datasets[lvl]["offset"], n=15,
        )

    max_feats = max(len(v) for v in top_per_level.values())
    row_height = max(6, max_feats * 0.45) * cm

    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(FULL_WIDTH, row_height * n_rows),
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    row_labels = {"centroid": "Centroid-level", "turbine": "Turbine-level"}
    for row_idx, lvl in enumerate(level_names):
        top = top_per_level[lvl]
        for col_idx, target in enumerate(["scalar", "offset"]):
            ax = axes[row_idx, col_idx]
            title = f"{row_labels[lvl]} — {target.capitalize()}"
            _plot_lasso_panel(ax, datasets[lvl][target], top, title)

    # Legend — show only groups present in the plotted features
    from matplotlib.patches import Patch
    plotted_features = set()
    for lvl in level_names:
        plotted_features.update(top_per_level[lvl])
    used_groups = set()
    for f in plotted_features:
        for g, cols in FEATURE_GROUPS.items():
            if f in cols:
                used_groups.add(g)
                break
    # Merge fleet/turbine into one legend entry "Turbine / Fleet"
    show = []
    if "turbine" in used_groups or "fleet" in used_groups:
        show.append(Patch(facecolor=GROUP_COLOURS["turbine"], label="Turbine / Fleet"))
        used_groups -= {"turbine", "fleet"}
    for g in ["terrain", "era5", "corine", "spatial"]:
        if g in used_groups:
            show.append(Patch(facecolor=GROUP_COLOURS[g], label=g.capitalize()))
    fig.legend(handles=show, loc="lower center", ncol=len(show),
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.tight_layout(rect=[0, 0.03, 1, 1.0])
    path = OUTPUT_DIR / "ch5_fig8_feature_selection.png"
    plt.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===========================================================================
# Figure 9: ML vs Interpolation Comparison
# ===========================================================================
def fig9_ml_vs_interpolation():
    """Compare ML (Elastic Net) vs interpolation (IDW, Kriging) MAE."""
    print("\n--- Figure 9: ML vs Interpolation ---")

    if not INTERP_CV_CSV.exists():
        print("  No interpolation CV scores, skipping.")
        return

    interp_cv = pd.read_csv(INTERP_CV_CSV, index_col=0)

    # ML results — use turbine_7feat_tuned for turbine-level, unified_ml for centroid-level
    if (TURBINE_7FEAT_DIR / "model_comparison_scalar.csv").exists():
        ml_scalar = pd.read_csv(TURBINE_7FEAT_DIR / "model_comparison_scalar.csv")
        ml_offset = pd.read_csv(TURBINE_7FEAT_DIR / "model_comparison_offset.csv")
    elif (UNIFIED_DIR / "model_comparison_scalar.csv").exists():
        ml_scalar = pd.read_csv(UNIFIED_DIR / "model_comparison_scalar.csv")
        ml_offset = pd.read_csv(UNIFIED_DIR / "model_comparison_offset.csv")
    else:
        ml_scalar = pd.read_csv(ML_DIR / "model_comparison_scalar.csv")
        ml_offset = pd.read_csv(ML_DIR / "model_comparison_offset.csv")

    # Best ML model (Elastic Net)
    best_s = ml_scalar.sort_values("mae_mean").iloc[0]
    best_o = ml_offset.sort_values("mae_mean").iloc[0]

    methods = ["IDW", "Kriging", f"ML ({MODEL_NAMES.get(best_s["model"], best_s["model"])})"]
    scalar_mae = [
        interp_cv.loc["idw", "scalar_mae_mean"],
        interp_cv.loc["kriging", "scalar_mae_mean"],
        best_s["mae_mean"],
    ]
    offset_mae = [
        interp_cv.loc["idw", "offset_mae_mean"],
        interp_cv.loc["kriging", "offset_mae_mean"],
        best_o["mae_mean"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 5.5 * cm))
    colours = [OKABE_ITO[4], OKABE_ITO[2], OKABE_ITO[5]]  # blue, green, vermillion

    for ax, mae_vals, title in zip(axes,
                                    [scalar_mae, offset_mae],
                                    ["Scalar", "Offset"]):
        x = np.arange(len(methods))
        bars = ax.bar(x, mae_vals, color=colours, edgecolor="black",
                      linewidth=0.3, width=0.55)

        for bar, val in zip(bars, mae_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=5,
                    fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("MAE (spatial CV)")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, max(mae_vals) * 1.15)
        sns.despine(ax=ax)

    fig.tight_layout()
    path = OUTPUT_DIR / "ch5_fig9_ml_vs_interpolation.png"
    plt.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===========================================================================
# Update CSV
# ===========================================================================
def update_csv():
    """Update ml_spatial_cv_scores.csv with latest results."""
    print("\n--- Updating CSV ---")

    if (TURBINE_7FEAT_DIR / "model_comparison_scalar.csv").exists():
        scalar_df = pd.read_csv(TURBINE_7FEAT_DIR / "model_comparison_scalar.csv")
        offset_df = pd.read_csv(TURBINE_7FEAT_DIR / "model_comparison_offset.csv")
    elif (UNIFIED_DIR / "model_comparison_scalar.csv").exists():
        scalar_df = pd.read_csv(UNIFIED_DIR / "model_comparison_scalar.csv")
        offset_df = pd.read_csv(UNIFIED_DIR / "model_comparison_offset.csv")
    else:
        scalar_df = pd.read_csv(ML_DIR / "model_comparison_scalar.csv")
        offset_df = pd.read_csv(ML_DIR / "model_comparison_offset.csv")

    best_s = scalar_df.sort_values("r2_mean", ascending=False).iloc[0]
    best_o = offset_df.sort_values("r2_mean", ascending=False).iloc[0]

    csv_data = {
        "best_model_scalar": [best_s["model"]],
        "scalar_mae_mean": [best_s["mae_mean"]],
        "scalar_rmse_mean": [best_s["rmse_mean"]],
        "scalar_r2_mean": [best_s["r2_mean"]],
        "scalar_r2_std": [best_s["r2_std"]],
        "best_model_offset": [best_o["model"]],
        "offset_mae_mean": [best_o["mae_mean"]],
        "offset_rmse_mean": [best_o["rmse_mean"]],
        "offset_r2_mean": [best_o["r2_mean"]],
        "offset_r2_std": [best_o["r2_std"]],
    }
    pd.DataFrame(csv_data).to_csv(OUTPUT_DIR / "ml_spatial_cv_scores.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / "ml_spatial_cv_scores.csv"}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 70)
    print("Generating Chapter 5 ML Model Figures")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Figures from saved CSVs (no retraining needed)
    fig1_model_comparison()
    fig4_ablation()
    fig5_results_progression()
    fig8_feature_selection()
    fig9_ml_vs_interpolation()
    update_csv()

    # Figures requiring retraining (build features once)
    print("\n" + "=" * 70)
    print("Building feature matrix for retraining plots...")
    print("=" * 70)
    features_df, feature_cols = build_features()

    fig2_feature_importance(features_df, feature_cols)
    fig3_predictions_scatter(features_df, feature_cols)
    fig6_per_fold_cv(features_df, feature_cols)
    fig7_random_vs_spatial_cv(features_df, feature_cols)

    print("\n" + "=" * 70)
    print("ALL FIGURES GENERATED")
    print("=" * 70)
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
