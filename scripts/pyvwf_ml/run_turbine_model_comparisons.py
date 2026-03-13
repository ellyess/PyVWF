#!/usr/bin/env python3
"""Run turbine-level model comparisons for Chapter 5 verification.

Produces three sets of model_comparison CSVs:
  1. 35 features, default hyperparameters  → turbine_35feat_default/
  2. 35 features, tuned hyperparameters    → turbine_35feat_tuned/
  3. 7 Lasso-selected features, tuned      → turbine_7feat_tuned/

All use 5-fold spatial_lon CV on the 23,009-turbine dataset.
"""

import os
import sys
from pathlib import Path

# Unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from vwf.ml_correction import (
    build_turbine_level_dataset,
    compare_interpolation_methods,
    create_feature_matrix,
    select_important_features,
)

# Paths (matching generate_ch5_ml_plots.py)
CORRECTIONS_CSV = PROJECT_ROOT / "output" / "pyvwf_to_grid" / "all_corrections_centroids.csv"
TERRAIN_NC = PROJECT_ROOT / "input" / "terrain" / "terrain_europe_enhanced.nc"
COASTLINE = PROJECT_ROOT / "input" / "terrain" / "coastlines.geojson"
INVARIANT_NC = PROJECT_ROOT / "input" / "era5" / "invariant" / "era5_invariant_europe.nc"
CORINE_NC = PROJECT_ROOT / "input" / "terrain" / "corine_europe.nc"
TURBINE_DIR = PROJECT_ROOT / "input" / "turbine_level_data"
ERA5_WIND_DIR = PROJECT_ROOT / "input" / "era5" / "EU"
DE_GEO = TURBINE_DIR / "DE" / "geolocate.germany.csv"

OUTPUT_BASE = PROJECT_ROOT / "output" / "pyvwf_ml"

# Exclude columns
EXCLUDE = {
    'scalar', 'offset', 'lon', 'lat', 'country_code', 'country_name',
    'cluster', 'cluster_mode', 'obs_level', 'area_km2',
    'land_cover_class', 'geometry', 'ID', 'turbine_id', 'domain', 'type',
}
SPATIAL_COLS = {'lat_norm', 'lon_norm'}

# The 7 features selected by Lasso (from turbine_level_corine CSVs)
LASSO_7_FEATURES = [
    'era5_wind_night_mean',
    'subgrid_variance',
    'era5_weibull_k',
    'era5_wind_seasonal_range',
    'is_forest',
    'aspect_category',
    'curvature',
]


def load_turbine_metadata(turbine_dir):
    """Load turbine metadata from DK/UK/DE."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "pyvwf_ml"))
    from train_unified_ml_corrections import load_turbine_metadata as _load
    return _load(turbine_dir, raw_de=True)


def build_features():
    """Build the turbine-level feature matrix."""
    print("=" * 70, flush=True)
    print("BUILDING TURBINE-LEVEL FEATURE MATRIX", flush=True)
    print("=" * 70, flush=True)

    corrections = pd.read_csv(CORRECTIONS_CSV)
    print(f"  Loaded {len(corrections)} correction centroids", flush=True)

    turbine_metadata = load_turbine_metadata(TURBINE_DIR)
    training_df = build_turbine_level_dataset(
        corrections, turbine_metadata, de_geolocate=DE_GEO,
    )
    print(f"  Turbine-level dataset: {len(training_df)} samples", flush=True)

    features_df = create_feature_matrix(
        training_df,
        terrain_nc=TERRAIN_NC,
        coastline_geojson=COASTLINE,
        invariant_nc=INVARIANT_NC,
        corine_nc=CORINE_NC,
        era5_wind_dir=ERA5_WIND_DIR,
        era5_wind10_dir=ERA5_WIND_DIR,
        turbine_metadata=None,  # turbine features already per-row
    )

    # Get feature columns (exclude spatial for turbine-level)
    feature_cols = [
        c for c in features_df.columns
        if c not in EXCLUDE and c not in SPATIAL_COLS
    ]
    feature_cols = [c for c in feature_cols if c in features_df.columns]
    print(f"  Features ({len(feature_cols)}): {feature_cols}", flush=True)
    return features_df, feature_cols


def run_comparison(features_df, feature_cols, output_dir, tune):
    """Run 8-model comparison for both targets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Output: {output_dir}", flush=True)
    print(f"  Features: {len(feature_cols)}, Tuned: {tune}", flush=True)

    for target in ['scalar', 'offset']:
        print(f"\n  --- {target.upper()} ---", flush=True)
        comparison = compare_interpolation_methods(
            features_df,
            feature_cols=feature_cols,
            target_col=target,
            cv_folds=5,
            cv_strategy='spatial_lon',
            tune_hyperparams=tune,
        )
        out_path = output_dir / f'model_comparison_{target}.csv'
        comparison.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}", flush=True)
        print(comparison.to_string(index=False), flush=True)


def main():
    features_df, all_feature_cols = build_features()
    n_feat = len(all_feature_cols)
    print(f"\n  Total features: {n_feat}")

    # --- Run 1: All features, default hyperparameters ---
    print("\n" + "=" * 70)
    print(f"RUN 1: {n_feat} features, DEFAULT hyperparameters")
    print("=" * 70)
    run_comparison(
        features_df, all_feature_cols,
        OUTPUT_BASE / "turbine_35feat_default",
        tune=False,
    )

    # --- Run 2: All features, tuned hyperparameters ---
    print("\n" + "=" * 70)
    print(f"RUN 2: {n_feat} features, TUNED hyperparameters")
    print("=" * 70)
    run_comparison(
        features_df, all_feature_cols,
        OUTPUT_BASE / "turbine_35feat_tuned",
        tune=True,
    )

    # --- Run 3: 7 Lasso-selected features, tuned ---
    print("\n" + "=" * 70)
    print("RUN 3: 7 Lasso-selected features, TUNED hyperparameters")
    print("=" * 70)
    # Verify selected features exist
    available = [f for f in LASSO_7_FEATURES if f in features_df.columns]
    if len(available) != 7:
        print(f"  WARNING: Only {len(available)} of 7 features available: {available}")
    run_comparison(
        features_df, available,
        OUTPUT_BASE / "turbine_7feat_tuned",
        tune=True,
    )

    # --- Also save a training summary for each ---
    for dirname, n_f, tuned in [
        ("turbine_35feat_default", n_feat, False),
        ("turbine_35feat_tuned", n_feat, True),
        ("turbine_7feat_tuned", len(available), True),
    ]:
        summary_path = OUTPUT_BASE / dirname / "training_summary.txt"
        with open(summary_path, "w") as f:
            f.write("ML Training Summary\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"CV strategy: spatial_lon\n")
            f.write(f"CV folds: 5\n")
            f.write(f"Tuned: {tuned}\n")
            cols = all_feature_cols if n_f == n_feat else available
            f.write(f"Features ({len(cols)}): {cols}\n")
            f.write(f"Samples: {len(features_df)}\n")

    print("\n" + "=" * 70)
    print("ALL RUNS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=0,
                        help="0=all, 1=35feat default, 2=35feat tuned, 3=7feat tuned")
    args = parser.parse_args()

    features_df, all_feature_cols = build_features()
    n_feat = len(all_feature_cols)
    print(f"\n  Total features: {n_feat}", flush=True)

    available_7 = [f for f in LASSO_7_FEATURES if f in features_df.columns]

    if args.run in (0, 1):
        print(f"\n{'='*70}", flush=True)
        print(f"RUN 1: {n_feat} features, DEFAULT hyperparameters", flush=True)
        print("=" * 70, flush=True)
        run_comparison(features_df, all_feature_cols,
                       OUTPUT_BASE / "turbine_35feat_default", tune=False)

    if args.run in (0, 2):
        print(f"\n{'='*70}", flush=True)
        print(f"RUN 2: {n_feat} features, TUNED hyperparameters", flush=True)
        print("=" * 70, flush=True)
        run_comparison(features_df, all_feature_cols,
                       OUTPUT_BASE / "turbine_35feat_tuned", tune=True)

    if args.run in (0, 3):
        print(f"\n{'='*70}", flush=True)
        print("RUN 3: 7 Lasso-selected features, TUNED hyperparameters", flush=True)
        print("=" * 70, flush=True)
        if len(available_7) != 7:
            print(f"  WARNING: Only {len(available_7)} of 7 features available: {available_7}", flush=True)
        run_comparison(features_df, available_7,
                       OUTPUT_BASE / "turbine_7feat_tuned", tune=True)

    # Save training summaries
    for dirname, cols, tuned in [
        ("turbine_35feat_default", all_feature_cols, False),
        ("turbine_35feat_tuned", all_feature_cols, True),
        ("turbine_7feat_tuned", available_7, True),
    ]:
        d = OUTPUT_BASE / dirname
        if d.exists():
            summary_path = d / "training_summary.txt"
            with open(summary_path, "w") as f:
                f.write("ML Training Summary\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"CV strategy: spatial_lon\n")
                f.write(f"CV folds: 5\n")
                f.write(f"Tuned: {tuned}\n")
                f.write(f"Features ({len(cols)}): {cols}\n")
                f.write(f"Samples: {len(features_df)}\n")

    print("\nDONE", flush=True)
