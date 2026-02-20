#!/usr/bin/env python3
"""Train ML models on unified European correction factors.

This script uses the unified Voronoi corrections dataset to train
ML models that predict correction factors from terrain features.

Workflow:
    1. Load unified corrections (all_corrections_centroids.csv)
    2. Create feature matrix (terrain, ERA5, fleet, CORINE, spatial)
    3. Train ML models with configurable CV strategy
    4. Optionally train regional or hybrid IDW+ML models
    5. Generate diagnostic plots and comparison summaries

Usage:
    # Basic training with spatial CV
    python train_unified_ml_corrections.py --cv-strategy spatial_lon

    # With all feature sources
    python train_unified_ml_corrections.py \
        --terrain-nc input/terrain/terrain_europe_enhanced.nc \
        --invariant-nc input/era5_archive/invariant/era5_invariant_europe.nc \
        --coastline input/terrain/coastlines.geojson \
        --cv-strategy spatial_lon

    # Compare models
    python train_unified_ml_corrections.py --compare-models --cv-strategy spatial_lon

    # Hybrid IDW+ML
    python train_unified_ml_corrections.py --hybrid --cv-strategy spatial_lon

    # Regional models
    python train_unified_ml_corrections.py --regional

    # Feature ablation study
    python train_unified_ml_corrections.py --ablation --cv-strategy spatial_lon
"""

import argparse
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, mean_absolute_error

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from vwf.ml_correction import (
    build_turbine_level_dataset,
    create_feature_matrix,
    train_correction_model,
    compare_interpolation_methods,
    select_important_features,
    train_regional_models,
    predict_regional,
    train_hybrid_model,
    compute_idw_at_points,
)


# =============================================================================
# Configuration
# =============================================================================

UNIFIED_CORRECTIONS_CSV = Path("output/pyvwf_to_grid/all_corrections_centroids.csv")
TERRAIN_NC = Path("input/terrain/terrain_europe_full.nc")
COASTLINE_GEOJSON = Path("input/terrain/coastlines.geojson")
INVARIANT_NC = Path("input/era5_archive/invariant/era5_invariant_europe.nc")
IDW_GRID_NC = Path("output/pyvwf_to_grid/grid_comparison/europe_corrections_idw.nc")


# =============================================================================
# Feature Group Definitions
# =============================================================================

FEATURE_GROUPS = {
    'terrain': [
        'elevation', 'slope', 'aspect', 'roughness', 'curvature',
        'distance_to_coast', 'is_coastal', 'subgrid_variance',
        'complexity', 'aspect_category', 'distance_to_coast_km',
    ],
    'era5': [
        'era5_elevation', 'era5_lsm', 'era5_roughness_length',
        'elevation_mismatch', 'era5_wind_mean', 'era5_wind_std',
    ],
    'era5_extended': [
        'era5_wind_shear',
        'era5_wind_winter_mean', 'era5_wind_summer_mean',
        'era5_wind_seasonal_range',
        'era5_weibull_k', 'era5_weibull_a',
        'era5_diurnal_amplitude', 'era5_wind_night_mean',
    ],
    'fleet': [
        'mean_hub_height', 'mean_rotor_diameter', 'mean_capacity',
    ],
    'turbine': [
        'hub_height', 'rotor_diameter', 'capacity', 'specific_power',
    ],
    'corine': [
        'is_urban', 'is_agricultural', 'is_forest', 'is_bare',
        'is_water', 'roughness_from_lc',
    ],
    'spatial': [
        'lat_norm', 'lon_norm',
    ],
}


# =============================================================================
# Turbine Metadata Loading
# =============================================================================

def load_turbine_metadata(turbine_dir: Path, *, raw_de: bool = False) -> dict[str, pd.DataFrame]:
    """Load and standardize turbine metadata from country CSVs.

    Args:
        turbine_dir: Directory containing DK/, UK/, DE/ subdirectories.
        raw_de: If True, keep raw DE columns (V1, kW, etc.) for
            geolocation via postcode.

    Returns dict mapping country prefix to standardized DataFrame.
    """
    metadata = {}

    # Denmark
    dk_path = turbine_dir / 'DK' / 'dk_md.csv'
    if dk_path.exists():
        dk = pd.read_csv(dk_path)
        metadata['DK'] = pd.DataFrame({
            'height': pd.to_numeric(dk['height'], errors='coerce'),
            'diameter': pd.to_numeric(dk['diameter'], errors='coerce'),
            'capacity': pd.to_numeric(dk['capacity'], errors='coerce'),
            'lon': pd.to_numeric(dk['lon'], errors='coerce'),
            'lat': pd.to_numeric(dk['lat'], errors='coerce'),
        })
        print(f"  DK: {len(metadata['DK'])} turbines")

    # UK
    uk_path = turbine_dir / 'UK' / 'uk_md.csv'
    if uk_path.exists():
        uk = pd.read_csv(uk_path)
        metadata['UK'] = pd.DataFrame({
            'height': pd.to_numeric(uk['height'], errors='coerce'),
            'diameter': pd.to_numeric(uk['diameter'], errors='coerce'),
            'capacity': pd.to_numeric(uk['capacity'], errors='coerce'),
            'lon': pd.to_numeric(uk['lon'], errors='coerce'),
            'lat': pd.to_numeric(uk['lat'], errors='coerce'),
        })
        print(f"  UK: {len(metadata['UK'])} turbines")

    # Germany (no coordinates)
    de_path = turbine_dir / 'DE' / 'DE_md.csv'
    if de_path.exists():
        de = pd.read_csv(de_path)
        if raw_de:
            # Keep raw columns so build_turbine_level_dataset can
            # geolocate via postcode from V1
            de_df = de[['V1', 'kW', 'Rotor..m.', 'Tower..m.']].copy()
            de_df = de_df.dropna(subset=['kW', 'Rotor..m.', 'Tower..m.'])
            de_df['kW'] = pd.to_numeric(de_df['kW'], errors='coerce')
            de_df['Rotor..m.'] = pd.to_numeric(de_df['Rotor..m.'], errors='coerce')
            de_df['Tower..m.'] = pd.to_numeric(de_df['Tower..m.'], errors='coerce')
            de_df = de_df.dropna()
            de_df = de_df[
                (de_df['kW'] > 0) & (de_df['kW'] < 20000)
                & (de_df['Rotor..m.'] > 5) & (de_df['Rotor..m.'] < 300)
                & (de_df['Tower..m.'] > 10) & (de_df['Tower..m.'] < 300)
            ]
            metadata['DE'] = de_df
        else:
            de_df = pd.DataFrame({
                'height': pd.to_numeric(de['Tower..m.'], errors='coerce'),
                'diameter': pd.to_numeric(de['Rotor..m.'], errors='coerce'),
                'capacity': pd.to_numeric(de['kW'], errors='coerce'),
            })
            # Filter obvious outliers
            de_df = de_df[
                (de_df['capacity'] > 0) & (de_df['capacity'] < 20000)
                & (de_df['diameter'] > 5) & (de_df['diameter'] < 300)
                & (de_df['height'] > 10) & (de_df['height'] < 300)
            ]
            metadata['DE'] = de_df
        print(f"  DE: {len(metadata['DE'])} turbines {'(raw for geolocation)' if raw_de else '(no coordinates)'}")

    return metadata


# =============================================================================
# Visualization
# =============================================================================

def plot_predictions(y_true, y_pred, target_name, save_path):
    """Plot actual vs predicted with residuals."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', lw=2, alpha=0.7)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    ax.set_xlabel(f'True {target_name}')
    ax.set_ylabel(f'Predicted {target_name}')
    ax.set_title(f'{target_name.upper()}: R²={r2:.3f}, MAE={mae:.3f}')

    ax = axes[1]
    residuals = y_pred - y_true
    ax.scatter(y_true, residuals, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    ax.axhline(0, color='r', linestyle='--', lw=2)
    ax.set_xlabel(f'True {target_name}')
    ax.set_ylabel('Residual')
    ax.set_title(f'Bias={residuals.mean():.3f}, Std={residuals.std():.3f}')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_feature_importance(result, target_name, save_path):
    """Plot feature importance from a model result dict."""
    fi = result.get('feature_importance')
    if fi is None:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(fi) * 0.3)))
    fi_sorted = fi.sort_values('importance', ascending=True)
    ax.barh(fi_sorted['feature'], fi_sorted['importance'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importance: {target_name.upper()}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# =============================================================================
# Feature Ablation
# =============================================================================

def run_ablation(features_df, feature_cols, args):
    """Run feature group ablation study."""
    print("\n" + "=" * 70)
    print("FEATURE ABLATION STUDY")
    print("=" * 70)

    # Determine which feature groups are present
    present_groups = {}
    for group_name, group_cols in FEATURE_GROUPS.items():
        cols_in_data = [c for c in group_cols if c in feature_cols]
        if cols_in_data:
            present_groups[group_name] = cols_in_data

    print(f"Feature groups present: {list(present_groups.keys())}")
    print(f"Total features: {len(feature_cols)}")

    ablation_results = []

    # Baseline: all features
    print("\n--- Baseline (all features) ---")
    baseline = train_correction_model(
        features_df, target_col='scalar', feature_cols=feature_cols,
        model_type=args.model_type, cv_strategy=args.cv_strategy,
        cv_folds=args.cv_folds,
    )
    baseline_r2 = baseline['cv_scores']['test_r2'].mean()
    ablation_results.append({
        'dropped_group': 'none (baseline)',
        'n_features': len(feature_cols),
        'r2': baseline_r2,
        'delta_r2': 0.0,
    })

    # Drop each group
    for group_name, group_cols in present_groups.items():
        remaining = [c for c in feature_cols if c not in group_cols]
        if not remaining:
            continue

        print(f"\n--- Dropping: {group_name} ({len(group_cols)} features) ---")
        result = train_correction_model(
            features_df, target_col='scalar', feature_cols=remaining,
            model_type=args.model_type, cv_strategy=args.cv_strategy,
            cv_folds=args.cv_folds,
        )
        r2 = result['cv_scores']['test_r2'].mean()
        ablation_results.append({
            'dropped_group': group_name,
            'n_features': len(remaining),
            'r2': r2,
            'delta_r2': r2 - baseline_r2,
        })

    ablation_df = pd.DataFrame(ablation_results)
    print("\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)
    print(ablation_df.to_string(index=False))

    ablation_path = args.output_dir / 'ablation_results.csv'
    ablation_df.to_csv(ablation_path, index=False)
    print(f"\nSaved: {ablation_path}")

    return ablation_df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train ML models on unified European correction factors",
    )
    parser.add_argument('--model-type', default='random_forest',
                        choices=['random_forest', 'gradient_boosting',
                                 'xgboost', 'lightgbm', 'ridge', 'lasso',
                                 'elastic_net'],
                        help='ML model type')
    parser.add_argument('--cv-folds', type=int, default=5)
    parser.add_argument('--cv-strategy', default='random',
                        choices=['random', 'spatial_lon', 'leave_country_out'],
                        help='Cross-validation strategy')
    parser.add_argument('--output-dir', type=Path, default=Path('output/pyvwf_ml/unified_ml'))

    # Data sources
    parser.add_argument('--terrain-nc', type=Path, default=TERRAIN_NC)
    parser.add_argument('--coastline', type=Path, default=COASTLINE_GEOJSON)
    parser.add_argument('--invariant-nc', type=Path, default=None,
                        help='ERA5 invariant NetCDF')
    parser.add_argument('--corine-nc', type=Path, default=None,
                        help='CORINE land cover NetCDF')
    parser.add_argument('--turbine-dir', type=Path, default=None,
                        help='Directory with turbine metadata CSVs')
    parser.add_argument('--de-geolocate', type=Path, default=None,
                        help='German postcode geolocation CSV')
    parser.add_argument('--corrections-csv', type=Path,
                        default=UNIFIED_CORRECTIONS_CSV)
    parser.add_argument('--era5-wind-dir', type=Path, default=None,
                        help='Directory with ERA5 u100/v100 NetCDF files')
    parser.add_argument('--era5-wind10-dir', type=Path, default=None,
                        help='Directory with ERA5 u10/v10 NetCDF files '
                             '(default: same as --era5-wind-dir)')

    # Feature control
    parser.add_argument('--feature-groups', type=str, default=None,
                        help='Comma-separated feature groups to use '
                             '(terrain,era5,fleet,corine,spatial)')
    parser.add_argument('--exclude-spatial-features', action='store_true')

    # Model modes
    parser.add_argument('--compare-models', action='store_true')
    parser.add_argument('--regional', action='store_true',
                        help='Train regional per-country models')
    parser.add_argument('--hybrid', action='store_true',
                        help='Train hybrid IDW+ML model')
    parser.add_argument('--ablation', action='store_true',
                        help='Run feature group ablation study')
    parser.add_argument('--turbine-level', action='store_true',
                        help='Train on individual turbines instead of centroids')
    parser.add_argument('--log-target', action='store_true',
                        help='Log-transform scalar target')
    parser.add_argument('--tune', action='store_true',
                        help='Run hyperparameter tuning before evaluation')
    parser.add_argument('--select-features', action='store_true',
                        help='Run feature selection (Lasso) before training')
    parser.add_argument('--select-top-n', type=int, default=None,
                        help='Keep only top N features (default: all non-zero)')
    parser.add_argument('--no-plots', action='store_true')

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ML Training on Unified European Corrections")
    print("=" * 70)
    print(f"  Model:       {args.model_type}")
    print(f"  CV strategy: {args.cv_strategy}")
    print(f"  CV folds:    {args.cv_folds}")
    print(f"  Terrain:     {args.terrain_nc}")
    if args.turbine_level:
        print(f"  Mode:        TURBINE-LEVEL")

    # =========================================================================
    # Step 1: Load corrections
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Load Data")
    print("=" * 70)

    corrections_df = pd.read_csv(args.corrections_csv)
    print(f"  Loaded {len(corrections_df)} correction centroids")
    print(f"  Countries: {corrections_df['country_code'].nunique()}")
    print(f"  Scalar: [{corrections_df['scalar'].min():.3f}, {corrections_df['scalar'].max():.3f}]")
    print(f"  Offset: [{corrections_df['offset'].min():.2f}, {corrections_df['offset'].max():.2f}] m/s")

    # =========================================================================
    # Step 2: Build feature matrix
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Build Feature Matrix")
    print("=" * 70)

    # Prepare optional data sources
    terrain_nc = args.terrain_nc if args.terrain_nc.exists() else None
    coastline = args.coastline if args.coastline.exists() else None
    invariant_nc = args.invariant_nc
    if invariant_nc is None and INVARIANT_NC.exists():
        invariant_nc = INVARIANT_NC
    corine_nc = args.corine_nc
    era5_wind_dir = args.era5_wind_dir
    if era5_wind_dir is None:
        default_wind = Path('input/era5_archive/EU')
        if default_wind.exists():
            era5_wind_dir = default_wind
    era5_wind10_dir = args.era5_wind10_dir
    if era5_wind10_dir is None and era5_wind_dir is not None:
        era5_wind10_dir = era5_wind_dir

    # Turbine-level mode: expand centroids to individual turbines
    if args.turbine_level:
        if not args.turbine_dir or not args.turbine_dir.exists():
            print("ERROR: --turbine-level requires --turbine-dir")
            return 1

        print("\nLoading turbine metadata (raw for geolocation)...")
        turbine_metadata = load_turbine_metadata(args.turbine_dir, raw_de=True)

        # Resolve DE geolocate path
        de_geo = args.de_geolocate
        if de_geo is None:
            default_geo = args.turbine_dir / 'DE' / 'geolocate.germany.csv'
            if default_geo.exists():
                de_geo = default_geo

        print("\nBuilding turbine-level dataset...")
        training_df = build_turbine_level_dataset(
            corrections_df, turbine_metadata, de_geolocate=de_geo,
        )

        print("\nCreating feature matrix at turbine locations...")
        features_df = create_feature_matrix(
            training_df,
            terrain_nc=terrain_nc,
            coastline_geojson=coastline,
            invariant_nc=invariant_nc,
            corine_nc=corine_nc,
            era5_wind_dir=era5_wind_dir,
            era5_wind10_dir=era5_wind10_dir,
            # No fleet aggregation - turbine features are already per-row
            turbine_metadata=None,
        )
    else:
        # Original centroid-level path
        turbine_metadata = None
        if args.turbine_dir and args.turbine_dir.exists():
            print("\nLoading turbine metadata...")
            turbine_metadata = load_turbine_metadata(args.turbine_dir)

        print("\nCreating feature matrix...")
        features_df = create_feature_matrix(
            corrections_df,
            terrain_nc=terrain_nc,
            coastline_geojson=coastline,
            invariant_nc=invariant_nc,
            corine_nc=corine_nc,
            era5_wind_dir=era5_wind_dir,
            era5_wind10_dir=era5_wind10_dir,
            turbine_metadata=turbine_metadata,
        )

    # Determine feature columns
    exclude = {
        'scalar', 'offset', 'lon', 'lat', 'country_code', 'country_name',
        'cluster', 'cluster_mode', 'obs_level', 'area_km2',
        'land_cover_class', 'geometry', 'ID', 'turbine_id', 'domain', 'type',
    }
    all_feature_cols = [c for c in features_df.columns if c not in exclude]

    # Apply feature group filter
    if args.feature_groups:
        allowed_groups = args.feature_groups.split(',')
        allowed_cols = set()
        for g in allowed_groups:
            allowed_cols.update(FEATURE_GROUPS.get(g.strip(), []))
        feature_cols = [c for c in all_feature_cols if c in allowed_cols]
        print(f"\n  Using feature groups: {allowed_groups}")
    elif args.exclude_spatial_features or args.turbine_level:
        # Turbine-level mode excludes spatial by default
        spatial_cols = set(FEATURE_GROUPS['spatial'])
        feature_cols = [c for c in all_feature_cols if c not in spatial_cols]
        if args.turbine_level:
            print("\n  Turbine-level mode: excluding lat/lon features")
        else:
            print("\n  Excluding spatial features")
    else:
        feature_cols = all_feature_cols

    # Filter to columns that actually exist in the data
    feature_cols = [c for c in feature_cols if c in features_df.columns]

    print(f"\n  Feature matrix: {features_df.shape}")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    # NaN summary
    n_nan = features_df[feature_cols].isna().sum()
    nan_cols = n_nan[n_nan > 0]
    if len(nan_cols) > 0:
        print(f"\n  NaN values (will be filled with median):")
        for col, count in nan_cols.items():
            print(f"    {col}: {count} NaN ({100*count/len(features_df):.1f}%)")

    # =========================================================================
    # Step 3: Train models
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Train ML Models")
    print("=" * 70)

    # Optional feature selection step
    if args.select_features:
        selected_per_target = {}
        for target in ['scalar', 'offset']:
            selected, importance_df = select_important_features(
                features_df,
                feature_cols=feature_cols,
                target_col=target,
                method='lasso',
                cv_folds=args.cv_folds,
                cv_strategy=args.cv_strategy,
                tune_hyperparams=args.tune,
                top_n=args.select_top_n,
            )
            selected_per_target[target] = selected
            importance_df.to_csv(
                args.output_dir / f'feature_importance_{target}.csv', index=False,
            )

        # Use union of features selected for either target
        all_selected = sorted(set(
            selected_per_target['scalar'] + selected_per_target['offset']
        ), key=lambda f: feature_cols.index(f))
        dropped = [f for f in feature_cols if f not in all_selected]

        print(f"\n{'='*60}")
        print("FEATURE SELECTION SUMMARY")
        print(f"{'='*60}")
        print(f"  Scalar selected: {len(selected_per_target['scalar'])}")
        print(f"  Offset selected: {len(selected_per_target['offset'])}")
        print(f"  Union (used):    {len(all_selected)}")
        print(f"  Dropped:         {dropped}")
        print(f"\n  Continuing with {len(all_selected)} features: {all_selected}")

        feature_cols = all_selected

    if args.ablation:
        run_ablation(features_df, feature_cols, args)
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        return 0

    if args.compare_models:
        print("\nComparing all model types...")
        for target in ['scalar', 'offset']:
            print(f"\n{'='*60}")
            print(f"TARGET: {target.upper()}")
            print(f"{'='*60}")
            comparison = compare_interpolation_methods(
                features_df,
                feature_cols=feature_cols,
                target_col=target,
                cv_folds=args.cv_folds,
                cv_strategy=args.cv_strategy,
                tune_hyperparams=args.tune,
            )
            comparison.to_csv(
                args.output_dir / f'model_comparison_{target}.csv', index=False,
            )

    elif args.hybrid:
        print("\nTraining hybrid IDW+ML models...")
        for target in ['scalar', 'offset']:
            print(f"\n{'='*60}")
            print(f"HYBRID IDW+ML: {target.upper()}")
            print(f"{'='*60}")
            hybrid_result = train_hybrid_model(
                features_df,
                target_col=target,
                feature_cols=feature_cols,
                model_type=args.model_type,
                cv_strategy=args.cv_strategy,
                cv_folds=args.cv_folds,
            )

            # Also compute pure IDW baseline
            idw_preds = compute_idw_at_points(features_df, target_col=target)
            idw_mae = np.nanmean(np.abs(features_df[target].values - idw_preds))
            print(f"\n  Pure IDW MAE: {idw_mae:.4f}")
            print(f"  Hybrid MAE:   {hybrid_result['cv_metrics']['mae'].mean():.4f}")

    elif args.regional:
        print("\nTraining regional models...")
        for target in ['scalar', 'offset']:
            print(f"\n{'='*60}")
            print(f"REGIONAL MODELS: {target.upper()}")
            print(f"{'='*60}")
            regional_results = train_regional_models(
                features_df,
                target_col=target,
                feature_cols=feature_cols,
                model_type=args.model_type,
                cv_strategy=args.cv_strategy,
                cv_folds=args.cv_folds,
            )
            print(f"\n  Trained models for: {[k for k in regional_results if k != '_global']}")

    else:
        # Standard single model training
        results = {}
        for target in ['scalar', 'offset']:
            print(f"\n--- {target.upper()} CORRECTION ---")
            use_log = args.log_target and target == 'scalar'
            result = train_correction_model(
                features_df,
                target_col=target,
                feature_cols=feature_cols,
                model_type=args.model_type,
                cv_folds=args.cv_folds,
                cv_strategy=args.cv_strategy,
                log_target=use_log,
                tune_hyperparams=args.tune,
            )
            results[target] = result

        # Generate plots
        if not args.no_plots:
            print("\n" + "=" * 70)
            print("STEP 4: Generate Plots")
            print("=" * 70)

            plots_dir = args.output_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)

            for target in ['scalar', 'offset']:
                result = results[target]
                y_true = features_df[target].values
                X = features_df[feature_cols].copy().fillna(
                    features_df[feature_cols].median()
                )
                y_pred = result['model'].predict(X)

                plot_predictions(
                    y_true, y_pred, target,
                    plots_dir / f'{target}_predictions.png',
                )
                plot_feature_importance(
                    result, target,
                    plots_dir / f'{target}_feature_importance.png',
                )

    # =========================================================================
    # Save summary
    # =========================================================================
    summary_path = args.output_dir / 'training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("ML Training Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {args.model_type}\n")
        f.write(f"CV strategy: {args.cv_strategy}\n")
        f.write(f"CV folds: {args.cv_folds}\n")
        f.write(f"Features ({len(feature_cols)}): {feature_cols}\n\n")
        f.write(f"Data sources:\n")
        f.write(f"  Terrain: {terrain_nc}\n")
        f.write(f"  ERA5 invariant: {invariant_nc}\n")
        f.write(f"  CORINE: {corine_nc}\n")
        f.write(f"  Turbine metadata: {args.turbine_dir}\n")
        f.write(f"  Coastline: {coastline}\n")
    print(f"\nSaved summary: {summary_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Results: {args.output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
