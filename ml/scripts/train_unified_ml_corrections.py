#!/usr/bin/env python3
"""Train ML models on unified European correction factors.

This script uses the unified Voronoi corrections dataset (1,712 regions) to train
ML models that predict correction factors from terrain features.

Key Advantage: Unlike previous attempts, all corrections use CONSISTENT METHODOLOGY,
eliminating country-specific calibration biases that plagued earlier ML efforts.

Workflow:
    1. Load unified corrections (all_corrections_centroids.csv)
    2. Extract terrain features at centroid locations
    3. Train multiple ML models (Random Forest, Gradient Boosting, etc.)
    4. Perform spatial cross-validation
    5. Compare ML to IDW interpolation
    6. Generate predictions on European grid
    7. Export results and diagnostic plots

Usage:
    # Train with default settings
    python train_unified_ml_corrections.py

    # Custom configuration
    python train_unified_ml_corrections.py \
        --model-type gradient_boosting \
        --cv-folds 10 \
        --output-dir ml/unified_ml_v2

    # Compare all models
    python train_unified_ml_corrections.py --compare-models

Requirements:
    - Unified corrections: output/unified_corrections/all_corrections_centroids.csv
    - Terrain data: input/terrain/terrain_north_sea_full.nc
    - Python packages: sklearn, xarray, pandas, matplotlib
"""

import argparse
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


# =============================================================================
# Configuration
# =============================================================================

UNIFIED_CORRECTIONS_CSV = Path("output/unified_corrections/all_corrections_centroids.csv")
TERRAIN_NC = Path("input/terrain/terrain_north_sea_full.nc")
IDW_GRID_NC = Path("output/unified_corrections_grid_comparison/europe_corrections_idw.nc")

MODEL_TYPES = {
    'random_forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),
    'gradient_boosting': GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ),
    'ridge': Ridge(alpha=1.0, random_state=42),
    'lasso': Lasso(alpha=0.1, random_state=42, max_iter=5000),
    'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=5000),
}


# =============================================================================
# Data Loading
# =============================================================================

def load_unified_corrections():
    """Load unified corrections from CSV.

    Returns:
        DataFrame with columns: lon, lat, scalar, offset, country_code, etc.
    """
    if not UNIFIED_CORRECTIONS_CSV.exists():
        raise FileNotFoundError(
            f"Unified corrections not found: {UNIFIED_CORRECTIONS_CSV}\n"
            f"Run create_unified_correction_dataframe.py first"
        )

    df = pd.read_csv(UNIFIED_CORRECTIONS_CSV)

    print(f"✓ Loaded {len(df)} correction centroids")
    print(f"  Countries: {df['country_code'].unique()}")
    print(f"  Scalar range: [{df['scalar'].min():.3f}, {df['scalar'].max():.3f}]")
    print(f"  Offset range: [{df['offset'].min():.2f}, {df['offset'].max():.2f}] m/s")

    return df


def load_terrain_data():
    """Load terrain features from NetCDF.

    Returns:
        xarray Dataset with terrain variables.
    """
    if not TERRAIN_NC.exists():
        raise FileNotFoundError(
            f"Terrain data not found: {TERRAIN_NC}\n"
            f"Run ml/quick_terrain_setup.py or ml/download_terrain_data.py"
        )

    ds = xr.open_dataset(TERRAIN_NC)

    print(f"\n✓ Loaded terrain data")
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Coverage: lon [{ds.lon.min().item():.1f}, {ds.lon.max().item():.1f}], "
          f"lat [{ds.lat.min().item():.1f}, {ds.lat.max().item():.1f}]")

    return ds


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_terrain_features(corrections_df, terrain_ds):
    """Extract terrain features at correction centroid locations.

    Args:
        corrections_df: DataFrame with lon, lat columns.
        terrain_ds: xarray Dataset with terrain variables.

    Returns:
        DataFrame with added terrain feature columns.
    """
    print("\nExtracting terrain features...")

    features_df = corrections_df.copy()

    # Extract each terrain variable
    for var_name in terrain_ds.data_vars:
        print(f"  - {var_name}")

        # Create xarray DataArray for point locations
        lons = xr.DataArray(features_df['lon'].values, dims='points')
        lats = xr.DataArray(features_df['lat'].values, dims='points')

        # Interpolate - this returns a 1D array
        values = terrain_ds[var_name].interp(
            lon=lons,
            lat=lats,
            method='linear'
        ).values

        features_df[f'terrain_{var_name}'] = values

    # Add spatial features
    print("  - spatial features")
    features_df['abs_lat'] = np.abs(features_df['lat'])
    features_df['lon_normalized'] = (features_df['lon'] - features_df['lon'].mean()) / features_df['lon'].std()
    features_df['lat_normalized'] = (features_df['lat'] - features_df['lat'].mean()) / features_df['lat'].std()

    # Check for NaN values and fill with appropriate defaults
    n_nan = features_df.isnull().sum().sum()
    if n_nan > 0:
        print(f"\n  ⚠ Found {n_nan} NaN values - filling with feature medians")
        # Fill NaN values with median for each feature column
        for col in features_df.columns:
            if features_df[col].isnull().any():
                if col.startswith('terrain_'):
                    # For terrain features, use median
                    median_val = features_df[col].median()
                    if pd.isna(median_val):
                        # If all values are NaN, use 0
                        features_df[col] = features_df[col].fillna(0)
                    else:
                        features_df[col] = features_df[col].fillna(median_val)
                else:
                    # For other features (scalar, offset, coordinates), use median
                    features_df[col] = features_df[col].fillna(features_df[col].median())

    print(f"\n✓ Extracted features for {len(features_df)} centroids")

    return features_df


# =============================================================================
# Model Training
# =============================================================================

def train_model(X_train, y_train, model_type='random_forest', scale_features=True):
    """Train an ML model.

    Args:
        X_train: Feature matrix (n_samples, n_features).
        y_train: Target values (n_samples,).
        model_type: Model type string.
        scale_features: Whether to scale features.

    Returns:
        Trained model and scaler (if used).
    """
    # Get model
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(MODEL_TYPES.keys())}")

    model = MODEL_TYPES[model_type]

    # Scale features if needed
    scaler = None
    if scale_features and model_type in ['ridge', 'lasso', 'elastic_net']:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    # Train
    model.fit(X_train, y_train)

    return model, scaler


def cross_validate_model(X, y, model_type='random_forest', cv_folds=5):
    """Perform cross-validation.

    Args:
        X: Feature matrix.
        y: Target values.
        model_type: Model type string.
        cv_folds: Number of CV folds.

    Returns:
        Dictionary with CV scores.
    """
    model = MODEL_TYPES[model_type]

    # Scale if needed
    if model_type in ['ridge', 'lasso', 'elastic_net']:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Perform CV
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    scores = {
        'r2': cross_val_score(model, X, y, cv=cv, scoring='r2'),
        'mae': -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error'),
        'rmse': np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')),
    }

    return scores


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_predictions(y_true, y_pred, target_name='scalar'):
    """Evaluate model predictions.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        target_name: Name of target variable.

    Returns:
        Dictionary with metrics.
    """
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'bias': np.mean(y_pred - y_true),
    }

    print(f"\n{target_name.upper()} Prediction Metrics:")
    print(f"  R² = {metrics['r2']:.4f}")
    print(f"  MAE = {metrics['mae']:.4f}")
    print(f"  RMSE = {metrics['rmse']:.4f}")
    print(f"  Bias = {metrics['bias']:.4f}")

    return metrics


def compare_to_idw(features_df, ml_predictions, target='scalar'):
    """Compare ML predictions to IDW interpolation.

    Args:
        features_df: DataFrame with corrections.
        ml_predictions: ML model predictions.
        target: 'scalar' or 'offset'.

    Returns:
        Dictionary with comparison metrics.
    """
    print(f"\n{'='*70}")
    print(f"Comparing ML to IDW for {target.upper()}")
    print(f"{'='*70}")

    # Load IDW grid
    if not IDW_GRID_NC.exists():
        print("  ✗ IDW grid not found - skipping comparison")
        return None

    idw_ds = xr.open_dataset(IDW_GRID_NC)

    # Interpolate IDW to correction centroid locations using xarray DataArrays
    lons = xr.DataArray(features_df['lon'].values, dims='points')
    lats = xr.DataArray(features_df['lat'].values, dims='points')

    idw_values = idw_ds[target].interp(
        lon=lons,
        lat=lats,
        method='linear'
    ).values

    # True values
    true_values = features_df[target].values

    # Filter out NaN values (points outside IDW coverage)
    valid_mask = ~np.isnan(idw_values)
    n_invalid = (~valid_mask).sum()

    if n_invalid > 0:
        print(f"\n  ⚠ {n_invalid} points outside IDW coverage - filtering")
        idw_values = idw_values[valid_mask]
        true_values_filtered = true_values[valid_mask]
        ml_predictions_filtered = ml_predictions[valid_mask]
    else:
        true_values_filtered = true_values
        ml_predictions_filtered = ml_predictions

    # Ensure we have enough points
    if len(idw_values) < 10:
        print(f"  ✗ Too few overlap points ({len(idw_values)}) - skipping IDW comparison")
        return None

    # Compute metrics for ML (on filtered points)
    ml_r2 = r2_score(true_values_filtered, ml_predictions_filtered)
    ml_mae = mean_absolute_error(true_values_filtered, ml_predictions_filtered)
    ml_rmse = np.sqrt(mean_squared_error(true_values_filtered, ml_predictions_filtered))

    print(f"\nML {target.upper()} Metrics (on {len(idw_values)} overlap points):")
    print(f"  R² = {ml_r2:.4f}")
    print(f"  MAE = {ml_mae:.4f}")
    print(f"  RMSE = {ml_rmse:.4f}")

    # Compute metrics for IDW
    print(f"\nIDW {target.upper()} Metrics:")
    idw_r2 = r2_score(true_values_filtered, idw_values)
    idw_mae = mean_absolute_error(true_values_filtered, idw_values)
    idw_rmse = np.sqrt(mean_squared_error(true_values_filtered, idw_values))

    print(f"  R² = {idw_r2:.4f}")
    print(f"  MAE = {idw_mae:.4f}")
    print(f"  RMSE = {idw_rmse:.4f}")

    # Comparison
    print(f"\nML vs IDW:")
    print(f"  Δ R² = {ml_r2 - idw_r2:+.4f} {'✓ ML better' if ml_r2 > idw_r2 else '✗ IDW better'}")
    print(f"  Δ MAE = {ml_mae - idw_mae:+.4f} {'✓ ML better' if ml_mae < idw_mae else '✗ IDW better'}")

    return {
        'ml': {'r2': ml_r2, 'mae': ml_mae, 'rmse': ml_rmse},
        'idw': {'r2': idw_r2, 'mae': idw_mae, 'rmse': idw_rmse},
        'n_overlap': len(idw_values)
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_predictions(y_true, y_pred, target_name='scalar', save_path=None):
    """Plot actual vs predicted values.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        target_name: Name of target variable.
        save_path: Path to save figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)

    # 1:1 line
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max())
    ]
    ax.plot(lims, lims, 'r--', lw=2, alpha=0.7, label='1:1 line')

    # Metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    ax.set_xlabel(f'True {target_name}', fontsize=11)
    ax.set_ylabel(f'Predicted {target_name}', fontsize=11)
    ax.set_title(f'ML Predictions: {target_name.upper()}\nR² = {r2:.3f}, MAE = {mae:.3f}',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Residuals
    ax = axes[1]
    residuals = y_pred - y_true
    ax.scatter(y_true, residuals, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    ax.axhline(0, color='r', linestyle='--', lw=2, alpha=0.7)

    ax.set_xlabel(f'True {target_name}', fontsize=11)
    ax.set_ylabel('Residual (Predicted - True)', fontsize=11)
    ax.set_title(f'Residuals\nBias = {residuals.mean():.3f}, Std = {residuals.std():.3f}',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")

    return fig


def plot_feature_importance(model, feature_names, target_name='scalar', save_path=None):
    """Plot feature importance.

    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names: List of feature names.
        target_name: Name of target variable.
        save_path: Path to save figure.
    """
    if not hasattr(model, 'feature_importances_'):
        print("  ⚠ Model does not have feature importance")
        return None

    # Get importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(range(len(indices)), importances[indices], align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title(f'Feature Importance for {target_name.upper()} Prediction',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")

    return fig


def plot_spatial_predictions(features_df, y_pred, target_name='scalar', save_path=None):
    """Plot predictions on map.

    Args:
        features_df: DataFrame with lon, lat columns.
        y_pred: Predicted values.
        target_name: Name of target variable.
        save_path: Path to save figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # True values
    ax = axes[0]
    sc = ax.scatter(
        features_df['lon'], features_df['lat'],
        c=features_df[target_name], s=30,
        cmap='RdYlGn' if target_name == 'scalar' else 'RdBu_r',
        edgecolors='k', linewidths=0.5
    )
    ax.set_xlabel('Longitude (°E)', fontsize=10)
    ax.set_ylabel('Latitude (°N)', fontsize=10)
    ax.set_title(f'True {target_name.upper()}', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label=target_name)

    # Predicted values
    ax = axes[1]
    sc = ax.scatter(
        features_df['lon'], features_df['lat'],
        c=y_pred, s=30,
        cmap='RdYlGn' if target_name == 'scalar' else 'RdBu_r',
        edgecolors='k', linewidths=0.5
    )
    ax.set_xlabel('Longitude (°E)', fontsize=10)
    ax.set_ylabel('Latitude (°N)', fontsize=10)
    ax.set_title(f'ML Predicted {target_name.upper()}', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label=target_name)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")

    return fig


# =============================================================================
# Main Workflow
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train ML models on unified European correction factors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--model-type',
        choices=list(MODEL_TYPES.keys()),
        default='random_forest',
        help='ML model type (default: random_forest)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('ml/unified_ml'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--compare-models',
        action='store_true',
        help='Compare all model types'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    parser.add_argument(
        '--terrain-nc',
        type=Path,
        default=TERRAIN_NC,
        help=f'Path to terrain NetCDF file (default: {TERRAIN_NC})'
    )
    parser.add_argument(
        '--validation-countries',
        type=str,
        default=None,
        help='Comma-separated list of countries to hold out for validation (e.g., "DE-onshore,UK-onshore")'
    )
    parser.add_argument(
        '--exclude-spatial-features',
        action='store_true',
        help='Exclude spatial features (lon/lat) - use only terrain features for true terrain learning test'
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("ML Training on Unified European Corrections")
    print("="*70)
    print(f"\nModel type: {args.model_type}")
    print(f"CV folds: {args.cv_folds}")
    print(f"Output directory: {args.output_dir}")
    print(f"Terrain file: {args.terrain_nc}")

    # Step 1: Load data
    print("\n" + "="*70)
    print("STEP 1: Load Data")
    print("="*70)

    corrections_df = load_unified_corrections()

    # Load terrain from specified path
    if not args.terrain_nc.exists():
        print(f"\n✗ Terrain file not found: {args.terrain_nc}")
        print(f"  Run ml/quick_terrain_setup.py or ml/enhance_terrain_features.py first")
        return 1

    terrain_ds = xr.open_dataset(args.terrain_nc)

    print(f"\n✓ Loaded terrain data")
    print(f"  Variables: {list(terrain_ds.data_vars)}")
    print(f"  Coverage: lon [{terrain_ds.lon.min().item():.1f}, {terrain_ds.lon.max().item():.1f}], "
          f"lat [{terrain_ds.lat.min().item():.1f}, {terrain_ds.lat.max().item():.1f}]")

    # Step 2: Extract features
    print("\n" + "="*70)
    print("STEP 2: Extract Terrain Features")
    print("="*70)

    features_df = extract_terrain_features(corrections_df, terrain_ds)

    # Separate features and targets
    if args.exclude_spatial_features:
        # Use only terrain features (no lon/lat/abs_lat)
        feature_cols = [col for col in features_df.columns if col.startswith('terrain_')]
        print(f"\n⚠️  EXCLUDING SPATIAL FEATURES - Using only terrain features for pure terrain learning test")
    else:
        # Use all features including spatial
        feature_cols = [col for col in features_df.columns if col.startswith('terrain_') or
                       col.endswith('_normalized') or col == 'abs_lat']

    # Spatial cross-validation: hold out validation countries
    if args.validation_countries:
        validation_countries = [c.strip() for c in args.validation_countries.split(',')]

        print(f"\n{'='*70}")
        print("SPATIAL CROSS-VALIDATION")
        print(f"{'='*70}")
        print(f"\nValidation countries (held out): {', '.join(validation_countries)}")

        # Split data
        val_mask = features_df['country_code'].isin(validation_countries)
        train_features = features_df[~val_mask].copy()
        val_features = features_df[val_mask].copy()

        print(f"\nTraining countries: {', '.join(train_features['country_code'].unique())}")
        print(f"Training samples: {len(train_features)}")
        print(f"Validation samples: {len(val_features)}")

        if len(val_features) == 0:
            print(f"\n✗ No validation samples found for countries: {validation_countries}")
            print(f"Available countries: {features_df['country_code'].unique()}")
            return 1

        # Training and validation sets
        X_train = train_features[feature_cols].values
        y_train_scalar = train_features['scalar'].values
        y_train_offset = train_features['offset'].values

        X_val = val_features[feature_cols].values
        y_val_scalar = val_features['scalar'].values
        y_val_offset = val_features['offset'].values

        # Use spatial validation instead of standard CV
        use_spatial_validation = True

    else:
        # Standard approach: use all data
        X = features_df[feature_cols].values
        y_scalar = features_df['scalar'].values
        y_offset = features_df['offset'].values
        use_spatial_validation = False

    print(f"\nFeature matrix shape: {X_train.shape if use_spatial_validation else X.shape}")
    print(f"Features: {feature_cols}")

    # Step 3: Train models
    print("\n" + "="*70)
    print("STEP 3: Train ML Models")
    print("="*70)

    results = {}

    if args.compare_models:
        # Compare all model types
        print("\nComparing all model types...")

        comparison_results = []

        for model_name in MODEL_TYPES.keys():
            print(f"\n{'─'*70}")
            print(f"Model: {model_name}")
            print(f"{'─'*70}")

            if use_spatial_validation:
                # Spatial validation: train on training countries, test on validation countries
                print("\nSpatial validation (held-out countries):")

                # Scalar
                scalar_model, scalar_scaler = train_model(X_train, y_train_scalar, model_name)
                X_val_scaled = scalar_scaler.transform(X_val) if scalar_scaler else X_val
                scalar_pred = scalar_model.predict(X_val_scaled)
                scalar_r2 = r2_score(y_val_scalar, scalar_pred)
                scalar_mae = mean_absolute_error(y_val_scalar, scalar_pred)
                print(f"  Scalar - R² = {scalar_r2:.4f}, MAE = {scalar_mae:.4f}")

                # Offset
                offset_model, offset_scaler = train_model(X_train, y_train_offset, model_name)
                X_val_scaled = offset_scaler.transform(X_val) if offset_scaler else X_val
                offset_pred = offset_model.predict(X_val_scaled)
                offset_r2 = r2_score(y_val_offset, offset_pred)
                offset_mae = mean_absolute_error(y_val_offset, offset_pred)
                print(f"  Offset - R² = {offset_r2:.4f}, MAE = {offset_mae:.4f}")

            else:
                # Standard CV
                print("\nScalar CV scores:")
                scalar_scores = cross_validate_model(X, y_scalar, model_name, args.cv_folds)
                scalar_r2 = scalar_scores['r2'].mean()
                scalar_mae = scalar_scores['mae'].mean()
                print(f"  R² = {scalar_r2:.4f} ± {scalar_scores['r2'].std():.4f}")
                print(f"  MAE = {scalar_mae:.4f} ± {scalar_scores['mae'].std():.4f}")

                print("\nOffset CV scores:")
                offset_scores = cross_validate_model(X, y_offset, model_name, args.cv_folds)
                offset_r2 = offset_scores['r2'].mean()
                offset_mae = offset_scores['mae'].mean()
                print(f"  R² = {offset_r2:.4f} ± {offset_scores['r2'].std():.4f}")
                print(f"  MAE = {offset_mae:.4f} ± {offset_scores['mae'].std():.4f}")

            comparison_results.append({
                'model': model_name,
                'scalar_r2': scalar_r2,
                'scalar_mae': scalar_mae,
                'offset_r2': offset_r2,
                'offset_mae': offset_mae,
            })

        # Save comparison
        comparison_df = pd.DataFrame(comparison_results)
        comparison_path = args.output_dir / 'model_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\n✓ Saved model comparison: {comparison_path}")

        # Print summary table
        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY")
        print("="*70)
        print(comparison_df.to_string(index=False))

    else:
        # Train single model
        print(f"\nTraining {args.model_type} models...")

        if use_spatial_validation:
            # Spatial validation
            print("\n--- SCALAR CORRECTION (Spatial Validation) ---")
            print(f"\nTraining on {len(X_train)} samples from {len(train_features['country_code'].unique())} countries...")
            print(f"Testing on {len(X_val)} samples from {len(validation_countries)} countries...")

            scalar_model, scalar_scaler = train_model(X_train, y_train_scalar, args.model_type)

            # Predict on validation set
            X_val_scaled = scalar_scaler.transform(X_val) if scalar_scaler else X_val
            scalar_predictions = scalar_model.predict(X_val_scaled)

            # Evaluate
            scalar_metrics = evaluate_predictions(y_val_scalar, scalar_predictions, 'scalar')

            # Offset
            print("\n--- OFFSET CORRECTION (Spatial Validation) ---")
            print(f"\nTraining on {len(X_train)} samples...")
            print(f"Testing on {len(X_val)} samples...")

            offset_model, offset_scaler = train_model(X_train, y_train_offset, args.model_type)

            # Predict on validation set
            X_val_scaled = offset_scaler.transform(X_val) if offset_scaler else X_val
            offset_predictions = offset_model.predict(X_val_scaled)

            # Evaluate
            offset_metrics = evaluate_predictions(y_val_offset, offset_predictions, 'offset')

            # Store results for plotting
            results = {
                'scalar_model': scalar_model,
                'scalar_scaler': scalar_scaler,
                'scalar_predictions': scalar_predictions,
                'scalar_true': y_val_scalar,
                'offset_model': offset_model,
                'offset_scaler': offset_scaler,
                'offset_predictions': offset_predictions,
                'offset_true': y_val_offset,
                'features_df': val_features,  # Use validation set for plots
            }

        else:
            # Standard CV
            print("\n--- SCALAR CORRECTION ---")
            print("\nCross-validation:")
            scalar_cv_scores = cross_validate_model(X, y_scalar, args.model_type, args.cv_folds)
            print(f"  R² = {scalar_cv_scores['r2'].mean():.4f} ± {scalar_cv_scores['r2'].std():.4f}")
            print(f"  MAE = {scalar_cv_scores['mae'].mean():.4f} ± {scalar_cv_scores['mae'].std():.4f}")
            print(f"  RMSE = {scalar_cv_scores['rmse'].mean():.4f} ± {scalar_cv_scores['rmse'].std():.4f}")

            print("\nTraining final model on all data:")
            scalar_model, scalar_scaler = train_model(X, y_scalar, args.model_type)

            # Predict
            X_scaled = scalar_scaler.transform(X) if scalar_scaler else X
            scalar_predictions = scalar_model.predict(X_scaled)

            # Evaluate
            scalar_metrics = evaluate_predictions(y_scalar, scalar_predictions, 'scalar')

            # Compare to IDW
            scalar_comparison = compare_to_idw(features_df, scalar_predictions, 'scalar')

            # Offset model
            print("\n--- OFFSET CORRECTION ---")
            print("\nCross-validation:")
            offset_cv_scores = cross_validate_model(X, y_offset, args.model_type, args.cv_folds)
            print(f"  R² = {offset_cv_scores['r2'].mean():.4f} ± {offset_cv_scores['r2'].std():.4f}")
            print(f"  MAE = {offset_cv_scores['mae'].mean():.4f} ± {offset_cv_scores['mae'].std():.4f}")
            print(f"  RMSE = {offset_cv_scores['rmse'].mean():.4f} ± {offset_cv_scores['rmse'].std():.4f}")

            print("\nTraining final model on all data:")
            offset_model, offset_scaler = train_model(X, y_offset, args.model_type)

            # Predict
            X_scaled = offset_scaler.transform(X) if offset_scaler else X
            offset_predictions = offset_model.predict(X_scaled)

            # Evaluate
            offset_metrics = evaluate_predictions(y_offset, offset_predictions, 'offset')

            # Compare to IDW
            offset_comparison = compare_to_idw(features_df, offset_predictions, 'offset')

            # Store results
            results = {
                'scalar_model': scalar_model,
                'scalar_scaler': scalar_scaler,
                'scalar_predictions': scalar_predictions,
                'scalar_true': y_scalar,
                'scalar_cv_scores': scalar_cv_scores,
                'offset_model': offset_model,
                'offset_scaler': offset_scaler,
                'offset_predictions': offset_predictions,
                'offset_true': y_offset,
                'offset_cv_scores': offset_cv_scores,
                'features_df': features_df,
            }

        # Step 4: Generate plots
        if not args.no_plots:
            print("\n" + "="*70)
            print("STEP 4: Generate Diagnostic Plots")
            print("="*70)

            plots_dir = args.output_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)

            # Get true values from results
            y_scalar_plot = results.get('scalar_true', results.get('scalar_predictions'))
            y_offset_plot = results.get('offset_true', results.get('offset_predictions'))          # Use stored true values
            features_for_plot = results['features_df']

            # Scalar plots
            print("\nScalar plots:")
            plot_predictions(
                y_scalar_plot, results['scalar_predictions'], 'scalar',
                plots_dir / 'scalar_predictions.png'
            )
            plot_spatial_predictions(
                features_for_plot, results['scalar_predictions'], 'scalar',
                plots_dir / 'scalar_spatial.png'
            )
            if args.model_type in ['random_forest', 'gradient_boosting']:
                plot_feature_importance(
                    results['scalar_model'], feature_cols, 'scalar',
                    plots_dir / 'scalar_feature_importance.png'
                )

            # Offset plots
            print("\nOffset plots:")
            plot_predictions(
                y_offset_plot, results['offset_predictions'], 'offset',
                plots_dir / 'offset_predictions.png'
            )
            plot_spatial_predictions(
                features_for_plot, results['offset_predictions'], 'offset',
                plots_dir / 'offset_spatial.png'
            )
            if args.model_type in ['random_forest', 'gradient_boosting']:
                plot_feature_importance(
                    results['offset_model'], feature_cols, 'offset',
                    plots_dir / 'offset_feature_importance.png'
                )

            plt.close('all')

        # Save summary
        summary_path = args.output_dir / 'training_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("ML Training Summary - Unified European Corrections\n")
            f.write("="*70 + "\n\n")
            f.write(f"Model: {args.model_type}\n")

            # Note if spatial features were excluded
            if args.exclude_spatial_features:
                f.write(f"Feature set: TERRAIN ONLY (spatial features excluded)\n")
            else:
                f.write(f"Feature set: Terrain + Spatial (lon/lat/abs_lat)\n")

            if use_spatial_validation:
                f.write(f"Validation type: Spatial (held-out countries)\n")
                f.write(f"Training countries: {', '.join(train_features['country_code'].unique())}\n")
                f.write(f"Validation countries: {', '.join(validation_countries)}\n")
                f.write(f"Training samples: {len(X_train)}\n")
                f.write(f"Validation samples: {len(X_val)}\n")
            else:
                f.write(f"Validation type: {args.cv_folds}-fold CV\n")
                f.write(f"Training samples: {len(X)}\n")

            f.write(f"Features: {len(feature_cols)}\n\n")

            f.write("SCALAR CORRECTION\n")
            f.write("-" * 70 + "\n")
            if use_spatial_validation:
                f.write(f"Validation R²: {scalar_metrics['r2']:.4f}\n")
                f.write(f"Validation MAE: {scalar_metrics['mae']:.4f}\n")
                f.write(f"Validation RMSE: {scalar_metrics['rmse']:.4f}\n\n")
            else:
                f.write(f"CV R²: {results['scalar_cv_scores']['r2'].mean():.4f} ± {results['scalar_cv_scores']['r2'].std():.4f}\n")
                f.write(f"CV MAE: {results['scalar_cv_scores']['mae'].mean():.4f} ± {results['scalar_cv_scores']['mae'].std():.4f}\n")
                f.write(f"CV RMSE: {results['scalar_cv_scores']['rmse'].mean():.4f} ± {results['scalar_cv_scores']['rmse'].std():.4f}\n\n")

            f.write("OFFSET CORRECTION\n")
            f.write("-" * 70 + "\n")
            if use_spatial_validation:
                f.write(f"Validation R²: {offset_metrics['r2']:.4f}\n")
                f.write(f"Validation MAE: {offset_metrics['mae']:.4f}\n")
                f.write(f"Validation RMSE: {offset_metrics['rmse']:.4f}\n")
            else:
                f.write(f"CV R²: {results['offset_cv_scores']['r2'].mean():.4f} ± {results['offset_cv_scores']['r2'].std():.4f}\n")
                f.write(f"CV MAE: {results['offset_cv_scores']['mae'].mean():.4f} ± {results['offset_cv_scores']['mae'].std():.4f}\n")
                f.write(f"CV RMSE: {results['offset_cv_scores']['rmse'].mean():.4f} ± {results['offset_cv_scores']['rmse'].std():.4f}\n")

        print(f"\n✓ Saved training summary: {summary_path}")

    # Final message
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
