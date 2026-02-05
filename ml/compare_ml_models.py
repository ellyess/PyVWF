#!/usr/bin/env python3
"""
Compare Multiple ML Models for Correction Factor Prediction

This script trains and compares different ML algorithms to find the best model
for predicting correction factors from terrain features.

Models tested:
- Random Forest
- Gradient Boosting
- XGBoost (if available)
- LightGBM (if available)
- Ridge Regression
- Lasso Regression
- Elastic Net
- Support Vector Regression

Usage:
    python compare_ml_models.py [options]
"""

import argparse
import sys
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR

# Optional advanced models
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Import from training script
sys.path.insert(0, str(Path(__file__).parent))
try:
    from train_europe_ml_corrections import (
        TRAINING_COUNTRIES,
        load_country_corrections,
        extract_terrain_features,
        prepare_training_data,
    )
except ImportError:
    print("Error: Could not import from train_europe_ml_corrections.py")
    sys.exit(1)

warnings.filterwarnings('ignore')


# =============================================================================
# Model Configurations
# =============================================================================

def get_model_configs():
    """Define all models to compare."""
    configs = {
        'random_forest': {
            'name': 'Random Forest',
            'model': RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'available': True,
            'scale': True,
        },
        'gradient_boosting': {
            'name': 'Gradient Boosting',
            'model': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'available': True,
            'scale': True,
        },
        'xgboost': {
            'name': 'XGBoost',
            'model': XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ) if HAS_XGBOOST else None,
            'available': HAS_XGBOOST,
            'scale': True,
        },
        'lightgbm': {
            'name': 'LightGBM',
            'model': LGBMRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ) if HAS_LIGHTGBM else None,
            'available': HAS_LIGHTGBM,
            'scale': True,
        },
        'ridge': {
            'name': 'Ridge Regression',
            'model': Ridge(alpha=1.0, random_state=42),
            'available': True,
            'scale': True,
        },
        'lasso': {
            'name': 'Lasso Regression',
            'model': Lasso(alpha=0.1, random_state=42, max_iter=2000),
            'available': True,
            'scale': True,
        },
        'elastic_net': {
            'name': 'Elastic Net',
            'model': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000),
            'available': True,
            'scale': True,
        },
        'svr': {
            'name': 'Support Vector Regression',
            'model': SVR(kernel='rbf', C=1.0, epsilon=0.1),
            'available': True,
            'scale': True,
        },
    }
    
    return configs


# =============================================================================
# Training and Evaluation
# =============================================================================

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, 
                             scaler=None, cv_folds=5):
    """Train a model and compute all metrics."""
    
    # Scale if needed
    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Train
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    # Predict
    start_time = time.time()
    y_pred = model.predict(X_test_scaled)
    pred_time = time.time() - start_time
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cross-validation
    if cv_folds > 0:
        try:
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train,
                cv=cv_folds, scoring='r2', n_jobs=-1
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except Exception as e:
            print(f"    CV failed: {e}")
            cv_mean = np.nan
            cv_std = np.nan
    else:
        cv_mean = np.nan
        cv_std = np.nan
    
    results = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'train_time': train_time,
        'pred_time': pred_time,
        'y_pred': y_pred,
    }
    
    return results


def compare_models(X, y, model_configs, test_size=0.2, cv_folds=5):
    """Compare all available models."""
    
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    results = {}
    
    for model_id, config in model_configs.items():
        if not config['available']:
            print(f"\n✗ {config['name']} - Not available (install required package)")
            continue
        
        print(f"\n{'='*70}")
        print(f"Training: {config['name']}")
        print('='*70)
        
        try:
            scaler = StandardScaler() if config['scale'] else None
            
            model_results = train_and_evaluate_model(
                config['model'],
                X_train, X_test, y_train, y_test,
                scaler=scaler,
                cv_folds=cv_folds
            )
            
            results[model_id] = {
                'name': config['name'],
                'model': config['model'],
                'scaler': scaler,
                **model_results
            }
            
            # Print results
            print(f"\n  Test Set Performance:")
            print(f"    R² Score:  {model_results['r2']:.4f}")
            print(f"    RMSE:      {model_results['rmse']:.4f}")
            print(f"    MAE:       {model_results['mae']:.4f}")
            print(f"    CV R²:     {model_results['cv_mean']:.4f} (+/- {model_results['cv_std']:.4f})")
            print(f"    Train time: {model_results['train_time']:.2f}s")
            print(f"    Pred time:  {model_results['pred_time']:.4f}s")
            
        except Exception as e:
            print(f"\n  ✗ Failed: {e}")
            continue
    
    # Add test data for plotting
    for model_id in results:
        results[model_id]['y_test'] = y_test
    
    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_model_comparison(results, target_name, output_dir):
    """Create comprehensive comparison plots."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Performance metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    model_names = [r['name'] for r in results.values()]
    r2_scores = [r['r2'] for r in results.values()]
    rmse_scores = [r['rmse'] for r in results.values()]
    mae_scores = [r['mae'] for r in results.values()]
    cv_scores = [r['cv_mean'] for r in results.values()]
    
    # R² comparison
    ax = axes[0, 0]
    bars = ax.barh(model_names, r2_scores, color='steelblue', alpha=0.7)
    ax.set_xlabel('R² Score')
    ax.set_title('R² Score Comparison (higher is better)')
    ax.set_xlim([0, 1])
    ax.axvline(x=0.7, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Excellent (0.7)')
    ax.axvline(x=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Good (0.5)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, r2_scores)):
        ax.text(val + 0.02, i, f'{val:.3f}', va='center')
    
    # RMSE comparison
    ax = axes[0, 1]
    bars = ax.barh(model_names, rmse_scores, color='coral', alpha=0.7)
    ax.set_xlabel('RMSE')
    ax.set_title('RMSE Comparison (lower is better)')
    ax.grid(True, alpha=0.3, axis='x')
    for i, (bar, val) in enumerate(zip(bars, rmse_scores)):
        ax.text(val + 0.002, i, f'{val:.3f}', va='center')
    
    # MAE comparison
    ax = axes[1, 0]
    bars = ax.barh(model_names, mae_scores, color='lightgreen', alpha=0.7)
    ax.set_xlabel('MAE')
    ax.set_title('MAE Comparison (lower is better)')
    ax.grid(True, alpha=0.3, axis='x')
    for i, (bar, val) in enumerate(zip(bars, mae_scores)):
        ax.text(val + 0.002, i, f'{val:.3f}', va='center')
    
    # CV R² comparison
    ax = axes[1, 1]
    bars = ax.barh(model_names, cv_scores, color='plum', alpha=0.7)
    ax.set_xlabel('Cross-Validation R²')
    ax.set_title('CV R² Score (5-fold)')
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis='x')
    for i, (bar, val) in enumerate(zip(bars, cv_scores)):
        if not np.isnan(val):
            ax.text(val + 0.02, i, f'{val:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{target_name}_model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {output_dir / f'{target_name}_model_comparison.png'}")
    plt.close()
    
    # 2. Prediction scatter plots
    n_models = len(results)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for ax, (model_id, result) in zip(axes, results.items()):
        y_test = result['y_test']
        y_pred = result['y_pred']
        
        ax.scatter(y_test, y_pred, alpha=0.5, s=20, color='steelblue')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'r--', lw=2, label='Perfect')
        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f"{result['name']}\nR²={result['r2']:.3f}, RMSE={result['rmse']:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(results), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{target_name}_predictions_all_models.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved predictions plot: {output_dir / f'{target_name}_predictions_all_models.png'}")
    plt.close()


def create_comparison_table(results, output_dir):
    """Create comparison table."""
    
    data = []
    for model_id, result in results.items():
        data.append({
            'Model': result['name'],
            'R²': result['r2'],
            'RMSE': result['rmse'],
            'MAE': result['mae'],
            'CV R² Mean': result['cv_mean'],
            'CV R² Std': result['cv_std'],
            'Train Time (s)': result['train_time'],
            'Pred Time (s)': result['pred_time'],
        })
    
    df = pd.DataFrame(data)
    
    # Sort by R² descending
    df = df.sort_values('R²', ascending=False)
    
    # Save to CSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / 'model_comparison_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved comparison table: {csv_path}")
    
    return df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare ML models for correction factors',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--countries',
        type=str,
        default='DK,UK,DE',
        help='Training countries'
    )
    parser.add_argument(
        '--target',
        choices=['scalar', 'offset', 'both'],
        default='scalar',
        help='Target variable'
    )
    parser.add_argument(
        '--terrain-nc',
        type=str,
        default='../input/terrain/terrain_north_sea_full.nc',
        help='Terrain NetCDF file'
    )
    parser.add_argument(
        '--coastline-geojson',
        type=str,
        default='../input/terrain/coastlines.geojson',
        help='Coastline GeoJSON'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='ml_comparison',
        help='Output directory'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Cross-validation folds'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set fraction'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("ML MODEL COMPARISON")
    print("="*70)
    print(f"Training countries: {args.countries}")
    print(f"Target: {args.target}")
    print(f"Output: {output_dir}")
    print("="*70)
    
    # Load data (same as train_europe_ml_corrections.py)
    print("\nLoading correction factors...")
    training_countries = args.countries.split(',')
    
    all_corrections = []
    for country in training_countries:
        if country in TRAINING_COUNTRIES:
            corrections = load_country_corrections(country, TRAINING_COUNTRIES[country])
            if corrections is not None:
                all_corrections.append(corrections)
    
    if not all_corrections:
        print("✗ No correction data loaded!")
        sys.exit(1)
    
    combined_corrections = pd.concat(all_corrections, ignore_index=True)
    print(f"  Total points: {len(combined_corrections):,}")
    
    # Extract terrain features
    print("\nExtracting terrain features...")
    coastline_path = args.coastline_geojson if Path(args.coastline_geojson).exists() else None
    
    features_df = extract_terrain_features(
        combined_corrections,
        args.terrain_nc,
        coastline_geojson=coastline_path
    )
    
    features_df = features_df.dropna()
    print(f"  After removing NaN: {len(features_df):,} points")
    
    # Get model configurations
    model_configs = get_model_configs()
    
    print(f"\nAvailable models: {sum(1 for c in model_configs.values() if c['available'])}/{len(model_configs)}")
    for model_id, config in model_configs.items():
        status = "✓" if config['available'] else "✗"
        print(f"  {status} {config['name']}")
    
    if not HAS_XGBOOST:
        print("\nNote: Install xgboost for additional models: pip install xgboost")
    if not HAS_LIGHTGBM:
        print("Note: Install lightgbm for additional models: pip install lightgbm")
    
    # Compare models for each target
    targets = ['scalar', 'offset'] if args.target == 'both' else [args.target]
    
    all_results = {}
    
    for target in targets:
        print("\n" + "="*70)
        print(f"COMPARING MODELS FOR: {target.upper()}")
        print("="*70)
        
        X, y, feature_cols = prepare_training_data(features_df, target_col=target)
        
        results = compare_models(
            X, y, model_configs,
            test_size=args.test_size,
            cv_folds=args.cv_folds
        )
        
        all_results[target] = results
        
        # Create plots
        plot_model_comparison(results, target, output_dir)
        
        # Create table
        df = create_comparison_table(results, output_dir / target)
        
        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY - {target.upper()}")
        print('='*70)
        print("\n" + df.to_string(index=False))
        
        # Best model
        best_model = max(results.items(), key=lambda x: x[1]['r2'])
        print(f"\n🏆 BEST MODEL: {best_model[1]['name']}")
        print(f"   R² = {best_model[1]['r2']:.4f}")
        print(f"   RMSE = {best_model[1]['rmse']:.4f}")
    
    # Final summary
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print(f"\nOutputs saved in: {output_dir}")
    print("\nGenerated files:")
    for target in targets:
        print(f"  ✓ {target}_model_comparison.png")
        print(f"  ✓ {target}_predictions_all_models.png")
        print(f"  ✓ {target}/model_comparison_table.csv")
    
    print("\nRecommendations:")
    for target in targets:
        results = all_results[target]
        best_model = max(results.items(), key=lambda x: x[1]['r2'])
        print(f"\n{target.capitalize()}:")
        print(f"  → Use {best_model[1]['name']} (R² = {best_model[1]['r2']:.4f})")
        
        # Alternative suggestions
        sorted_models = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
        if len(sorted_models) > 1:
            print(f"  Alternative: {sorted_models[1][1]['name']} (R² = {sorted_models[1][1]['r2']:.4f})")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
