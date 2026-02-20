#!/usr/bin/env python3
"""Test Kriging configuration improvements via 5-fold spatial CV.

Compares multiple Kriging configurations against the IDW baseline:
  1. Baseline: OK with spherical variogram, euclidean coordinates (current default)
  2. Geographic coordinates: OK with great-circle distances (pykrige 1.7+)
  3. Projected km coordinates: simple cos-lat scaling
  4. Universal Kriging: regional linear drift term (euclidean only)
  5. Universal Kriging with projected km coordinates

Key findings (Feb 2026, 1,474 control points):
  - Geographic coordinates help most (3-6% improvement over euclidean)
  - Exponential variogram best for scalar, linear best for offset
  - Universal Kriging actually hurts performance
  - Best Kriging (OK, exponential, geographic) still can't beat IDW:
      IDW:     scalar MAE=0.1680 | offset MAE=0.6653
      Kriging: scalar MAE=0.1741 | offset MAE=0.6342

Uses vectorized predictions via krig.execute('points', array, array)
for ~10x speedup over point-by-point loop.

Usage:
    python scripts/pyvwf_to_grid/test_kriging_improvements.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
import warnings

warnings.filterwarnings('ignore')

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CSV = PROJECT_ROOT / 'output' / 'pyvwf_to_grid' / 'all_corrections_centroids.csv'


def spatial_cv_split(df, n_splits=5):
    """Create spatial CV splits sorted by longitude."""
    sorted_idx = df['lon'].argsort().values
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


def run_cv(df, splits, variogram_model, x_col, y_col, use_uk=False,
           coords_type='euclidean', label=''):
    """Run 5-fold spatial CV for a Kriging configuration."""
    scalar_maes, offset_maes = [], []
    t0 = time.time()
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        for target in ['scalar', 'offset']:
            try:
                kwargs = dict(
                    variogram_model=variogram_model,
                    verbose=False, enable_plotting=False,
                )
                # Only OrdinaryKriging supports coordinates_type
                if not use_uk and coords_type == 'geographic':
                    kwargs['coordinates_type'] = 'geographic'

                if use_uk:
                    krig = UniversalKriging(
                        train[x_col].values, train[y_col].values,
                        train[target].values,
                        drift_terms=['regional_linear'],
                        **kwargs,
                    )
                else:
                    krig = OrdinaryKriging(
                        train[x_col].values, train[y_col].values,
                        train[target].values,
                        **kwargs,
                    )
                # Vectorized prediction
                pred, _ = krig.execute(
                    'points',
                    test[x_col].values,
                    test[y_col].values,
                )
                pred = np.asarray(pred).ravel()
                true = test[target].values
                mae = np.abs(pred - true).mean()
                if target == 'scalar':
                    scalar_maes.append(mae)
                else:
                    offset_maes.append(mae)
            except Exception as e:
                print(f"  ERROR fold {fold_i+1} {target}: {e}")
                (scalar_maes if target == 'scalar' else offset_maes).append(np.nan)
    elapsed = time.time() - t0
    s = np.nanmean(scalar_maes)
    o = np.nanmean(offset_maes)
    print(f"  {label:55s}  scalar MAE={s:.4f}  |  offset MAE={o:.4f}  ({elapsed:.0f}s)")
    return s, o


def main():
    print("Loading data...")
    df = pd.read_csv(CSV)
    df = df.dropna(subset=['lon', 'lat', 'scalar', 'offset'])
    print(f"Control points: {len(df)}")

    # Add projected coordinates (simple cos-lat scaling for longitude)
    mean_lat = np.radians(df['lat'].mean())
    df['x_proj'] = df['lon'] * np.cos(mean_lat) * 111.32  # km
    df['y_proj'] = df['lat'] * 110.57  # km

    splits = spatial_cv_split(df, 5)

    print("=" * 120)
    print("KRIGING IMPROVEMENT TESTS (5-fold spatial CV, vectorized)")
    print("=" * 120)

    results = {}

    # 1. Baseline
    print("\n--- 1. BASELINE (current settings) ---")
    results['OK_spherical_euclidean'] = run_cv(
        df, splits, 'spherical', 'lon', 'lat',
        label='OK, spherical, euclidean')

    # 2. Geographic coordinates (great-circle distance)
    print("\n--- 2. GEOGRAPHIC COORDINATES (OK only) ---")
    for model in ['spherical', 'exponential', 'hole-effect', 'linear']:
        results[f'OK_{model}_geographic'] = run_cv(
            df, splits, model, 'lon', 'lat', coords_type='geographic',
            label=f'OK, {model}, geographic')

    # 3. Projected km coordinates
    print("\n--- 3. PROJECTED COORDINATES (km) ---")
    for model in ['spherical', 'exponential']:
        results[f'OK_{model}_projected'] = run_cv(
            df, splits, model, 'x_proj', 'y_proj',
            label=f'OK, {model}, projected_km')

    # 4. Universal Kriging euclidean (UK doesn't support geographic)
    print("\n--- 4. UNIVERSAL KRIGING (euclidean) ---")
    for model in ['spherical', 'exponential', 'linear']:
        results[f'UK_{model}_euclidean'] = run_cv(
            df, splits, model, 'lon', 'lat', use_uk=True,
            label=f'UK, {model}, euclidean, regional_linear')

    # 5. Universal Kriging projected
    print("\n--- 5. UNIVERSAL KRIGING (projected km) ---")
    for model in ['spherical', 'exponential']:
        results[f'UK_{model}_projected'] = run_cv(
            df, splits, model, 'x_proj', 'y_proj', use_uk=True,
            label=f'UK, {model}, projected_km, regional_linear')

    # Save results
    output_dir = PROJECT_ROOT / 'output' / 'pyvwf_to_grid' / 'grid_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([
        {'config': k, 'scalar_mae': v[0], 'offset_mae': v[1]}
        for k, v in results.items()
    ])
    csv_path = output_dir / 'kriging_improvement_cv_scores.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results: {csv_path}")

    print("\n" + "=" * 120)
    print("COMPARISON BASELINES (from grid comparison run):")
    print("  IDW:     scalar MAE=0.1680  |  offset MAE=0.6653")
    print("  Kriging: scalar MAE=0.1851  |  offset MAE=0.6894")
    print("  RBF:     scalar MAE=0.2599  |  offset MAE=0.9631")
    print("=" * 120)


if __name__ == '__main__':
    main()
