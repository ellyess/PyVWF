#!/usr/bin/env python3
"""Compare different feature sets for within-Denmark prediction.

This script evaluates multiple feature subsets using ridge regression and
reports test and cross-validation metrics.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load cached DK features
df = pd.read_csv('ml_europe/cache/features_train_DK.csv')

# Test different feature sets
feature_sets = {
    'Terrain only': ['elevation', 'slope', 'aspect', 'roughness', 'curvature', 'distance_to_coast_km'],
    'Terrain + ERA5': ['elevation', 'slope', 'aspect', 'roughness', 'curvature', 'distance_to_coast_km', 
                       'era5_elevation_error', 'era5_roughness', 'subgrid_terrain_variance'],
    'Terrain + Geographic': ['elevation', 'slope', 'aspect', 'roughness', 'curvature', 'distance_to_coast_km', 
                             'latitude', 'longitude'],
    'Terrain + Spatial': ['elevation', 'slope', 'aspect', 'roughness', 'curvature', 'distance_to_coast_km', 
                          'nearby_mean_correction', 'nearby_correction_variance', 'turbine_density'],
    'All features': [c for c in df.columns if c not in ['lon', 'lat', 'cluster', 'scalar', 'offset', 'type', 'country']]
}

print('='*80)
print('Denmark-Only Performance by Feature Set (Ridge Regression)')
print('='*80)
print(f'Total samples: {len(df)}')
print(f'Scalar range: [{df["scalar"].min():.3f}, {df["scalar"].max():.3f}]')
print(f'Train/test split: 80/20, random_state=42')
print()

results = []

for name, features in feature_sets.items():
    # Prepare data
    X = df[features].values
    y = df['scalar'].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cross-val
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    print(f'{name}:')
    print(f'  Features: {len(features)}')
    print(f'  Test R²: {r2:.4f}')
    print(f'  RMSE: {rmse:.4f}')
    print(f'  MAE: {mae:.4f}')
    print(f'  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})')
    print()
    
    results.append({
        'Feature Set': name,
        'Num Features': len(features),
        'Test R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'CV R² Mean': cv_scores.mean(),
        'CV R² Std': cv_scores.std()
    })

# Summary table
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test R²', ascending=False)

print('='*80)
print('Summary (sorted by Test R²)')
print('='*80)
print(results_df.to_string(index=False))
print()

# Save
results_df.to_csv('ml_europe/denmark_feature_comparison.csv', index=False)
print('Results saved to ml_europe/denmark_feature_comparison.csv')
