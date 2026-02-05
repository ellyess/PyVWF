"""
Quick demonstration of ML-based bias correction with terrain features.

This script shows the most common workflow:
1. Load correction points from bias correction
2. Extract/merge terrain features
3. Train ML model
4. Export gridded corrections
"""
from pathlib import Path
import pandas as pd
import numpy as np


def create_synthetic_example():
    """Create synthetic data for demonstration (when real data not available)."""
    np.random.seed(42)
    n_points = 1000
    
    # Create synthetic correction points
    lon = np.random.uniform(-10, 20, n_points)
    lat = np.random.uniform(45, 60, n_points)
    
    # Synthetic terrain features
    elevation = np.random.uniform(0, 2000, n_points)
    slope = np.random.uniform(0, 45, n_points)
    roughness = np.random.uniform(0.001, 2.0, n_points)
    distance_to_coast = np.random.uniform(0, 500, n_points)
    
    # Synthetic correction factors (scalar and offset)
    # Make them depend on terrain features for realism
    scalar = 1.0 + 0.3 * (elevation / 1000) + 0.1 * (slope / 45) + np.random.normal(0, 0.1, n_points)
    offset = 0.05 * (distance_to_coast / 500) - 0.1 * (roughness / 2.0) + np.random.normal(0, 0.05, n_points)
    
    # Clip to reasonable ranges
    scalar = np.clip(scalar, 0.5, 1.5)
    offset = np.clip(offset, -0.3, 0.3)
    
    df = pd.DataFrame({
        'lon': lon,
        'lat': lat,
        'elevation': elevation,
        'slope': slope,
        'roughness': roughness,
        'distance_to_coast_km': distance_to_coast,
        'scalar': scalar,
        'offset': offset,
    })
    
    return df


def main():
    """Main demonstration."""
    print("="*70)
    print("PyVWF - ML-Based Bias Correction Quickstart")
    print("="*70)
    print()
    
    # Option 1: Use your real data
    # corrections_file = Path('out/correction_points.csv')
    # if corrections_file.exists():
    #     print(f"Loading real data from {corrections_file}")
    #     corrections = pd.read_csv(corrections_file)
    # else:
    
    # Option 2: Use synthetic data for demo
    print("Creating synthetic data for demonstration...")
    corrections = create_synthetic_example()
    print(f"Created {len(corrections)} synthetic correction points")
    print()
    
    # Display sample
    print("Sample of correction data:")
    print(corrections.head())
    print()
    
    # Check available features
    feature_cols = [c for c in corrections.columns 
                   if c not in ['lon', 'lat', 'scalar', 'offset']]
    print(f"Available features: {feature_cols}")
    print()
    
    # =================================================================
    # STEP 1: Train ML model for scalar correction
    # =================================================================
    print("-" * 70)
    print("STEP 1: Training ML model for SCALAR correction")
    print("-" * 70)
    
    from vwf.ml_correction import train_correction_model
    
    scalar_model = train_correction_model(
        corrections,
        target_col='scalar',
        feature_cols=feature_cols,
        model_type='random_forest',
        scale_features=True,
        cv_folds=5,
        # Hyperparameters
        n_estimators=100,
        max_depth=10,
    )
    
    print()
    if scalar_model['feature_importance'] is not None:
        print("Top 5 features for SCALAR:")
        print(scalar_model['feature_importance'].head())
    print()
    
    # =================================================================
    # STEP 2: Train ML model for offset correction
    # =================================================================
    print("-" * 70)
    print("STEP 2: Training ML model for OFFSET correction")
    print("-" * 70)
    
    offset_model = train_correction_model(
        corrections,
        target_col='offset',
        feature_cols=feature_cols,
        model_type='random_forest',
        scale_features=True,
        cv_folds=5,
        n_estimators=100,
        max_depth=10,
    )
    
    print()
    if offset_model['feature_importance'] is not None:
        print("Top 5 features for OFFSET:")
        print(offset_model['feature_importance'].head())
    print()
    
    # =================================================================
    # STEP 3: Compare different ML models (optional)
    # =================================================================
    print("-" * 70)
    print("STEP 3: Comparing different ML models")
    print("-" * 70)
    
    from vwf.ml_correction import compare_interpolation_methods
    
    try:
        comparison = compare_interpolation_methods(
            corrections,
            feature_cols=feature_cols,
            target_col='scalar',
            models=['random_forest', 'gradient_boosting', 'ridge'],
            cv_folds=3,  # Fewer folds for speed
        )
        
        print("\nBest model:", comparison.iloc[0]['model'])
        print(f"R² score: {comparison.iloc[0]['r2_mean']:.3f}")
    except Exception as e:
        print(f"Model comparison failed (optional step): {e}")
    
    print()
    
    # =================================================================
    # STEP 4: Make predictions on test set
    # =================================================================
    print("-" * 70)
    print("STEP 4: Testing predictions on holdout set")
    print("-" * 70)
    
    # Split data for testing
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(corrections, test_size=0.2, random_state=42)
    
    # Retrain on training set only
    train_model = train_correction_model(
        train_df,
        target_col='scalar',
        feature_cols=feature_cols,
        model_type='random_forest',
        cv_folds=3,
        n_estimators=100,
    )
    
    # Predict on test set
    X_test = test_df[feature_cols].fillna(test_df[feature_cols].median())
    y_test = test_df['scalar'].values
    y_pred = train_model['model'].predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Test set performance:")
    print(f"  R² score: {r2:.3f}")
    print(f"  MAE: {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print()
    
    # =================================================================
    # SUMMARY
    # =================================================================
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Trained ML models for scalar and offset corrections")
    print(f"✓ Used {len(feature_cols)} terrain features")
    print(f"✓ Achieved R² = {scalar_model['cv_scores']['test_r2'].mean():.3f} (scalar)")
    print(f"✓ Achieved R² = {offset_model['cv_scores']['test_r2'].mean():.3f} (offset)")
    print()
    print("Next steps:")
    print("  1. Apply model to spatial grid using predict_correction_grid()")
    print("  2. Export as NetCDF using export_ml_correction_grid()")
    print("  3. Use in atlite workflow for bias-corrected wind simulations")
    print()
    print("See examples/ml_terrain_correction.py for more advanced usage")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nTip: Make sure scikit-learn is installed:")
        print("  pip install scikit-learn")
