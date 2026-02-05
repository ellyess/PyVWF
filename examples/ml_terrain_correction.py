"""
Example: Machine Learning-based Bias Correction with Terrain Features

This example demonstrates how to use ML models to learn the relationship
between terrain features and bias correction factors.

Use cases:
1. Understanding which terrain features drive model bias
2. Transferring corrections to regions without observations
3. Improving spatial interpolation with physical predictors
"""
from pathlib import Path
import pandas as pd
import numpy as np
from vwf.ml_correction import (
    create_feature_matrix,
    train_correction_model,
    compare_interpolation_methods,
    export_ml_correction_grid,
)


def example_basic_ml_training():
    """
    Basic example: Train an ML model to predict correction factors
    from terrain features.
    """
    print("="*60)
    print("Example 1: Basic ML Training")
    print("="*60)
    
    # Load your correction points (from bias correction workflow)
    corrections = pd.read_csv('out/correction_points.csv')
    # Expected columns: lon, lat, scalar, offset, [optional: elevation, roughness, etc.]
    
    # Option 1: Use existing terrain features in the CSV
    feature_cols = ['elevation', 'slope', 'roughness', 'distance_to_coast_km']
    
    # Train model for scalar correction
    scalar_result = train_correction_model(
        corrections,
        target_col='scalar',
        feature_cols=feature_cols,
        model_type='random_forest',
        scale_features=True,
        cv_folds=5,
        # Random forest hyperparameters
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
    )
    
    # Train model for offset correction
    offset_result = train_correction_model(
        corrections,
        target_col='offset',
        feature_cols=feature_cols,
        model_type='random_forest',
        scale_features=True,
        cv_folds=5,
    )
    
    # Print feature importance
    print("\nFeature Importance for SCALAR:")
    print(scalar_result['feature_importance'])
    
    print("\nFeature Importance for OFFSET:")
    print(offset_result['feature_importance'])
    
    return scalar_result, offset_result


def example_with_terrain_extraction():
    """
    Example: Extract terrain features from NetCDF, then train model.
    """
    print("="*60)
    print("Example 2: With Terrain Feature Extraction")
    print("="*60)
    
    # Load correction points
    corrections = pd.read_csv('out/correction_points.csv')
    
    # Create feature matrix by extracting from terrain NetCDF
    features = create_feature_matrix(
        corrections,
        terrain_nc='input/terrain/europe_terrain.nc',  # Elevation, slope, etc.
        coastline_geojson='input/regions/coastline.geojson',
        lon_col='lon',
        lat_col='lat',
    )
    
    # Now features has terrain data merged in
    print(f"Features available: {features.columns.tolist()}")
    
    # Auto-detect feature columns (excludes target and ID columns)
    scalar_result = train_correction_model(
        features,
        target_col='scalar',
        feature_cols=None,  # Auto-detect
        model_type='gradient_boosting',
    )
    
    return scalar_result


def example_model_comparison():
    """
    Example: Compare different ML models.
    """
    print("="*60)
    print("Example 3: Model Comparison")
    print("="*60)
    
    corrections = pd.read_csv('out/correction_points.csv')
    
    # Create features
    features = create_feature_matrix(
        corrections,
        terrain_nc='input/terrain/europe_terrain.nc',
        coastline_geojson='input/regions/coastline.geojson',
    )
    
    # Define feature columns
    feature_cols = [
        'elevation', 'slope', 'aspect', 'roughness', 'curvature',
        'distance_to_coast_km', 'lat_norm', 'lon_norm'
    ]
    
    # Compare models
    comparison = compare_interpolation_methods(
        features,
        feature_cols=feature_cols,
        target_col='scalar',
        models=[
            'random_forest',
            'gradient_boosting',
            'xgboost',      # Requires: pip install xgboost
            'lightgbm',     # Requires: pip install lightgbm
            'ridge',
            'elastic_net',
        ],
        cv_folds=5,
    )
    
    # Save comparison
    comparison.to_csv('out/model_comparison.csv', index=False)
    
    return comparison


def example_separate_onshore_offshore():
    """
    Example: Train separate models for onshore and offshore.
    """
    print("="*60)
    print("Example 4: Separate Onshore/Offshore Models")
    print("="*60)
    
    from vwf import add_domain_column, filter_by_domain
    
    # Load corrections
    corrections = pd.read_csv('out/correction_points.csv')
    
    # Add domain classification if not present
    if 'domain' not in corrections.columns:
        corrections = add_domain_column(
            corrections,
            onshore_geojson='input/regions/country_shapes.geojson',
            offshore_geojson='input/regions/north_sea_shape.geojson',
        )
    
    # Create features
    features = create_feature_matrix(
        corrections,
        terrain_nc='input/terrain/europe_terrain.nc',
        coastline_geojson='input/regions/coastline.geojson',
    )
    
    # Split by domain
    features_onshore = filter_by_domain(features, 'onshore')
    features_offshore = filter_by_domain(features, 'offshore')
    
    print(f"Onshore points: {len(features_onshore)}")
    print(f"Offshore points: {len(features_offshore)}")
    
    # Train separate models
    feature_cols = ['elevation', 'slope', 'roughness', 'distance_to_coast_km']
    
    onshore_model = train_correction_model(
        features_onshore,
        target_col='scalar',
        feature_cols=feature_cols,
        model_type='random_forest',
    )
    
    offshore_model = train_correction_model(
        features_offshore,
        target_col='scalar',
        feature_cols=feature_cols,
        model_type='random_forest',
    )
    
    return onshore_model, offshore_model


def example_full_pipeline():
    """
    Example: Full end-to-end pipeline - train and export gridded corrections.
    """
    print("="*60)
    print("Example 5: Full Pipeline with Grid Export")
    print("="*60)
    
    # This runs the complete workflow:
    # 1. Load correction points
    # 2. Extract terrain features
    # 3. Train ML models
    # 4. Apply to grid
    # 5. Export NetCDF
    
    out_path = export_ml_correction_grid(
        corrections_csv='out/correction_points.csv',
        grid_nc='cutouts/europe-2023-sarah3-era5.nc',
        out_nc='out/ml_bias_correction_grid.nc',
        terrain_nc='input/terrain/europe_terrain.nc',
        coastline_geojson='input/regions/coastline.geojson',
        onshore_mask_geojson='input/regions/country_shapes.geojson',
        offshore_mask_geojson='input/regions/north_sea_shape.geojson',
        model_type='random_forest',
        # Model hyperparameters
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
    )
    
    print(f"\nExported grid to: {out_path}")
    
    return out_path


def example_transfer_learning():
    """
    Example: Train on one region, predict on another (transfer learning).
    """
    print("="*60)
    print("Example 6: Transfer Learning Across Regions")
    print("="*60)
    
    # Load training data from Region A (e.g., UK)
    train_corrections = pd.read_csv('out/uk_correction_points.csv')
    
    # Create training features
    train_features = create_feature_matrix(
        train_corrections,
        terrain_nc='input/terrain/europe_terrain.nc',
        coastline_geojson='input/regions/coastline.geojson',
    )
    
    # Train model on Region A
    model_result = train_correction_model(
        train_features,
        target_col='scalar',
        model_type='gradient_boosting',
        cv_folds=5,
    )
    
    print("\nModel trained on UK data")
    print(f"CV R²: {model_result['cv_scores']['test_r2'].mean():.3f}")
    
    # Now apply to Region B (e.g., Germany) which has no observations
    from vwf.ml_correction import predict_correction_grid
    
    de_predictions = predict_correction_grid(
        model_result,
        grid_nc='cutouts/germany-2023-era5.nc',
        terrain_nc='input/terrain/europe_terrain.nc',
        coastline_geojson='input/regions/coastline.geojson',
    )
    
    # Save predictions
    de_predictions.to_netcdf('out/germany_predicted_corrections.nc')
    
    print("\nPredictions exported for Germany")
    
    return model_result, de_predictions


def example_feature_engineering():
    """
    Example: Advanced feature engineering for better performance.
    """
    print("="*60)
    print("Example 7: Advanced Feature Engineering")
    print("="*60)
    
    import pandas as pd
    import numpy as np
    
    corrections = pd.read_csv('out/correction_points.csv')
    
    # Basic features
    features = create_feature_matrix(
        corrections,
        terrain_nc='input/terrain/europe_terrain.nc',
        coastline_geojson='input/regions/coastline.geojson',
    )
    
    # Add custom engineered features
    
    # 1. Interaction terms
    features['slope_x_elevation'] = features['slope'] * features['elevation']
    features['rough_x_distance'] = features['roughness'] * features['distance_to_coast_km']
    
    # 2. Polynomial features
    features['elevation_squared'] = features['elevation'] ** 2
    features['distance_squared'] = features['distance_to_coast_km'] ** 2
    
    # 3. Binned features (capturing non-linear effects)
    features['elevation_bin'] = pd.cut(
        features['elevation'],
        bins=[-np.inf, 0, 100, 500, 1000, np.inf],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # 4. Trigonometric (for aspect/orientation)
    if 'aspect' in features.columns:
        features['aspect_north'] = np.cos(np.radians(features['aspect']))
        features['aspect_east'] = np.sin(np.radians(features['aspect']))
    
    # 5. Distance categories
    features['is_coastal'] = (features['distance_to_coast_km'] < 50).astype(int)
    features['is_inland'] = (features['distance_to_coast_km'] > 100).astype(int)
    
    # Train with engineered features
    feature_cols = [c for c in features.columns 
                   if c not in ['scalar', 'offset', 'lon', 'lat', 'ID', 'domain']]
    
    model_result = train_correction_model(
        features,
        target_col='scalar',
        feature_cols=feature_cols,
        model_type='xgboost',  # XGBoost handles interactions well
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
    )
    
    print(f"\nTrained with {len(feature_cols)} features (including engineered)")
    
    return model_result


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PyVWF - ML-Based Bias Correction Examples")
    print("="*60 + "\n")
    
    # Run examples (uncomment the ones you want to try)
    
    # Example 1: Basic training
    # example_basic_ml_training()
    
    # Example 2: Extract terrain features
    # example_with_terrain_extraction()
    
    # Example 3: Compare models
    # example_model_comparison()
    
    # Example 4: Separate onshore/offshore
    # example_separate_onshore_offshore()
    
    # Example 5: Full pipeline
    # example_full_pipeline()
    
    # Example 6: Transfer learning
    # example_transfer_learning()
    
    # Example 7: Feature engineering
    # example_feature_engineering()
    
    print("\nTo run examples, uncomment the desired example in __main__")
    print("Make sure you have the required input files and dependencies installed:")
    print("  - pip install scikit-learn xgboost lightgbm")
