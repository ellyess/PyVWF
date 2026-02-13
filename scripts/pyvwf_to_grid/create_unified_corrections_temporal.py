#!/usr/bin/env python3
"""Create unified correction dataframe from TEMPORAL training (2015-2019 → 2023).

This script loads correction factors from:
- Country-level temporal training (2015-2019 → 2023)
- Turbine-level temporal training (2015-2019 → 2023)

And combines them into a single unified dataframe for spatial interpolation.
"""

import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

# ============================================================================
# Configuration for TEMPORAL training results
# ============================================================================

# Country-level corrections (temporal 2015-2021 → 2023)
COUNTRY_LEVEL_TEMPORAL = {
    "NL": {
        "factors_path": "output/temporal_2015_2021_to_2023/output/run/NL-all-obs_country-corrected-calc_z0/training/correction-factors/NL_factors_fixed_5.csv",
        "geoms_path": "input/country_level_data/grid_points/nl/nl_correction_regions.geojson",
        "name": "Netherlands",
        "n_clusters": 5,
    },
    "FR": {
        "factors_path": "output/temporal_2015_2021_to_2023/output/run/FR-all-obs_country-corrected-calc_z0/training/correction-factors/FR_factors_fixed_10.csv",
        "geoms_path": "input/country_level_data/grid_points/fr/fr_correction_regions.geojson",
        "name": "France",
        "n_clusters": 10,
    },
    "BE": {
        "factors_path": "output/temporal_2015_2021_to_2023/output/run/BE-all-obs_country-corrected-calc_z0/training/correction-factors/BE_factors_fixed_3.csv",
        "geoms_path": "input/country_level_data/grid_points/be/be_correction_regions.geojson",
        "name": "Belgium",
        "n_clusters": 3,
    },
    "NO": {
        "factors_path": "output/temporal_2015_2021_to_2023/output/run/NO-all-obs_country-corrected-calc_z0/training/correction-factors/NO_factors_fixed_5.csv",
        "geoms_path": "input/country_level_data/grid_points/no/no_bidding_zones.geojson",
        "name": "Norway",
        "n_clusters": 5,
    },
    "ES": {
        "factors_path": "output/temporal_2015_2021_to_2023/output/run/ES-all-obs_country-corrected-calc_z0/training/correction-factors/ES_factors_fixed_4.csv",
        "geoms_path": "input/country_level_data/grid_points/es/es_correction_regions.geojson",
        "name": "Spain",
        "n_clusters": 4,
    },
    "SE": {
        "factors_path": "output/temporal_2015_2021_to_2023/output/run/SE-all-obs_country-corrected-calc_z0/training/correction-factors/SE_factors_fixed_4.csv",
        "geoms_path": "input/country_level_data/grid_points/se/se_bidding_zones.geojson",
        "name": "Sweden",
        "n_clusters": 4,
    },
    "IT": {
        "factors_path": "output/temporal_2015_2021_to_2023/output/run/IT-all-obs_country-corrected-calc_z0/training/correction-factors/IT_factors_fixed_3.csv",
        "geoms_path": "input/country_level_data/grid_points/it/it_correction_regions.geojson",
        "name": "Italy",
        "n_clusters": 3,
    },
    "PT": {
        "factors_path": "output/temporal_2015_2021_to_2023/output/run/PT-all-obs_country-corrected-calc_z0/training/correction-factors/PT_factors_fixed_3.csv",
        "geoms_path": "input/country_level_data/grid_points/pt/pt_correction_regions.geojson",
        "name": "Portugal",
        "n_clusters": 3,
    },
    "IE": {
        "factors_path": "output/temporal_2015_2021_to_2023/output/run/IE-all-obs_country-corrected-calc_z0/training/correction-factors/IE_factors_fixed_3.csv",
        "geoms_path": "input/country_level_data/grid_points/ie/ie_correction_regions.geojson",
        "name": "Ireland",
        "n_clusters": 3,
    },
}

# Turbine-level corrections (temporal 2015-2019 → 2023)
TURBINE_LEVEL_TEMPORAL = {
    "DE-onshore": {
        "factors_path": "output/turbine_temporal_2015_2019_to_2023/output/run/DE-onshore-obs_turbine-corrected-calc_z0/training/correction-factors/DE_factors_fixed_500.csv",
        "geoms_path": "output/cluster_geometries/de/de_onshore_correction_regions_500.geojson",
        "name": "Germany",
        "cluster_mode": "onshore",
        "n_clusters": 500,
    },
    "DK-onshore": {
        "factors_path": "output/turbine_temporal_2015_2019_to_2023/output/run/DK-onshore-obs_turbine-corrected-calc_z0/training/correction-factors/DK_factors_fixed_1000.csv",
        "geoms_path": "output/cluster_geometries/dk/dk_onshore_correction_regions_1000.geojson",
        "name": "Denmark",
        "cluster_mode": "onshore",
        "n_clusters": 1000,
    },
    "DK-offshore": {
        "factors_path": "output/turbine_temporal_2015_2019_to_2023/output/run/DK-offshore-obs_turbine-corrected-calc_z0/training-factors/DK_factors_fixed_2.csv",
        "geoms_path": "output/cluster_geometries/dk/dk_offshore_correction_regions_2.geojson",
        "name": "Denmark",
        "cluster_mode": "offshore",
        "n_clusters": 2,
    },
    "UK-onshore": {
        "factors_path": "output/turbine_temporal_2015_2019_to_2023/output/run/UK-onshore-obs_turbine-corrected-calc_z0/training/correction-factors/UK_factors_fixed_300.csv",
        "geoms_path": "output/cluster_geometries/uk/uk_onshore_correction_regions_300.geojson",
        "name": "United Kingdom",
        "cluster_mode": "onshore",
        "n_clusters": 300,
    },
    "UK-offshore": {
        "factors_path": "output/turbine_temporal_2015_2019_to_2023/output/run/UK-offshore-obs_turbine-corrected-calc_z0/training/correction-factors/UK_factors_fixed_10.csv",
        "geoms_path": "output/cluster_geometries/uk/uk_offshore_correction_regions_10.geojson",
        "name": "United Kingdom",
        "cluster_mode": "offshore",
        "n_clusters": 10,
    },
}


def create_unified_corrections_temporal():
    """Create unified corrections CSV from temporal training.

    Uses:
    - Country-level: 2015-2021 → 2023 (9 countries, 40 clusters)
    - Turbine-level: 2015-2019 → 2023 (5 configs, 1687 clusters)

    Note: Turbine-level uses 2015-2019 because THEOPE database only has data through 2019.
    """
    print("="*80)
    print("Creating Unified Corrections from TEMPORAL Training")
    print("Country-level: 2015-2021 → 2023")
    print("Turbine-level: 2015-2019 → 2023")
    print("="*80)

    all_centroids = []

    # Process country-level configs
    print("\nProcessing country-level corrections...")
    for country_code, config in COUNTRY_LEVEL_TEMPORAL.items():
        print(f"  - {country_code} ({config['name']})")

        factors_path = Path(config['factors_path'])
        geoms_path = Path(config['geoms_path'])

        if not factors_path.exists():
            print(f"    ⚠️  Skipping: {factors_path} not found")
            continue

        if not geoms_path.exists():
            print(f"    ⚠️  Skipping: {geoms_path} not found")
            continue

        # Load correction factors
        factors = pd.read_csv(factors_path)

        # Load geometries
        geometries = gpd.read_file(geoms_path)

        # Merge
        gdf = geometries.merge(factors, on='cluster', how='left')

        # Compute centroids
        gdf['centroid'] = gdf.geometry.centroid
        gdf['lon'] = gdf['centroid'].x
        gdf['lat'] = gdf['centroid'].y
        gdf['area_km2'] = gdf.geometry.area / 1e6  # Convert m² to km²

        # Create dataframe
        df = pd.DataFrame({
            'country_code': country_code,
            'country_name': config['name'],
            'cluster': gdf['cluster'],
            'cluster_mode': 'all',
            'obs_level': 'country',
            'lon': gdf['lon'],
            'lat': gdf['lat'],
            'scalar': gdf['scalar'],
            'offset': gdf['offset'],
            'area_km2': gdf['area_km2'],
        })

        all_centroids.append(df)
        print(f"    ✓ Added {len(df)} clusters")

    # Process turbine-level configs
    print("\nProcessing turbine-level corrections...")
    for config_name, config in TURBINE_LEVEL_TEMPORAL.items():
        print(f"  - {config_name} ({config['name']} {config['cluster_mode']})")

        factors_path = Path(config['factors_path'])
        geoms_path = Path(config['geoms_path'])

        if not factors_path.exists():
            print(f"    ⚠️  Skipping: {factors_path} not found")
            continue

        if not geoms_path.exists():
            print(f"    ⚠️  Skipping: {geoms_path} not found")
            continue

        # Load correction factors
        factors = pd.read_csv(factors_path)

        # Load geometries
        geometries = gpd.read_file(geoms_path)

        # Merge
        gdf = geometries.merge(factors, on='cluster', how='left')

        # Compute centroids
        gdf['centroid'] = gdf.geometry.centroid
        gdf['lon'] = gdf['centroid'].x
        gdf['lat'] = gdf['centroid'].y
        gdf['area_km2'] = gdf.geometry.area / 1e6  # Convert m² to km²

        # Create dataframe
        df = pd.DataFrame({
            'country_code': config_name,
            'country_name': config['name'],
            'cluster': gdf['cluster'],
            'cluster_mode': config['cluster_mode'],
            'obs_level': 'turbine',
            'lon': gdf['lon'],
            'lat': gdf['lat'],
            'scalar': gdf['scalar'],
            'offset': gdf['offset'],
            'area_km2': gdf['area_km2'],
        })

        all_centroids.append(df)
        print(f"    ✓ Added {len(df)} clusters")

    # Combine all
    print("\nCombining all corrections...")
    unified_df = pd.concat(all_centroids, ignore_index=True)

    # Filter out rows with NaN lat/lon (missing geometries)
    before_filter = len(unified_df)
    unified_df = unified_df.dropna(subset=['lon', 'lat'])
    after_filter = len(unified_df)
    if before_filter > after_filter:
        print(f"  ⚠️  Filtered out {before_filter - after_filter} clusters with missing coordinates")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total clusters: {len(unified_df)}")
    print(f"\nBy observation level:")
    print(unified_df.groupby('obs_level').size())
    print(f"\nBy cluster mode:")
    print(unified_df.groupby('cluster_mode').size())
    print(f"\nBy country:")
    print(unified_df.groupby('country_code').size())

    # Save
    output_dir = Path("output/unified_corrections_temporal_2015_2021")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "all_corrections_centroids_temporal_2015_2021.csv"

    unified_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")
    print("="*80)

    return unified_df


if __name__ == "__main__":
    df = create_unified_corrections_temporal()
    print(f"\nUnified corrections dataframe shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
