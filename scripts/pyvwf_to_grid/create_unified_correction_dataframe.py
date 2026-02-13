"""Create unified correction geometry dataframe for spatial interpolation.

This script:
1. Loads all correction factor GeoDataFrames (country-level and turbine-level)
2. Combines them into a single dataframe with cluster centroids
3. Adds country metadata for tracking
4. Exports as both GeoJSON (for visualization) and CSV (for interpolation)

Output can be used for:
- Kriging/IDW interpolation to arbitrary locations
- Training machine learning models for correction prediction
- Spatial analysis of correction patterns across Europe
- Creating continent-wide correction maps
"""

import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

# Country-level corrections
COUNTRY_LEVEL_CONFIGS = {
    "NL": {
        "path": "output/correction_geodataframes/nl/nl_corrections_fixed_5.geojson",
        "name": "Netherlands",
        "obs_level": "country",
        "n_clusters": 5,
    },
    "FR": {
        "path": "output/correction_geodataframes/fr/fr_corrections_fixed_10.geojson",
        "name": "France",
        "obs_level": "country",
        "n_clusters": 10,
    },
    "BE": {
        "path": "output/correction_geodataframes/be/be_corrections_fixed_3.geojson",
        "name": "Belgium",
        "obs_level": "country",
        "n_clusters": 3,
    },
    "NO": {
        "factors_path": "output/run/NO-all-obs_country-corrected-calc_z0/training/correction-factors/NO_factors_fixed_5.csv",
        "geoms_path": "input/country_level_data/grid_points/no/no_bidding_zones.geojson",
        "name": "Norway",
        "obs_level": "country",
        "n_clusters": 5,
    },
    # Phase 1 Countries (multi-cluster)
    "ES": {
        "factors_path": "output/run/ES-all-obs_country-corrected-calc_z0/training/correction-factors/ES_factors_fixed_4.csv",
        "geoms_path": "input/country_level_data/grid_points/es/es_correction_regions.geojson",
        "name": "Spain",
        "obs_level": "country",
        "n_clusters": 4,
    },
    "SE": {
        "factors_path": "output/run/SE-all-obs_country-corrected-calc_z0/training/correction-factors/SE_factors_fixed_4.csv",
        "geoms_path": "input/country_level_data/grid_points/se/se_bidding_zones.geojson",
        "name": "Sweden",
        "obs_level": "country",
        "n_clusters": 4,
        "use_bidding_zones": True,
    },
    "IT": {
        "factors_path": "output/run/IT-all-obs_country-corrected-calc_z0/training/correction-factors/IT_factors_fixed_3.csv",
        "geoms_path": "input/country_level_data/grid_points/it/it_correction_regions.geojson",
        "name": "Italy",
        "obs_level": "country",
        "n_clusters": 3,
    },
    "PT": {
        "factors_path": "output/run/PT-all-obs_country-corrected-calc_z0/training/correction-factors/PT_factors_fixed_3.csv",
        "geoms_path": "input/country_level_data/grid_points/pt/pt_correction_regions.geojson",
        "name": "Portugal",
        "obs_level": "country",
        "n_clusters": 3,
    },
    "IE": {
        "factors_path": "output/run/IE-all-obs_country-corrected-calc_z0/training/correction-factors/IE_factors_fixed_3.csv",
        "geoms_path": "input/country_level_data/grid_points/ie/ie_correction_regions.geojson",
        "name": "Ireland",
        "obs_level": "country",
        "n_clusters": 3,
    },
}

# Turbine-level corrections
TURBINE_LEVEL_CONFIGS = {
    "DE-onshore": {
        "path": "output/correction_geodataframes_turbine/de/de_onshore_corrections_fixed_500.geojson",
        "name": "Germany",
        "obs_level": "turbine",
        "cluster_mode": "onshore",
        "n_clusters": 500,
    },
    "DK-onshore": {
        "path": "output/correction_geodataframes_turbine/dk/dk_onshore_corrections_fixed_1000.geojson",
        "name": "Denmark",
        "obs_level": "turbine",
        "cluster_mode": "onshore",
        "n_clusters": 1000,
    },
    "DK-offshore": {
        "path": "output/correction_geodataframes_turbine/dk/dk_offshore_corrections_fixed_2.geojson",
        "name": "Denmark",
        "obs_level": "turbine",
        "cluster_mode": "offshore",
        "n_clusters": 2,
    },
    "UK-onshore": {
        "path": "output/correction_geodataframes_turbine/uk/uk_onshore_corrections_fixed_300.geojson",
        "name": "United Kingdom",
        "obs_level": "turbine",
        "cluster_mode": "onshore",
        "n_clusters": 300,
    },
    "UK-offshore": {
        "path": "output/correction_geodataframes_turbine/uk/uk_offshore_corrections_fixed_10.geojson",
        "name": "United Kingdom",
        "obs_level": "turbine",
        "cluster_mode": "offshore",
        "n_clusters": 10,
    },
}


def create_no_correction_geodataframe():
    """Create correction GeoDataFrame for Norway (bidding zones)."""
    print("Creating Norway correction GeoDataFrame...")

    # Load correction factors
    factors = pd.read_csv(COUNTRY_LEVEL_CONFIGS["NO"]["factors_path"])

    # Load bidding zone geometries
    zones = gpd.read_file(COUNTRY_LEVEL_CONFIGS["NO"]["geoms_path"])

    # Merge on cluster ID
    # Bidding zones use cluster IDs 0-4 for NO_1 to NO_5
    gdf = zones.merge(factors, on='cluster', how='left')

    # Save
    output_dir = Path("output/correction_geodataframes/no")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "no_corrections_fixed_5.geojson"

    gdf.to_file(output_path, driver='GeoJSON')
    print(f"  ✓ Saved: {output_path}")

    # Update config
    COUNTRY_LEVEL_CONFIGS["NO"]["path"] = str(output_path)

    return gdf


def create_country_correction_geodataframe(country_code, config):
    """Create correction GeoDataFrame from factors and geometries.

    Args:
        country_code: Country code (e.g., 'ES', 'SE')
        config: Country configuration dictionary

    Returns:
        GeoDataFrame with corrections and geometries
    """
    print(f"Creating {config['name']} correction GeoDataFrame...")

    # Load correction factors
    factors = pd.read_csv(config["factors_path"])

    # Load cluster geometries
    geoms = gpd.read_file(config["geoms_path"])

    # Merge on cluster ID
    gdf = geoms.merge(factors, on='cluster', how='left')

    # Save
    output_dir = Path(f"output/correction_geodataframes/{country_code.lower()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{country_code.lower()}_corrections_fixed_{config['n_clusters']}.geojson"

    gdf.to_file(output_path, driver='GeoJSON')
    print(f"  ✓ Saved: {output_path}")

    # Update config
    config["path"] = str(output_path)

    return gdf


def extract_cluster_centroid(geometry):
    """Extract centroid from polygon geometry.

    Args:
        geometry: Shapely polygon or multipolygon.

    Returns:
        Tuple of (lon, lat) for centroid, or (None, None) if geometry is invalid.
    """
    if geometry is None or geometry.is_empty:
        return None, None

    # Handle GeometryCollections from clipping
    if geometry.geom_type == 'GeometryCollection':
        # Get the largest polygon from the collection
        polygons = [g for g in geometry.geoms if g.geom_type in ['Polygon', 'MultiPolygon']]
        if not polygons:
            return None, None
        geometry = max(polygons, key=lambda g: g.area)

    centroid = geometry.centroid
    if centroid.is_empty:
        return None, None

    return centroid.x, centroid.y


def load_all_corrections():
    """Load all correction GeoDataFrames and combine into unified dataframe.

    Returns:
        GeoDataFrame with all corrections and metadata.
    """
    print("\nLoading all correction GeoDataFrames...")

    all_corrections = []

    # Load country-level corrections
    print("\n" + "="*80)
    print("Country-Level Corrections")
    print("="*80)

    for country_code, config in COUNTRY_LEVEL_CONFIGS.items():
        print(f"\n  Loading {config['name']} ({country_code})...")

        # Create GeoDataFrame if it doesn't exist (for countries with factors_path/geoms_path)
        if "path" not in config:
            if country_code == "NO":
                gdf = create_no_correction_geodataframe()
            else:
                gdf = create_country_correction_geodataframe(country_code, config)
        else:
            try:
                gdf = gpd.read_file(config["path"])
            except Exception as e:
                print(f"    ✗ Error loading {config['path']}: {e}")
                continue

        # Extract centroid coordinates
        gdf['centroid_lon'], gdf['centroid_lat'] = zip(*gdf['geometry'].apply(extract_cluster_centroid))

        # Filter out clusters with empty geometries (None centroids)
        valid_mask = gdf['centroid_lon'].notna()
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            print(f"    ! Filtered out {n_invalid} clusters with empty geometries")
            gdf = gdf[valid_mask].copy()

        # Add metadata
        gdf['country_code'] = country_code
        gdf['country_name'] = config['name']
        gdf['obs_level'] = config['obs_level']
        gdf['cluster_mode'] = 'all'  # Country-level doesn't separate onshore/offshore

        # Calculate cluster area (for weighting if needed)
        gdf['area_km2'] = gdf.geometry.to_crs(epsg=3857).area / 1e6  # Convert m² to km²

        print(f"    ✓ Loaded {len(gdf)} clusters")
        print(f"      Scalar range: [{gdf['scalar'].min():.3f}, {gdf['scalar'].max():.3f}]")
        print(f"      Offset range: [{gdf['offset'].min():.3f}, {gdf['offset'].max():.3f}]")

        all_corrections.append(gdf)

    # Load turbine-level corrections
    print("\n" + "="*80)
    print("Turbine-Level Corrections")
    print("="*80)

    for config_name, config in TURBINE_LEVEL_CONFIGS.items():
        print(f"\n  Loading {config['name']} {config['cluster_mode']} ({config_name})...")

        try:
            gdf = gpd.read_file(config["path"])
        except Exception as e:
            print(f"    ✗ Error loading {config['path']}: {e}")
            continue

        # Extract centroid coordinates
        gdf['centroid_lon'], gdf['centroid_lat'] = zip(*gdf['geometry'].apply(extract_cluster_centroid))

        # Filter out clusters with empty geometries (None centroids)
        valid_mask = gdf['centroid_lon'].notna()
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            print(f"    ! Filtered out {n_invalid} clusters with empty geometries")
            gdf = gdf[valid_mask].copy()

        # Add metadata
        # For turbine-level, use full config name as country_code to differentiate onshore/offshore
        gdf['country_code'] = config_name  # e.g., "DE-onshore", "DK-offshore"
        gdf['country_name'] = config['name']
        gdf['obs_level'] = config['obs_level']
        gdf['cluster_mode'] = config['cluster_mode']

        # Calculate cluster area
        gdf['area_km2'] = gdf.geometry.to_crs(epsg=3857).area / 1e6

        print(f"    ✓ Loaded {len(gdf)} clusters")
        print(f"      Scalar range: [{gdf['scalar'].min():.3f}, {gdf['scalar'].max():.3f}]")
        print(f"      Offset range: [{gdf['offset'].min():.3f}, {gdf['offset'].max():.3f}]")

        all_corrections.append(gdf)

    # Combine all
    combined = gpd.GeoDataFrame(pd.concat(all_corrections, ignore_index=True))

    print(f"\n✓ Combined {len(combined)} total clusters from {len(all_corrections)} configurations")
    print(f"  Country-level: {len(COUNTRY_LEVEL_CONFIGS)} configurations")
    print(f"  Turbine-level: {len(TURBINE_LEVEL_CONFIGS)} configurations")

    return combined


def create_interpolation_dataframe(gdf):
    """Create simplified dataframe for interpolation.

    Converts from polygon geometries to point centroids with correction values.

    Args:
        gdf: GeoDataFrame with correction polygons.

    Returns:
        DataFrame with centroid coordinates and correction values.
    """
    print("\nCreating interpolation dataframe...")

    # Create point geometries from centroids
    points = gdf.copy()
    points['geometry'] = [Point(lon, lat) for lon, lat in zip(gdf['centroid_lon'], gdf['centroid_lat'])]

    # Select columns for interpolation
    interp_df = points[[
        'country_code',
        'country_name',
        'cluster',
        'cluster_mode',
        'obs_level',
        'centroid_lon',
        'centroid_lat',
        'scalar',
        'offset',
        'area_km2',
        'geometry'
    ]].copy()

    # Rename for clarity
    interp_df = interp_df.rename(columns={
        'centroid_lon': 'lon',
        'centroid_lat': 'lat'
    })

    print(f"  ✓ Created interpolation dataframe with {len(interp_df)} points")

    return interp_df


def save_outputs(gdf, interp_df):
    """Save outputs in multiple formats.

    Args:
        gdf: Full GeoDataFrame with polygon geometries.
        interp_df: Simplified dataframe with point centroids.
    """
    print("\nSaving outputs...")

    output_dir = Path("output/unified_corrections")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full GeoDataFrame with polygons (for visualization)
    polygons_path = output_dir / "all_corrections_polygons.geojson"
    gdf.to_file(polygons_path, driver='GeoJSON')
    print(f"  ✓ Polygons GeoJSON: {polygons_path}")

    # 2. Centroid points GeoDataFrame (for interpolation)
    points_path = output_dir / "all_corrections_centroids.geojson"
    interp_df.to_file(points_path, driver='GeoJSON')
    print(f"  ✓ Centroids GeoJSON: {points_path}")

    # 3. CSV without geometry (for ML/interpolation)
    csv_path = output_dir / "all_corrections_centroids.csv"
    interp_df_csv = interp_df.copy()
    interp_df_csv = interp_df_csv.drop(columns=['geometry'])
    interp_df_csv.to_csv(csv_path, index=False)
    print(f"  ✓ CSV (no geometry): {csv_path}")

    # 4. Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print("\nCountry-Level Corrections:")
    print("-" * 80)
    for country in ['NL', 'FR', 'BE', 'NO', 'ES', 'SE', 'IT', 'PT', 'IE']:
        if country in gdf['country_code'].values:
            country_data = gdf[gdf['country_code'] == country]
            print(f"\n{country} ({country_data['country_name'].iloc[0]}):")
            print(f"  Clusters: {len(country_data)}")
            print(f"  Total area: {country_data['area_km2'].sum():.0f} km²")
            print(f"  Scalar: {country_data['scalar'].mean():.3f} ± {country_data['scalar'].std():.3f}")
            print(f"  Offset: {country_data['offset'].mean():.3f} ± {country_data['offset'].std():.3f}")

    print("\n" + "-" * 80)
    print("Turbine-Level Corrections:")
    print("-" * 80)
    for config_name in ['DE-onshore', 'DK-onshore', 'DK-offshore', 'UK-onshore', 'UK-offshore']:
        if config_name in gdf['country_code'].values:
            config_data = gdf[gdf['country_code'] == config_name]
            print(f"\n{config_name} ({config_data['country_name'].iloc[0]}):")
            print(f"  Clusters: {len(config_data)}")
            print(f"  Total area: {config_data['area_km2'].sum():.0f} km²")
            print(f"  Scalar: {config_data['scalar'].mean():.3f} ± {config_data['scalar'].std():.3f}")
            print(f"  Offset: {config_data['offset'].mean():.3f} ± {config_data['offset'].std():.3f}")

    print("\n" + "="*80)
    print(f"Overall Statistics:")
    print(f"  Total clusters: {len(gdf)}")
    print(f"  Total area: {gdf['area_km2'].sum():.0f} km²")
    print(f"  Scalar range: [{gdf['scalar'].min():.3f}, {gdf['scalar'].max():.3f}]")
    print(f"  Offset range: [{gdf['offset'].min():.3f}, {gdf['offset'].max():.3f}]")

    # Breakdown by obs_level
    print(f"\nBy observation level:")
    for obs_level in gdf['obs_level'].unique():
        count = len(gdf[gdf['obs_level'] == obs_level])
        print(f"  {obs_level}: {count} clusters")

    return polygons_path, points_path, csv_path


def main():
    """Main execution."""
    print("="*80)
    print("Creating Unified Correction Geometry Dataframe")
    print("="*80)

    # Load all corrections
    combined_gdf = load_all_corrections()

    # Create interpolation dataframe
    interp_df = create_interpolation_dataframe(combined_gdf)

    # Save outputs
    save_outputs(combined_gdf, interp_df)

    print("\n" + "="*80)
    print("✓ COMPLETE")
    print("="*80)
    print("\nOutputs:")
    print("  1. output/unified_corrections/all_corrections_polygons.geojson")
    print("     - Full cluster polygons for visualization in QGIS")
    print("\n  2. output/unified_corrections/all_corrections_centroids.geojson")
    print("     - Cluster centroids as points for interpolation")
    print("\n  3. output/unified_corrections/all_corrections_centroids.csv")
    print("     - CSV format for ML/interpolation (lon, lat, scalar, offset)")

    print("\nUsage examples:")
    print("  - Kriging: Use centroids CSV with pykrige")
    print("  - IDW: Use centroids CSV with scipy.interpolate")
    print("  - ML models: Train on (lon, lat) → (scalar, offset)")
    print("  - Visualization: Load polygons GeoJSON in QGIS")


if __name__ == "__main__":
    main()
