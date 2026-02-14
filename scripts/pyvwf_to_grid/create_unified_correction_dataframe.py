"""Create unified correction geometry dataframe for spatial interpolation.

This script:
1. Loads all correction factor CSVs and cluster geometry files
2. Combines them into a single dataframe with cluster centroids
3. Adds country metadata for tracking
4. Exports as both GeoJSON (for visualization) and CSV (for interpolation)

Output can be used for:
- Kriging/IDW interpolation to arbitrary locations
- Training machine learning models for correction prediction
- Spatial analysis of correction patterns across Europe
- Creating continent-wide correction maps
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

# Base path for the turbine_grid run
RUNS_DIR = Path("output/runs/turbine_grid")

# Country-level corrections: factors from run, geometries from input data
COUNTRY_LEVEL_CONFIGS = {
    "NL": {
        "factors_path": RUNS_DIR / "NL-all-obs_country-corrected-calc_z0/training/correction-factors/NL_factors_fixed_5.csv",
        "geoms_path": "input/country_level_data/grid_points/nl/nl_correction_regions.geojson",
        "name": "Netherlands",
        "obs_level": "country",
        "n_clusters": 5,
    },
    "FR": {
        "factors_path": RUNS_DIR / "FR-all-obs_country-corrected-calc_z0/training/correction-factors/FR_factors_fixed_10.csv",
        "geoms_path": "input/country_level_data/grid_points/fr/fr_correction_regions.geojson",
        "name": "France",
        "obs_level": "country",
        "n_clusters": 10,
    },
    "BE": {
        "factors_path": RUNS_DIR / "BE-all-obs_country-corrected-calc_z0/training/correction-factors/BE_factors_fixed_3.csv",
        "geoms_path": "input/country_level_data/grid_points/be/be_correction_regions.geojson",
        "name": "Belgium",
        "obs_level": "country",
        "n_clusters": 3,
    },
    "NO": {
        "factors_path": RUNS_DIR / "NO-all-obs_country-corrected-calc_z0/training/correction-factors/NO_factors_fixed_5.csv",
        "geoms_path": "input/country_level_data/grid_points/no/no_correction_regions.geojson",
        "name": "Norway",
        "obs_level": "country",
        "n_clusters": 5,
    },
    "ES": {
        "factors_path": RUNS_DIR / "ES-all-obs_country-corrected-calc_z0/training/correction-factors/ES_factors_fixed_4.csv",
        "geoms_path": "input/country_level_data/grid_points/es/es_correction_regions.geojson",
        "name": "Spain",
        "obs_level": "country",
        "n_clusters": 4,
    },
    "SE": {
        "factors_path": RUNS_DIR / "SE-all-obs_country-corrected-calc_z0/training/correction-factors/SE_factors_fixed_4.csv",
        "geoms_path": "input/country_level_data/grid_points/se/se_correction_regions.geojson",
        "name": "Sweden",
        "obs_level": "country",
        "n_clusters": 4,
    },
    "IT": {
        "factors_path": RUNS_DIR / "IT-all-obs_country-corrected-calc_z0/training/correction-factors/IT_factors_fixed_3.csv",
        "geoms_path": "input/country_level_data/grid_points/it/it_correction_regions.geojson",
        "name": "Italy",
        "obs_level": "country",
        "n_clusters": 3,
    },
    "PT": {
        "factors_path": RUNS_DIR / "PT-all-obs_country-corrected-calc_z0/training/correction-factors/PT_factors_fixed_3.csv",
        "geoms_path": "input/country_level_data/grid_points/pt/pt_correction_regions.geojson",
        "name": "Portugal",
        "obs_level": "country",
        "n_clusters": 3,
    },
    "IE": {
        "factors_path": RUNS_DIR / "IE-all-obs_country-corrected-calc_z0/training/correction-factors/IE_factors_fixed_3.csv",
        "geoms_path": "input/country_level_data/grid_points/ie/ie_correction_regions.geojson",
        "name": "Ireland",
        "obs_level": "country",
        "n_clusters": 3,
    },
}

# Turbine-level corrections: factors from run, geometries from cluster geometry files
TURBINE_LEVEL_CONFIGS = {
    "DE-onshore": {
        "factors_path": RUNS_DIR / "DE-onshore-obs_turbine-corrected-calc_z0/training/correction-factors/DE_factors_fixed_500.csv",
        "geoms_path": "output/pyvwf_to_grid/cluster_geometries/de/de_onshore_correction_regions_500.geojson",
        "name": "Germany",
        "obs_level": "turbine",
        "cluster_mode": "onshore",
        "n_clusters": 500,
    },
    "DK-onshore": {
        "factors_path": RUNS_DIR / "DK-onshore-obs_turbine-corrected-calc_z0/training/correction-factors/DK_factors_fixed_700.csv",
        "geoms_path": "output/grid_run/turbine_grid/cluster_geometries/dk/dk_onshore_correction_regions_700.geojson",
        "name": "Denmark",
        "obs_level": "turbine",
        "cluster_mode": "onshore",
        "n_clusters": 700,
    },
    "DK-offshore": {
        "factors_path": RUNS_DIR / "DK-offshore-obs_turbine-corrected-calc_z0/training/correction-factors/DK_factors_fixed_2.csv",
        "geoms_path": "output/pyvwf_to_grid/cluster_geometries/dk/dk_offshore_correction_regions_2.geojson",
        "name": "Denmark",
        "obs_level": "turbine",
        "cluster_mode": "offshore",
        "n_clusters": 2,
    },
    "UK-onshore": {
        "factors_path": RUNS_DIR / "UK-onshore-obs_turbine-corrected-calc_z0/training/correction-factors/UK_factors_fixed_300.csv",
        "geoms_path": "output/pyvwf_to_grid/cluster_geometries/uk/uk_onshore_correction_regions_300.geojson",
        "name": "United Kingdom",
        "obs_level": "turbine",
        "cluster_mode": "onshore",
        "n_clusters": 300,
    },
    "UK-offshore": {
        "factors_path": RUNS_DIR / "UK-offshore-obs_turbine-corrected-calc_z0/training/correction-factors/UK_factors_fixed_10.csv",
        "geoms_path": "output/pyvwf_to_grid/cluster_geometries/uk/uk_offshore_correction_regions_10.geojson",
        "name": "United Kingdom",
        "obs_level": "turbine",
        "cluster_mode": "offshore",
        "n_clusters": 10,
    },
}


def create_correction_geodataframe(config_name, config):
    """Create correction GeoDataFrame from factors CSV and geometry file.

    Args:
        config_name: Config identifier (e.g., 'NL', 'DE-onshore').
        config: Configuration dictionary with factors_path and geoms_path.

    Returns:
        GeoDataFrame with corrections and geometries.
    """
    print(f"  Creating {config['name']} correction GeoDataFrame...")

    # Load correction factors
    factors = pd.read_csv(config["factors_path"])

    # Load cluster geometries
    geoms = gpd.read_file(config["geoms_path"])

    # Drop any existing scalar/offset columns from geoms (in case it's a pre-merged file)
    drop_cols = [c for c in ['scalar', 'offset', 'fixed'] if c in geoms.columns]
    if drop_cols:
        geoms = geoms.drop(columns=drop_cols)

    # Merge on cluster ID
    gdf = geoms.merge(factors, on='cluster', how='left')

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

        try:
            gdf = create_correction_geodataframe(country_code, config)
        except Exception as e:
            print(f"    ✗ Error: {e}")
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
            gdf = create_correction_geodataframe(config_name, config)
        except Exception as e:
            print(f"    ✗ Error: {e}")
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


def save_outputs(gdf, interp_df, output_dir):
    """Save outputs in multiple formats.

    Args:
        gdf: Full GeoDataFrame with polygon geometries.
        interp_df: Simplified dataframe with point centroids.
        output_dir: Output directory path.
    """
    print("\nSaving outputs...")

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
    parser = argparse.ArgumentParser(
        description="Create unified correction geometry dataframe"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/grid_run/turbine_grid"),
        help="Output directory (default: output/grid_run/turbine_grid)",
    )
    args = parser.parse_args()

    print("="*80)
    print("Creating Unified Correction Geometry Dataframe")
    print("="*80)

    # Load all corrections
    combined_gdf = load_all_corrections()

    # Create interpolation dataframe
    interp_df = create_interpolation_dataframe(combined_gdf)

    # Save outputs
    save_outputs(combined_gdf, interp_df, args.output_dir)

    print("\n" + "="*80)
    print("✓ COMPLETE")
    print("="*80)
    print(f"\nOutputs in {args.output_dir}:")
    print("  1. all_corrections_polygons.geojson")
    print("     - Full cluster polygons for visualization in QGIS")
    print("\n  2. all_corrections_centroids.geojson")
    print("     - Cluster centroids as points for interpolation")
    print("\n  3. all_corrections_centroids.csv")
    print("     - CSV format for ML/interpolation (lon, lat, scalar, offset)")


if __name__ == "__main__":
    main()
