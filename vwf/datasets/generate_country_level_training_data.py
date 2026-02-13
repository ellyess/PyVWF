"""Generate Training and Test Data for Country-Level PyVWF Workflows.

This script prepares data for NL, NO, FR, BE, and Phase 1 countries (ES, SE, IT, PT, IE) using:
1. ENTSO-E API for country-level wind generation observations
2. Grid-based sampling points with representative turbine metadata
3. Training/test split by year

Designed to work with PyVWF model using obs_level="country".

Requirements:
    pip install entsoe-py pandas geopandas shapely

Setup:
    export ENTSOE_API_KEY='your-key-here'
    Get key at: https://transparency.entsoe.eu/

Usage:
    # Generate all countries with default settings
    python vwf/datasets/generate_country_level_training_data.py

    # Custom years and countries
    python vwf/datasets/generate_country_level_training_data.py \
        --countries NL FR ES SE \
        --train-years 2018 2019 \
        --test-year 2020

    # Specific output directory
    python vwf/datasets/generate_country_level_training_data.py \
        --output-dir input/country_level_data
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
from shapely.geometry import box

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vwf.datasets.fetch_entsoe_capacity_factors import ENTSOEWindDataFetcher, save_capacity_factors
from vwf.clustering import create_sampling_points, cluster_with_geometries


# Country configurations
COUNTRY_CONFIGS = {
    "NL": {
        "name": "Netherlands",
        "bounds": box(3.3, 50.7, 7.2, 53.6),
        "height": 100.0,        # Modern onshore fleet
        "model": "V90",         # Vestas 3MW
        "capacity": 3.0,        # Average capacity
        "grid_resolution": 0.25,  # ~25km grid
        "num_clusters": 5,      # Spatial regions
    },
    "FR": {
        "name": "France",
        "bounds": box(-5.0, 42.0, 8.5, 51.2),
        "height": 90.0,         # Mix of old and modern
        "model": "V80",         # Vestas 2MW
        "capacity": 2.5,
        "grid_resolution": 0.5,   # ~50km grid (larger country)
        "num_clusters": 10,     # More regions
    },
    "BE": {
        "name": "Belgium",
        "bounds": box(2.5, 49.5, 6.4, 51.5),
        "height": 100.0,        # Modern fleet
        "model": "V90",
        "capacity": 3.0,
        "grid_resolution": 0.25,
        "num_clusters": 3,      # Smaller country
    },
    "NO": {
        "name": "Norway",
        "bounds": box(4.5, 58.0, 31.0, 71.5),  # Full country (for reference)
        "height": 80.0,         # Mountain terrain, lower heights
        "model": "V90",
        "capacity": 3.0,
        "grid_resolution": 1.0,   # ~100km grid (large country, sparse turbines)
        "use_bidding_zones": True,  # ← Use zones instead of KMeans
        "note": "Norway uses bidding zones (NO_1..NO_5) for market structure"
    },
    # Phase 1 Countries (moderate clusters with bbox optimization)
    "ES": {
        "name": "Spain",
        "bounds": box(-9.5, 36.0, 3.5, 43.8),
        "height": 90.0,         # Modern fleet
        "model": "Vestas.V90.2000",  # 2 MW turbine
        "capacity": 2.5,
        "grid_resolution": 1.0,   # ~100km grid (large country)
        "num_clusters": 4,      # Increased with bbox optimization
    },
    "SE": {
        "name": "Sweden",
        "bounds": box(11.0, 55.3, 24.2, 69.0),  # Full country (for reference)
        "height": 100.0,        # Modern, tall turbines
        "model": "Vestas.V90.3000",  # 3 MW turbine
        "capacity": 3.0,
        "grid_resolution": 1.5,   # ~150km grid (very large country)
        "use_bidding_zones": True,  # ← Use zones instead of KMeans
        "note": "Sweden uses bidding zones (SE_1..SE_4) for market structure"
    },
    "IT": {
        "name": "Italy",
        "bounds": box(6.6, 36.6, 18.5, 47.1),
        "height": 80.0,         # Mix of old and modern
        "model": "Vestas.V80.2000",  # 2 MW turbine
        "capacity": 2.0,
        "grid_resolution": 1.0,   # ~100km grid
        "num_clusters": 3,      # Increased with bbox optimization
    },
    "PT": {
        "name": "Portugal",
        "bounds": box(-9.5, 37.0, -6.2, 42.2),
        "height": 80.0,
        "model": "Vestas.V80.2000",  # 2 MW turbine
        "capacity": 2.5,
        "grid_resolution": 0.5,   # ~50km grid
        "num_clusters": 3,      # Increased with bbox optimization
    },
    "IE": {
        "name": "Ireland",
        "bounds": box(-10.5, 51.4, -5.4, 55.4),
        "height": 85.0,
        "model": "Vestas.V90.2000",  # 2 MW turbine
        "capacity": 2.5,
        "grid_resolution": 0.5,   # ~50km grid
        "num_clusters": 3,      # Increased with bbox optimization
    },
}

# Norwegian bidding zone boundaries (approximate)
# Based on electricity market zones, not exact geographic boundaries
NORWAY_ZONES = {
    "NO_1": {
        "name": "NO1 - Oslo / Eastern Norway",
        "bounds": box(9.5, 58.0, 12.5, 62.0),
        "grid_resolution": 0.5,
    },
    "NO_2": {
        "name": "NO2 - Kristiansand / Southern Norway",
        "bounds": box(5.5, 58.0, 9.5, 60.0),
        "grid_resolution": 0.5,
    },
    "NO_3": {
        "name": "NO3 - Trondheim / Mid-Norway",
        "bounds": box(8.0, 62.0, 14.0, 65.5),
        "grid_resolution": 0.75,
    },
    "NO_4": {
        "name": "NO4 - Tromsø / Northern Norway",
        "bounds": box(15.0, 65.5, 31.0, 71.5),
        "grid_resolution": 1.0,  # Larger, sparser region
    },
    "NO_5": {
        "name": "NO5 - Bergen / Western Norway",
        "bounds": box(4.5, 60.0, 8.0, 62.5),
        "grid_resolution": 0.5,
    },
}

SWEDEN_ZONES = {
    "SE_1": {
        "name": "SE1 - Luleå / Northern Sweden",
        "bounds": box(11.0, 63.5, 24.2, 69.0),
        "grid_resolution": 1.5,  # Largest, most sparse region
    },
    "SE_2": {
        "name": "SE2 - Sundsvall / North-Central Sweden",
        "bounds": box(11.5, 60.5, 20.0, 63.5),
        "grid_resolution": 1.0,
    },
    "SE_3": {
        "name": "SE3 - Stockholm / Central Sweden",
        "bounds": box(11.5, 58.0, 19.0, 60.5),
        "grid_resolution": 0.75,
    },
    "SE_4": {
        "name": "SE4 - Malmö / Southern Sweden",
        "bounds": box(11.0, 55.3, 19.0, 58.0),
        "grid_resolution": 0.5,  # Most wind capacity, finest resolution
    },
}


def generate_grid_points(
    country: str,
    config: dict,
    output_dir: Path,
    save_geojson: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate grid points with turbine metadata and cluster geometries.

    For Norway, automatically creates separate grids for each bidding zone.

    Args:
        country: Country code (NL, FR, BE, NO).
        config: Country configuration dictionary.
        output_dir: Output directory for saving files.
        save_geojson: If True, save cluster geometries as GeoJSON.

    Returns:
        Tuple of (grid_points_clustered, cluster_geometries).
    """
    print(f"\n{'='*70}")
    print(f"Generating Grid Points for {config['name']} ({country})")
    print(f"{'='*70}")

    # Special handling for Norway - use bidding zones
    if country.upper() == "NO" and config.get("use_bidding_zones", False):
        return generate_norway_zone_grids(config, output_dir, save_geojson)

    # Special handling for Sweden - use bidding zones
    if country.upper() == "SE" and config.get("use_bidding_zones", False):
        return generate_sweden_zone_grids(config, output_dir, save_geojson)

    # Standard KMeans clustering for other countries
    return generate_kmeans_grid(country, config, output_dir, save_geojson)


def generate_norway_zone_grids(
    config: dict,
    output_dir: Path,
    save_geojson: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate grid points for Norwegian bidding zones using actual zone geometries.

    Creates separate grid for each zone, with zone ID as cluster.

    Args:
        config: Norway configuration dictionary.
        output_dir: Output directory.
        save_geojson: If True, save zone geometries.

    Returns:
        Tuple of (all_grid_points, zone_geometries).
    """
    print("\n✓ Norway detected - using actual bidding zone geometries")
    print("  Each zone will be treated as a separate cluster\n")

    # Load actual bidding zone geometries
    import geopandas as gpd
    zones_path = Path("input/regions/no_bidding_zones.geojson")
    if not zones_path.exists():
        print(f"  ✗ Bidding zones file not found: {zones_path}")
        print("  Falling back to bounding boxes")
        # Fall back to old behavior with bounding boxes
        return generate_norway_zone_grids_fallback(config, output_dir, save_geojson)

    zones_gdf = gpd.read_file(zones_path)
    print(f"  ✓ Loaded {len(zones_gdf)} bidding zones from: {zones_path}")

    all_zones_grids = []
    zone_geometries = []

    # Map zone names: 'NO 1' -> 'NO_1' for consistency
    zone_name_map = {
        'NO 1': 'NO_1',
        'NO 2': 'NO_2',
        'NO 3': 'NO_3',
        'NO 4': 'NO_4',
        'NO 5': 'NO_5',
    }

    for idx, zone_row in zones_gdf.iterrows():
        zone_name_orig = zone_row['Price area']
        zone_id = zone_name_map.get(zone_name_orig, zone_name_orig.replace(' ', '_'))
        zone_geom = zone_row.geometry

        # Extract zone number (NO_1 -> 0, NO_2 -> 1, etc.)
        zone_num = int(zone_id.split('_')[1]) - 1

        # Get grid resolution from NORWAY_ZONES config
        zone_config = NORWAY_ZONES.get(zone_id, {'grid_resolution': 0.5})
        grid_resolution = zone_config.get('grid_resolution', 0.5)

        print(f"{'─'*70}")
        print(f"Zone {zone_id} ({zone_name_orig})")
        print(f"{'─'*70}")
        print(f"  Grid resolution: {grid_resolution}°")
        print(f"  Turbine metadata: {config['height']}m, {config['model']}, {config['capacity']}MW")

        # Create grid for this zone using actual geometry bounds
        zone_bounds = zone_geom.bounds  # (minx, miny, maxx, maxy)
        zone_bbox = box(zone_bounds[0], zone_bounds[1], zone_bounds[2], zone_bounds[3])

        zone_grid = create_sampling_points(
            country_bounds=zone_bbox,
            method="grid",
            resolution=grid_resolution,
            add_metadata=True,
            default_height=config['height'],
            default_model=config['model'],
            default_capacity=config['capacity'],
        )

        # Filter grid points to only those inside the actual zone geometry
        # Create point geometries for filtering
        from shapely.geometry import Point as ShapelyPoint
        zone_grid_gdf = gpd.GeoDataFrame(
            zone_grid,
            geometry=[ShapelyPoint(lon, lat) for lon, lat in zip(zone_grid['lon'], zone_grid['lat'])],
            crs="EPSG:4326"
        )

        # Keep only points inside the actual zone geometry
        zone_grid_filtered = zone_grid_gdf[zone_grid_gdf.geometry.within(zone_geom)].copy()
        zone_grid_filtered.drop(columns=['geometry'], inplace=True)

        # Assign zone as cluster
        zone_grid_filtered['cluster'] = zone_num
        zone_grid_filtered['zone'] = zone_id

        print(f"  ✓ Created {len(zone_grid_filtered)} grid points for {zone_id} (filtered from {len(zone_grid)})")

        all_zones_grids.append(zone_grid_filtered)

        # Use actual zone geometry
        zone_geometries.append({
            'cluster': zone_num,
            'zone': zone_id,
            'name': zone_config.get('name', zone_name_orig),
            'geometry': zone_geom,
            'n_points': len(zone_grid_filtered)
        })

    # Combine all zones
    grid_all_zones = pd.concat(all_zones_grids, ignore_index=True)

    print(f"\n{'─'*70}")
    print(f"✓ Combined grid: {len(grid_all_zones)} points across {len(zone_geometries)} zones")
    print(f"\nZone distribution:")
    for zone_id in sorted(grid_all_zones['zone'].unique()):
        count = len(grid_all_zones[grid_all_zones['zone'] == zone_id])
        print(f"  {zone_id}: {count} points")

    # Save grid points
    grid_dir = output_dir / "grid_points" / "no"
    grid_dir.mkdir(parents=True, exist_ok=True)

    grid_path = grid_dir / "no_grid_points_zones.csv"
    grid_all_zones.to_csv(grid_path, index=False)
    print(f"\n✓ Saved grid points: {grid_path}")

    # Save zone geometries as GeoJSON
    if save_geojson:
        try:
            zone_gdf = gpd.GeoDataFrame(zone_geometries, crs="EPSG:4326")
            geom_path = grid_dir / "no_bidding_zones.geojson"
            zone_gdf.to_file(geom_path, driver="GeoJSON")
            print(f"✓ Saved zone geometries: {geom_path}")
        except ImportError:
            print("⚠ GeoPandas not available - skipping GeoJSON export")
            zone_gdf = pd.DataFrame(zone_geometries)

    else:
        zone_gdf = pd.DataFrame(zone_geometries)

    return grid_all_zones, zone_gdf


def generate_sweden_zone_grids(
    config: dict,
    output_dir: Path,
    save_geojson: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate grid points for Swedish bidding zones using actual zone geometries.

    Creates separate grid for each zone, with zone ID as cluster.

    Args:
        config: Sweden configuration dictionary.
        output_dir: Output directory.
        save_geojson: If True, save zone geometries.

    Returns:
        Tuple of (all_grid_points, zone_geometries).
    """
    print("\n✓ Sweden detected - using actual bidding zone geometries")
    print("  Each zone will be treated as a separate cluster\n")

    # Load actual bidding zone geometries
    import geopandas as gpd
    zones_path = Path("input/regions/se_bidding_zones.geojson")
    if not zones_path.exists():
        print(f"  ✗ Bidding zones file not found: {zones_path}")
        print("  ⚠ Cannot proceed without zone geometries")
        raise FileNotFoundError(f"Swedish bidding zones file not found: {zones_path}")

    zones_gdf = gpd.read_file(zones_path)
    print(f"  ✓ Loaded {len(zones_gdf)} bidding zones from: {zones_path}")

    all_zones_grids = []
    zone_geometries = []

    for idx, zone_row in zones_gdf.iterrows():
        # Get zone_id from properties (already in SE_1 format from our extraction)
        zone_id = zone_row.get('zone_id') or zone_row.get('zone_name')
        if not zone_id:
            print(f"  ⚠ Warning: Missing zone_id for feature {idx}, skipping")
            continue

        zone_geom = zone_row.geometry

        # Extract zone number (SE_1 -> 0, SE_2 -> 1, etc.)
        zone_num = int(zone_id.split('_')[1]) - 1

        # Get grid resolution from SWEDEN_ZONES config
        zone_config = SWEDEN_ZONES.get(zone_id, {'grid_resolution': 0.75})
        grid_resolution = zone_config.get('grid_resolution', 0.75)

        print(f"{'─'*70}")
        print(f"Zone {zone_id}")
        print(f"{'─'*70}")
        print(f"  Grid resolution: {grid_resolution}°")
        print(f"  Turbine metadata: {config['height']}m, {config['model']}, {config['capacity']}MW")

        # Create grid for this zone using actual geometry bounds
        zone_bounds = zone_geom.bounds  # (minx, miny, maxx, maxy)
        zone_bbox = box(zone_bounds[0], zone_bounds[1], zone_bounds[2], zone_bounds[3])

        zone_grid = create_sampling_points(
            country_bounds=zone_bbox,
            method="grid",
            resolution=grid_resolution,
            add_metadata=True,
            default_height=config['height'],
            default_model=config['model'],
            default_capacity=config['capacity'],
        )

        # Filter grid points to only those inside the actual zone geometry
        # Create point geometries for filtering
        from shapely.geometry import Point as ShapelyPoint
        zone_grid_gdf = gpd.GeoDataFrame(
            zone_grid,
            geometry=[ShapelyPoint(lon, lat) for lon, lat in zip(zone_grid['lon'], zone_grid['lat'])],
            crs="EPSG:4326"
        )

        # Keep only points inside the actual zone geometry
        zone_grid_filtered = zone_grid_gdf[zone_grid_gdf.geometry.within(zone_geom)].copy()
        zone_grid_filtered.drop(columns=['geometry'], inplace=True)

        # Assign zone as cluster
        zone_grid_filtered['cluster'] = zone_num
        zone_grid_filtered['zone'] = zone_id

        print(f"  Grid points: {len(zone_grid_filtered)}")

        if len(zone_grid_filtered) == 0:
            print(f"  ⚠ Warning: No grid points generated for {zone_id}")
            continue

        all_zones_grids.append(zone_grid_filtered)

        # Use actual zone geometry
        zone_geometries.append({
            'cluster': zone_num,
            'zone': zone_id,
            'name': zone_config.get('name', zone_id),
            'geometry': zone_geom,
            'n_points': len(zone_grid_filtered)
        })

    # Combine all zones
    grid_all_zones = pd.concat(all_zones_grids, ignore_index=True)

    print(f"\n{'─'*70}")
    print(f"✓ Combined grid: {len(grid_all_zones)} points across {len(zone_geometries)} zones")
    print(f"\nZone distribution:")
    for zone_id in sorted(grid_all_zones['zone'].unique()):
        count = len(grid_all_zones[grid_all_zones['zone'] == zone_id])
        print(f"  {zone_id}: {count} points")

    # Save grid points
    grid_dir = output_dir / "grid_points" / "se"
    grid_dir.mkdir(parents=True, exist_ok=True)

    grid_path = grid_dir / "se_grid_points_zones.csv"
    grid_all_zones.to_csv(grid_path, index=False)
    print(f"\n✓ Saved grid points: {grid_path}")

    # Save zone geometries as GeoJSON
    if save_geojson:
        try:
            zone_gdf = gpd.GeoDataFrame(zone_geometries, crs="EPSG:4326")
            geom_path = grid_dir / "se_bidding_zones.geojson"
            zone_gdf.to_file(geom_path, driver="GeoJSON")
            print(f"✓ Saved zone geometries: {geom_path}")
        except ImportError:
            print("⚠ GeoPandas not available - skipping GeoJSON export")
            zone_gdf = pd.DataFrame(zone_geometries)

    else:
        zone_gdf = pd.DataFrame(zone_geometries)

    return grid_all_zones, zone_gdf


def generate_norway_zone_grids_fallback(
    config: dict,
    output_dir: Path,
    save_geojson: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fallback: Generate grid points for Norwegian bidding zones using bounding boxes.

    Creates separate grid for each zone, with zone ID as cluster.

    Args:
        config: Norway configuration dictionary.
        output_dir: Output directory.
        save_geojson: If True, save zone geometries.

    Returns:
        Tuple of (all_grid_points, zone_geometries).
    """
    print("\n✓ Using bounding box fallback for Norway zones")

    all_zones_grids = []
    zone_geometries = []

    for zone_id, zone_config in NORWAY_ZONES.items():
        print(f"{'─'*70}")
        print(f"Zone {zone_id}: {zone_config['name']}")
        print(f"{'─'*70}")

        # Extract zone number (NO_1 -> 0, NO_2 -> 1, etc.)
        zone_num = int(zone_id.split('_')[1]) - 1

        print(f"  Grid resolution: {zone_config['grid_resolution']}°")
        print(f"  Turbine metadata: {config['height']}m, {config['model']}, {config['capacity']}MW")

        # Create grid for this zone
        zone_grid = create_sampling_points(
            country_bounds=zone_config['bounds'],
            method="grid",
            resolution=zone_config['grid_resolution'],
            add_metadata=True,
            default_height=config['height'],
            default_model=config['model'],
            default_capacity=config['capacity'],
        )

        # Assign zone as cluster
        zone_grid['cluster'] = zone_num
        zone_grid['zone'] = zone_id

        print(f"  ✓ Created {len(zone_grid)} grid points for {zone_id}")

        all_zones_grids.append(zone_grid)

        # Create zone geometry (bounding box)
        from shapely.geometry import mapping
        zone_geometries.append({
            'cluster': zone_num,
            'zone': zone_id,
            'name': zone_config['name'],
            'geometry': zone_config['bounds'],
            'n_points': len(zone_grid)
        })

    # Combine all zones
    grid_all_zones = pd.concat(all_zones_grids, ignore_index=True)

    print(f"\n{'─'*70}")
    print(f"✓ Combined grid: {len(grid_all_zones)} points across {len(NORWAY_ZONES)} zones")
    print(f"\nZone distribution:")
    for zone_id in sorted(grid_all_zones['zone'].unique()):
        count = len(grid_all_zones[grid_all_zones['zone'] == zone_id])
        print(f"  {zone_id}: {count} points")

    # Save grid points
    grid_dir = output_dir / "grid_points" / "no"
    grid_dir.mkdir(parents=True, exist_ok=True)

    grid_path = grid_dir / "no_grid_points_zones.csv"
    grid_all_zones.to_csv(grid_path, index=False)
    print(f"\n✓ Saved grid points: {grid_path}")

    # Save zone geometries as GeoJSON
    if save_geojson:
        try:
            import geopandas as gpd
            zone_gdf = gpd.GeoDataFrame(zone_geometries, crs="EPSG:4326")
            geom_path = grid_dir / "no_bidding_zones.geojson"
            zone_gdf.to_file(geom_path, driver="GeoJSON")
            print(f"✓ Saved zone geometries: {geom_path}")
        except ImportError:
            print("⚠ GeoPandas not available - skipping GeoJSON export")
            zone_gdf = pd.DataFrame(zone_geometries)

    else:
        zone_gdf = pd.DataFrame(zone_geometries)

    return grid_all_zones, zone_gdf


def generate_kmeans_grid(
    country: str,
    config: dict,
    output_dir: Path,
    save_geojson: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate grid with KMeans clustering (standard approach).

    Args:
        country: Country code.
        config: Country configuration.
        output_dir: Output directory.
        save_geojson: If True, save geometries.

    Returns:
        Tuple of (grid_clustered, cluster_geometries).
    """

    # Create grid with turbine metadata
    print(f"\nGrid resolution: {config['grid_resolution']}° (~{config['grid_resolution']*100:.0f} km)")
    print(f"Representative turbine:")
    print(f"  Hub height: {config['height']} m")
    print(f"  Power curve: {config['model']}")
    print(f"  Capacity: {config['capacity']} MW")

    grid_points = create_sampling_points(
        country_bounds=config['bounds'],
        method="grid",
        resolution=config['grid_resolution'],
        add_metadata=True,
        default_height=config['height'],
        default_model=config['model'],
        default_capacity=config['capacity'],
    )

    print(f"\n✓ Created {len(grid_points)} grid points")

    # Cluster spatially with Voronoi tessellation
    print(f"\nClustering into {config['num_clusters']} regions (Voronoi)...")
    grid_clustered, cluster_geoms = cluster_with_geometries(
        sampling_points=grid_points,
        num_clusters=config['num_clusters'],
        method="kmeans",
        country_code=country,
        cluster_mode="onshore",
        geometry_type="voronoi"
    )

    # Print cluster distribution
    cluster_counts = grid_clustered['cluster'].value_counts().sort_index()
    print(f"\nCluster distribution:")
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} points")

    # Save grid points
    grid_dir = output_dir / "grid_points" / country.lower()
    grid_dir.mkdir(parents=True, exist_ok=True)

    grid_path = grid_dir / f"{country.lower()}_grid_points.csv"
    grid_clustered.to_csv(grid_path, index=False)
    print(f"\n✓ Saved grid points: {grid_path}")

    # Save cluster geometries as GeoJSON
    if save_geojson and cluster_geoms is not None:
        geom_path = grid_dir / f"{country.lower()}_correction_regions.geojson"
        cluster_geoms.to_file(geom_path, driver="GeoJSON")
        print(f"✓ Saved cluster geometries: {geom_path}")

    # Print metadata summary
    print(f"\nGrid points ready for PyVWF simulation!")
    print(f"Columns: {list(grid_clustered.columns)}")

    return grid_clustered, cluster_geoms


def fetch_observations(
    fetcher: ENTSOEWindDataFetcher,
    countries: list[str],
    train_years: list[int],
    test_year: int,
    output_dir: Path,
    psr_type: Literal["onshore", "offshore", "all"] = "all",
) -> dict:
    """Fetch ENTSO-E observations for training and test periods.

    For Norway and Sweden, automatically fetches each bidding zone separately.

    Args:
        fetcher: ENTSO-E API client.
        countries: List of country codes.
        train_years: List of training years.
        test_year: Test year.
        output_dir: Output directory.
        psr_type: Wind type ('onshore', 'offshore', 'all').

    Returns:
        Dictionary with training and test data for each country.
    """
    results = {}

    for country in countries:
        print(f"\n{'='*70}")
        print(f"Fetching Observations for {country}")
        print(f"{'='*70}")

        # Special handling for Norway - fetch zones separately
        if country.upper() == "NO":
            zone_results = fetch_norway_zone_observations(
                fetcher, train_years, test_year, output_dir, psr_type
            )
            results["NO"] = zone_results
            continue

        # Special handling for Sweden - fetch zones separately
        if country.upper() == "SE":
            zone_results = fetch_sweden_zone_observations(
                fetcher, train_years, test_year, output_dir, psr_type
            )
            results["SE"] = zone_results
            continue

        # Standard country-level fetch
        results[country] = fetch_country_observations(
            fetcher, country, train_years, test_year, output_dir, psr_type
        )

    return results


def fetch_norway_zone_observations(
    fetcher: ENTSOEWindDataFetcher,
    train_years: list[int],
    test_year: int,
    output_dir: Path,
    psr_type: Literal["onshore", "offshore", "all"] = "all",
) -> dict:
    """Fetch observations for Norwegian bidding zones.

    Args:
        fetcher: ENTSO-E API client.
        train_years: List of training years.
        test_year: Test year.
        output_dir: Output directory.
        psr_type: Wind type.

    Returns:
        Dictionary with zone data and aggregated country data.
    """
    print("\n✓ Norway detected - fetching bidding zones separately")

    zone_train_data = {}
    zone_test_data = {}

    # Fetch each zone
    for zone_id in NORWAY_ZONES.keys():
        print(f"\n{'─'*70}")
        print(f"Fetching {zone_id}")
        print(f"{'─'*70}")

        zone_results = fetch_country_observations(
            fetcher, zone_id, train_years, test_year, output_dir, psr_type
        )

        if zone_results:
            zone_train_data[zone_id] = zone_results["train"]
            zone_test_data[zone_id] = zone_results["test"]

    # Aggregate zones for country-level
    print(f"\n{'─'*70}")
    print("Aggregating zones to country-level")
    print(f"{'─'*70}")

    if zone_train_data:
        # Combine all zones with sum
        train_combined = pd.concat([
            df[['generation_mw', 'capacity_mw']] for df in zone_train_data.values()
        ], axis=1)

        train_agg = pd.DataFrame({
            'generation_mw': train_combined.filter(like='generation').sum(axis=1),
            'capacity_mw': train_combined.filter(like='capacity').sum(axis=1),
        })
        train_agg['capacity_factor'] = train_agg['generation_mw'] / train_agg['capacity_mw']
        train_agg['capacity_factor'] = train_agg['capacity_factor'].clip(0, 1.5)

        print(f"  ✓ Aggregated training: {len(train_agg)} data points")
        print(f"    Mean CF: {train_agg['capacity_factor'].mean():.2%}")

    else:
        train_agg = pd.DataFrame()

    if zone_test_data:
        test_combined = pd.concat([
            df[['generation_mw', 'capacity_mw']] for df in zone_test_data.values()
        ], axis=1)

        test_agg = pd.DataFrame({
            'generation_mw': test_combined.filter(like='generation').sum(axis=1),
            'capacity_mw': test_combined.filter(like='capacity').sum(axis=1),
        })
        test_agg['capacity_factor'] = test_agg['generation_mw'] / test_agg['capacity_mw']
        test_agg['capacity_factor'] = test_agg['capacity_factor'].clip(0, 1.5)

        print(f"  ✓ Aggregated test: {len(test_agg)} data points")
        print(f"    Mean CF: {test_agg['capacity_factor'].mean():.2%}")

    else:
        test_agg = pd.DataFrame()

    # Save aggregated country-level data
    obs_dir = output_dir / "observations" / "no"
    obs_dir.mkdir(parents=True, exist_ok=True)

    if not train_agg.empty:
        train_path = obs_dir / f"no_train_{min(train_years)}_{max(train_years)}_aggregated.csv"
        train_agg.to_csv(train_path)
        print(f"\n✓ Saved aggregated training: {train_path}")

    if not test_agg.empty:
        test_path = obs_dir / f"no_test_{test_year}_aggregated.csv"
        test_agg.to_csv(test_path)
        print(f"✓ Saved aggregated test: {test_path}")

    return {
        "zones": {
            "train": zone_train_data,
            "test": zone_test_data,
        },
        "aggregated": {
            "train": train_agg,
            "test": test_agg,
        }
    }


def fetch_sweden_zone_observations(
    fetcher: ENTSOEWindDataFetcher,
    train_years: list[int],
    test_year: int,
    output_dir: Path,
    psr_type: Literal["onshore", "offshore", "all"] = "all",
) -> dict:
    """Fetch observations for Swedish bidding zones.

    Args:
        fetcher: ENTSO-E API client.
        train_years: List of training years.
        test_year: Test year.
        output_dir: Output directory.
        psr_type: Wind type.

    Returns:
        Dictionary with zone data and aggregated country data.
    """
    print("\n✓ Sweden detected - fetching bidding zones separately")

    zone_train_data = {}
    zone_test_data = {}

    # Fetch each zone
    for zone_id in SWEDEN_ZONES.keys():
        print(f"\n{'─'*70}")
        print(f"Fetching {zone_id}")
        print(f"{'─'*70}")

        zone_results = fetch_country_observations(
            fetcher, zone_id, train_years, test_year, output_dir, psr_type
        )

        if zone_results:
            zone_train_data[zone_id] = zone_results["train"]
            zone_test_data[zone_id] = zone_results["test"]

    # Aggregate zones for country-level
    print(f"\n{'─'*70}")
    print("Aggregating zones to country-level")
    print(f"{'─'*70}")

    if zone_train_data:
        # Combine all zones with sum
        train_combined = pd.concat([
            df[['generation_mw', 'capacity_mw']] for df in zone_train_data.values()
        ], axis=1)

        train_agg = pd.DataFrame({
            'generation_mw': train_combined.filter(like='generation').sum(axis=1),
            'capacity_mw': train_combined.filter(like='capacity').sum(axis=1),
        })
        train_agg['capacity_factor'] = train_agg['generation_mw'] / train_agg['capacity_mw']
        train_agg['capacity_factor'] = train_agg['capacity_factor'].clip(0, 1.5)

        print(f"  ✓ Aggregated training: {len(train_agg)} data points")
        print(f"    Mean CF: {train_agg['capacity_factor'].mean():.2%}")

    else:
        train_agg = pd.DataFrame()

    if zone_test_data:
        test_combined = pd.concat([
            df[['generation_mw', 'capacity_mw']] for df in zone_test_data.values()
        ], axis=1)

        test_agg = pd.DataFrame({
            'generation_mw': test_combined.filter(like='generation').sum(axis=1),
            'capacity_mw': test_combined.filter(like='capacity').sum(axis=1),
        })
        test_agg['capacity_factor'] = test_agg['generation_mw'] / test_agg['capacity_mw']
        test_agg['capacity_factor'] = test_agg['capacity_factor'].clip(0, 1.5)

        print(f"  ✓ Aggregated test: {len(test_agg)} data points")
        print(f"    Mean CF: {test_agg['capacity_factor'].mean():.2%}")

    else:
        test_agg = pd.DataFrame()

    # Save aggregated country-level data
    obs_dir = output_dir / "observations" / "se"
    obs_dir.mkdir(parents=True, exist_ok=True)

    if not train_agg.empty:
        train_path = obs_dir / f"se_train_{min(train_years)}_{max(train_years)}_aggregated.csv"
        train_agg.to_csv(train_path)
        print(f"\n✓ Saved aggregated training: {train_path}")

    if not test_agg.empty:
        test_path = obs_dir / f"se_test_{test_year}_aggregated.csv"
        test_agg.to_csv(test_path)
        print(f"✓ Saved aggregated test: {test_path}")

    return {
        "zones": {
            "train": zone_train_data,
            "test": zone_test_data,
        },
        "aggregated": {
            "train": train_agg,
            "test": test_agg,
        }
    }


def fetch_country_observations(
    fetcher: ENTSOEWindDataFetcher,
    country: str,
    train_years: list[int],
    test_year: int,
    output_dir: Path,
    psr_type: Literal["onshore", "offshore", "all"] = "all",
) -> dict:
    """Fetch observations for a single country or zone.

    Args:
        fetcher: ENTSO-E API client.
        country: Country or zone code.
        train_years: List of training years.
        test_year: Test year.
        output_dir: Output directory.
        psr_type: Wind type.

    Returns:
        Dictionary with train and test data, or None if no data.
    """
    # Training data
    print(f"\nTraining period: {min(train_years)}-{max(train_years)}")
    train_start = pd.Timestamp(f"{min(train_years)}-01-01", tz="UTC")
    train_end = pd.Timestamp(f"{max(train_years)}-12-31 23:59", tz="UTC")

    train_data = fetcher.calculate_capacity_factor(
        country=country,
        start=train_start,
        end=train_end,
        psr_type=psr_type,
    )

    if train_data.empty:
        print(f"  ✗ No training data for {country}")
        return None

    print(f"  ✓ Training: {len(train_data)} data points")
    print(f"    Mean CF: {train_data['capacity_factor'].mean():.2%}")
    print(f"    Capacity: {train_data['capacity_mw'].mean():.0f} MW")

    # Test data
    print(f"\nTest period: {test_year}")
    test_start = pd.Timestamp(f"{test_year}-01-01", tz="UTC")
    test_end = pd.Timestamp(f"{test_year}-12-31 23:59", tz="UTC")

    test_data = fetcher.calculate_capacity_factor(
        country=country,
        start=test_start,
        end=test_end,
        psr_type=psr_type,
    )

    if test_data.empty:
        print(f"  ✗ No test data for {country}")
        return None

    print(f"  ✓ Test: {len(test_data)} data points")
    print(f"    Mean CF: {test_data['capacity_factor'].mean():.2%}")
    print(f"    Capacity: {test_data['capacity_mw'].mean():.0f} MW")

    # Save training and test data separately
    obs_dir = output_dir / "observations" / country.lower()
    obs_dir.mkdir(parents=True, exist_ok=True)

    train_path = obs_dir / f"{country.lower()}_train_{min(train_years)}_{max(train_years)}.csv"
    train_data.to_csv(train_path)
    print(f"\n✓ Saved training data: {train_path}")

    test_path = obs_dir / f"{country.lower()}_test_{test_year}.csv"
    test_data.to_csv(test_path)
    print(f"✓ Saved test data: {test_path}")

    return {
        "train": train_data,
        "test": test_data,
    }


def generate_pyvwf_config(
    countries: list[str],
    train_years: list[int],
    test_year: int,
    output_dir: Path,
):
    """Generate configuration file for PyVWF workflows.

    Args:
        countries: List of country codes.
        train_years: List of training years.
        test_year: Test year.
        output_dir: Output directory.
    """
    config_path = output_dir / "pyvwf_config.py"

    config_content = f'''"""PyVWF Configuration for Country-Level Workflows.

Generated by: generate_country_level_training_data.py
Training years: {min(train_years)}-{max(train_years)}
Test year: {test_year}

Usage:
    from pyvwf_config import get_config
    config = get_config("NL")

    vwf_model = model.PyVWF(
        "",
        config["country"],
        True,
        calc_z0=config["calc_z0"],
        cluster_mode=config["cluster_mode"],
        cluster_list=config["cluster_list"],
        time_res_list=config["time_res_list"],
        obs_level="country"
    )
"""

from pathlib import Path

# Base directory for country-level data
DATA_DIR = Path(__file__).parent

# Country configurations
CONFIGS = {{
'''

    for country in countries:
        config = COUNTRY_CONFIGS.get(country.upper())
        if config is None:
            continue

        # Special handling for Norway with bidding zones
        if country.upper() == "NO" and config.get("use_bidding_zones", False):
            config_content += f'''    "{country.upper()}": {{
        "country": "{country.upper()}",
        "name": "{config['name']}",
        "calc_z0": True,
        "cluster_mode": "all",
        "cluster_list": [5],  # 5 bidding zones
        "time_res_list": ["fixed"],
        "train_years": {train_years},
        "test_year": {test_year},
        "use_bidding_zones": True,
        "zones": ["NO_1", "NO_2", "NO_3", "NO_4", "NO_5"],
        "grid_points_path": DATA_DIR / "grid_points" / "no" / "no_grid_points_zones.csv",
        "train_obs_path": DATA_DIR / "observations" / "no" / "no_train_{min(train_years)}_{max(train_years)}_aggregated.csv",
        "test_obs_path": DATA_DIR / "observations" / "no" / "no_test_{test_year}_aggregated.csv",
        "cluster_geoms_path": DATA_DIR / "grid_points" / "no" / "no_bidding_zones.geojson",
        "turbine_metadata": {{
            "height": {config['height']},
            "model": "{config['model']}",
            "capacity": {config['capacity']},
        }},
        "note": "Uses bidding zones instead of KMeans clustering",
    }},
'''
        # Special handling for Sweden with bidding zones
        elif country.upper() == "SE" and config.get("use_bidding_zones", False):
            config_content += f'''    "{country.upper()}": {{
        "country": "{country.upper()}",
        "name": "{config['name']}",
        "calc_z0": True,
        "cluster_mode": "all",
        "cluster_list": [4],  # 4 bidding zones
        "time_res_list": ["fixed"],
        "train_years": {train_years},
        "test_year": {test_year},
        "use_bidding_zones": True,
        "zones": ["SE_1", "SE_2", "SE_3", "SE_4"],
        "grid_points_path": DATA_DIR / "grid_points" / "se" / "se_grid_points_zones.csv",
        "train_obs_path": DATA_DIR / "observations" / "se" / "se_train_{min(train_years)}_{max(train_years)}_aggregated.csv",
        "test_obs_path": DATA_DIR / "observations" / "se" / "se_test_{test_year}_aggregated.csv",
        "cluster_geoms_path": DATA_DIR / "grid_points" / "se" / "se_bidding_zones.geojson",
        "turbine_metadata": {{
            "height": {config['height']},
            "model": "{config['model']}",
            "capacity": {config['capacity']},
        }},
        "note": "Uses bidding zones instead of KMeans clustering",
    }},
'''
        else:
            # Standard configuration for NL, FR, BE
            num_clusters = config.get('num_clusters', 5)
            config_content += f'''    "{country.upper()}": {{
        "country": "{country.upper()}",
        "name": "{config['name']}",
        "calc_z0": True,
        "cluster_mode": "all",
        "cluster_list": [{num_clusters}],
        "time_res_list": ["fixed"],
        "train_years": {train_years},
        "test_year": {test_year},
        "grid_points_path": DATA_DIR / "grid_points" / "{country.lower()}" / "{country.lower()}_grid_points.csv",
        "train_obs_path": DATA_DIR / "observations" / "{country.lower()}" / "{country.lower()}_train_{min(train_years)}_{max(train_years)}.csv",
        "test_obs_path": DATA_DIR / "observations" / "{country.lower()}" / "{country.lower()}_test_{test_year}.csv",
        "cluster_geoms_path": DATA_DIR / "grid_points" / "{country.lower()}" / "{country.lower()}_correction_regions.geojson",
        "turbine_metadata": {{
            "height": {config['height']},
            "model": "{config['model']}",
            "capacity": {config['capacity']},
        }},
    }},
'''

    config_content += '''}


def get_config(country: str) -> dict:
    """Get configuration for a country.

    Args:
        country: Country code (NL, FR, BE, NO).

    Returns:
        Configuration dictionary.

    Example:
        >>> config = get_config("NL")
        >>> print(config["num_clusters"])
        5
    """
    country_upper = country.upper()
    if country_upper not in CONFIGS:
        raise ValueError(f"Country {country} not configured. Available: {list(CONFIGS.keys())}")

    return CONFIGS[country_upper]


def get_all_configs() -> dict:
    """Get all country configurations.

    Returns:
        Dictionary mapping country code to configuration.
    """
    return CONFIGS


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        country = sys.argv[1]
        config = get_config(country)
        print(f"Configuration for {config['name']}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        print("Available countries:", list(CONFIGS.keys()))
        print("\\nUsage: python pyvwf_config.py NL")
'''

    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"\n✓ Saved PyVWF config: {config_path}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Generate training and test data for country-level PyVWF workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=["NL", "FR", "BE", "NO", "ES", "SE", "IT", "PT", "IE"],
        choices=["NL", "FR", "BE", "NO", "ES", "SE", "IT", "PT", "IE"],
        help="Countries to process (default: NL FR BE NO ES SE IT PT IE)",
    )
    parser.add_argument(
        "--train-years",
        nargs="+",
        type=int,
        default=[2015, 2016, 2017, 2018],
        help="Training years (default: 2015 2016 2017 2018)",
    )
    parser.add_argument(
        "--test-year",
        type=int,
        default=2019,
        help="Test year (default: 2019)",
    )
    parser.add_argument(
        "--psr-type",
        choices=["onshore", "offshore", "all"],
        default="onshore",
        help="Wind type (default: onshore)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("input/country_level_data"),
        help="Output directory (default: input/country_level_data)",
    )
    parser.add_argument(
        "--skip-observations",
        action="store_true",
        help="Skip fetching ENTSO-E observations (only generate grids)",
    )
    parser.add_argument(
        "--skip-grids",
        action="store_true",
        help="Skip generating grid points (only fetch observations)",
    )

    args = parser.parse_args()

    # Check API key
    if not args.skip_observations and not os.getenv("ENTSOE_API_KEY"):
        print("=" * 70)
        print("ERROR: ENTSOE_API_KEY environment variable not set!")
        print("=" * 70)
        print("\nYou need an ENTSO-E API key to fetch observations.")
        print("\nSteps:")
        print("1. Register at: https://transparency.entsoe.eu/")
        print("2. Generate API key in your account")
        print("3. Set environment variable:")
        print("   export ENTSOE_API_KEY='your-key-here'")
        print("\nAlternatively, use --skip-observations to only generate grid points.")
        return 1

    # Validate years
    if args.test_year in args.train_years:
        print("ERROR: Test year cannot be in training years!")
        return 1

    if max(args.train_years) >= args.test_year:
        print("Warning: Training years should be before test year!")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Country-Level Training Data Generation for PyVWF")
    print("=" * 70)
    print(f"\nCountries: {', '.join(args.countries)}")
    print(f"Training years: {min(args.train_years)}-{max(args.train_years)}")
    print(f"Test year: {args.test_year}")
    print(f"Output directory: {args.output_dir}")
    print(f"Wind type: {args.psr_type}")

    # Generate grid points
    if not args.skip_grids:
        print("\n" + "=" * 70)
        print("STEP 1: Generate Grid Points with Turbine Metadata")
        print("=" * 70)

        for country in args.countries:
            config = COUNTRY_CONFIGS.get(country.upper())
            if config is None:
                print(f"\n✗ No configuration for {country}")
                continue

            try:
                grid_points, cluster_geoms = generate_grid_points(
                    country=country.upper(),
                    config=config,
                    output_dir=args.output_dir,
                    save_geojson=True,
                )
            except Exception as e:
                print(f"\n✗ Error generating grid for {country}: {e}")
                import traceback
                traceback.print_exc()

    # Fetch observations
    if not args.skip_observations:
        print("\n" + "=" * 70)
        print("STEP 2: Fetch Country-Level Observations from ENTSO-E")
        print("=" * 70)

        try:
            fetcher = ENTSOEWindDataFetcher()
            observations = fetch_observations(
                fetcher=fetcher,
                countries=[c.upper() for c in args.countries],
                train_years=args.train_years,
                test_year=args.test_year,
                output_dir=args.output_dir,
                psr_type=args.psr_type,
            )
        except Exception as e:
            print(f"\n✗ Error fetching observations: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Generate PyVWF configuration
    print("\n" + "=" * 70)
    print("STEP 3: Generate PyVWF Configuration File")
    print("=" * 70)

    try:
        generate_pyvwf_config(
            countries=[c.upper() for c in args.countries],
            train_years=args.train_years,
            test_year=args.test_year,
            output_dir=args.output_dir,
        )
    except Exception as e:
        print(f"\n✗ Error generating config: {e}")
        import traceback
        traceback.print_exc()

    # Print summary
    print("\n" + "=" * 70)
    print("✓ DATA GENERATION COMPLETE")
    print("=" * 70)

    print(f"\nGenerated files in: {args.output_dir}")
    print("\nDirectory structure:")
    print(f"  {args.output_dir}/")
    print("    ├── grid_points/")
    for country in args.countries:
        print(f"    │   ├── {country.lower()}/")
        print(f"    │   │   ├── {country.lower()}_grid_points.csv")
        print(f"    │   │   └── {country.lower()}_correction_regions.geojson")

    if not args.skip_observations:
        print("    ├── observations/")
        for country in args.countries:
            print(f"    │   ├── {country.lower()}/")
            print(f"    │   │   ├── {country.lower()}_train_{min(args.train_years)}_{max(args.train_years)}.csv")
            print(f"    │   │   └── {country.lower()}_test_{args.test_year}.csv")

    print("    └── pyvwf_config.py")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)

    print("""
1. Use the grid points for ERA5 simulation:

   from pyvwf_config import get_config
   config = get_config("NL")

   grid_points = pd.read_csv(config["grid_points_path"])
   # Grid points have: lat, lon, ID, height, model, capacity, cluster

2. Load observations for training:

   obs_train = pd.read_csv(config["train_obs_path"], index_col=0, parse_dates=True)
   obs_test = pd.read_csv(config["test_obs_path"], index_col=0, parse_dates=True)

3. Run PyVWF workflow:

   vwf_model = model.PyVWF(
       "",
       config["country"],
       True,
       calc_z0=config["calc_z0"],
       cluster_mode=config["cluster_mode"],
       cluster_list=config["cluster_list"],
       time_res_list=config["time_res_list"],
       obs_level="country"  # ← KEY: Country-level observations!
   )

   vwf_model.train(False)
   vwf_model.simulate_cf(config["test_year"])

4. Visualize correction regions:

   import geopandas as gpd
   cluster_geoms = gpd.read_file(config["cluster_geoms_path"])
   cluster_geoms.plot(column='cluster', cmap='Set3', edgecolor='black')

5. For Norway, consider using bidding zones (NO_1..NO_5) instead of KMeans:

   # Fetch NO zones separately
   python vwf/datasets/fetch_entsoe_capacity_factors.py \\
       --countries NO_1 NO_2 NO_3 NO_4 NO_5 \\
       --year-start 2018 --year-end 2020
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
