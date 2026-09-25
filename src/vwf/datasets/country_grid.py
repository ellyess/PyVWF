"""Grid points for the country-level regions: where the sampling points are.

A country-level region has one observed series, national or per bidding zone,
so its fleet is a set of grid points rather than real turbines. This module
builds them: a regular grid inside the country's box, clustered with k-means
into Voronoi regions, or, for Norway and Sweden, one grid per bidding zone
with the zone as its cluster. Every point carries the same representative
turbine, named by its key in the licensed curve library (the open library has
none of the three, and a country run refuses a missing curve);
``scripts/region_tools/weight_country_grid_points.py`` replaces the uniform
capacities with real ones from the Global Wind Power Tracker.

Split from ``generate_country_level_training_data.py``, whose ``main`` runs it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from shapely.geometry import box

from vwf.clustering import cluster_with_geometries, create_sampling_points


# Country configurations
COUNTRY_CONFIGS = {
    "NL": {
        "name": "Netherlands",
        "bounds": box(3.3, 50.7, 7.2, 53.6),
        "height": 100.0,  # Modern onshore fleet
        "model": "Vestas.V90.3000",  # 3 MW turbine
        "capacity": 3.0,  # Average capacity
        "grid_resolution": 0.25,  # ~25km grid
        "num_clusters": 5,  # Spatial regions
    },
    "FR": {
        "name": "France",
        "bounds": box(-5.0, 42.0, 8.5, 51.2),
        "height": 90.0,  # Mix of old and modern
        "model": "Vestas.V80.2000",  # 2 MW turbine
        "capacity": 2.5,
        "grid_resolution": 0.5,  # ~50km grid (larger country)
        "num_clusters": 10,  # More regions
    },
    "BE": {
        "name": "Belgium",
        "bounds": box(2.5, 49.5, 6.4, 51.5),
        "height": 100.0,  # Modern fleet
        "model": "Vestas.V90.3000",  # 3 MW turbine
        "capacity": 3.0,
        "grid_resolution": 0.25,
        "num_clusters": 3,  # Smaller country
    },
    "NO": {
        "name": "Norway",
        "bounds": box(4.5, 58.0, 31.0, 71.5),  # Full country (for reference)
        "height": 80.0,  # Mountain terrain, lower heights
        "model": "Vestas.V90.3000",  # 3 MW turbine
        "capacity": 3.0,
        "grid_resolution": 1.0,  # ~100km grid (large country, sparse turbines)
        "use_bidding_zones": True,  # ← Use zones instead of KMeans
        "note": "Norway uses bidding zones (NO_1..NO_5) for market structure",
    },
    # Phase 1 Countries (moderate clusters with bbox optimization)
    "ES": {
        "name": "Spain",
        "bounds": box(-9.5, 36.0, 3.5, 43.8),
        "height": 90.0,  # Modern fleet
        "model": "Vestas.V90.2000",  # 2 MW turbine
        "capacity": 2.5,
        "grid_resolution": 1.0,  # ~100km grid (large country)
        "num_clusters": 4,  # Increased with bbox optimization
    },
    "SE": {
        "name": "Sweden",
        "bounds": box(11.0, 55.3, 24.2, 69.0),  # Full country (for reference)
        "height": 100.0,  # Modern, tall turbines
        "model": "Vestas.V90.3000",  # 3 MW turbine
        "capacity": 3.0,
        "grid_resolution": 1.5,  # ~150km grid (very large country)
        "use_bidding_zones": True,  # ← Use zones instead of KMeans
        "note": "Sweden uses bidding zones (SE_1..SE_4) for market structure",
    },
    "IT": {
        "name": "Italy",
        "bounds": box(6.6, 36.6, 18.5, 47.1),
        "height": 80.0,  # Mix of old and modern
        "model": "Vestas.V80.2000",  # 2 MW turbine
        "capacity": 2.0,
        "grid_resolution": 1.0,  # ~100km grid
        "num_clusters": 3,  # Increased with bbox optimization
    },
    "PT": {
        "name": "Portugal",
        "bounds": box(-9.5, 37.0, -6.2, 42.2),
        "height": 80.0,
        "model": "Vestas.V80.2000",  # 2 MW turbine
        "capacity": 2.5,
        "grid_resolution": 0.5,  # ~50km grid
        "num_clusters": 3,  # Increased with bbox optimization
    },
    "IE": {
        "name": "Ireland",
        "bounds": box(-10.5, 51.4, -5.4, 55.4),
        "height": 85.0,
        "model": "Vestas.V90.2000",  # 2 MW turbine
        "capacity": 2.5,
        "grid_resolution": 0.5,  # ~50km grid
        "num_clusters": 3,  # Increased with bbox optimization
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
    print(f"\n{'=' * 70}")
    print(f"Generating Grid Points for {config['name']} ({country})")
    print(f"{'=' * 70}")

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

    zones_path = Path("input/reference/shapes/no_bidding_zones.geojson")
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
        "NO 1": "NO_1",
        "NO 2": "NO_2",
        "NO 3": "NO_3",
        "NO 4": "NO_4",
        "NO 5": "NO_5",
    }

    for idx, zone_row in zones_gdf.iterrows():
        zone_name_orig = zone_row["Price area"]
        zone_id = zone_name_map.get(zone_name_orig, zone_name_orig.replace(" ", "_"))
        zone_geom = zone_row.geometry

        # Extract zone number (NO_1 -> 0, NO_2 -> 1, etc.)
        zone_num = int(zone_id.split("_")[1]) - 1

        # Get grid resolution from NORWAY_ZONES config
        zone_config = NORWAY_ZONES.get(zone_id, {"grid_resolution": 0.5})
        grid_resolution = zone_config.get("grid_resolution", 0.5)

        print(f"{'─' * 70}")
        print(f"Zone {zone_id} ({zone_name_orig})")
        print(f"{'─' * 70}")
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
            default_height=config["height"],
            default_model=config["model"],
            default_capacity=config["capacity"],
        )

        # Filter grid points to only those inside the actual zone geometry
        # Create point geometries for filtering
        from shapely.geometry import Point as ShapelyPoint

        zone_grid_gdf = gpd.GeoDataFrame(
            zone_grid,
            geometry=[
                ShapelyPoint(lon, lat) for lon, lat in zip(zone_grid["lon"], zone_grid["lat"])
            ],
            crs="EPSG:4326",
        )

        # Keep only points inside the actual zone geometry
        zone_grid_filtered = zone_grid_gdf[zone_grid_gdf.geometry.within(zone_geom)].copy()
        zone_grid_filtered.drop(columns=["geometry"], inplace=True)

        # Assign zone as cluster
        zone_grid_filtered["cluster"] = zone_num
        zone_grid_filtered["zone"] = zone_id

        print(
            f"  ✓ Created {len(zone_grid_filtered)} grid points for {zone_id} (filtered from {len(zone_grid)})"
        )

        all_zones_grids.append(zone_grid_filtered)

        # Use actual zone geometry
        zone_geometries.append(
            {
                "cluster": zone_num,
                "zone": zone_id,
                "name": zone_config.get("name", zone_name_orig),
                "geometry": zone_geom,
                "n_points": len(zone_grid_filtered),
            }
        )

    # Combine all zones
    grid_all_zones = pd.concat(all_zones_grids, ignore_index=True)

    print(f"\n{'─' * 70}")
    print(f"✓ Combined grid: {len(grid_all_zones)} points across {len(zone_geometries)} zones")
    print("\nZone distribution:")
    for zone_id in sorted(grid_all_zones["zone"].unique()):
        count = len(grid_all_zones[grid_all_zones["zone"] == zone_id])
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
            print("Warning: GeoPandas not available - skipping GeoJSON export")
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

    zones_path = Path("input/reference/shapes/se_bidding_zones.geojson")
    if not zones_path.exists():
        print(f"  ✗ Bidding zones file not found: {zones_path}")
        print("  Warning: Cannot proceed without zone geometries")
        raise FileNotFoundError(f"Swedish bidding zones file not found: {zones_path}")

    zones_gdf = gpd.read_file(zones_path)
    print(f"  ✓ Loaded {len(zones_gdf)} bidding zones from: {zones_path}")

    all_zones_grids = []
    zone_geometries = []

    for idx, zone_row in zones_gdf.iterrows():
        # Get zone_id from properties (already in SE_1 format from our extraction)
        zone_id = zone_row.get("zone_id") or zone_row.get("zone_name")
        if not zone_id:
            print(f"  Warning: Missing zone_id for feature {idx}, skipping")
            continue

        zone_geom = zone_row.geometry

        # Extract zone number (SE_1 -> 0, SE_2 -> 1, etc.)
        zone_num = int(zone_id.split("_")[1]) - 1

        # Get grid resolution from SWEDEN_ZONES config
        zone_config = SWEDEN_ZONES.get(zone_id, {"grid_resolution": 0.75})
        grid_resolution = zone_config.get("grid_resolution", 0.75)

        print(f"{'─' * 70}")
        print(f"Zone {zone_id}")
        print(f"{'─' * 70}")
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
            default_height=config["height"],
            default_model=config["model"],
            default_capacity=config["capacity"],
        )

        # Filter grid points to only those inside the actual zone geometry
        # Create point geometries for filtering
        from shapely.geometry import Point as ShapelyPoint

        zone_grid_gdf = gpd.GeoDataFrame(
            zone_grid,
            geometry=[
                ShapelyPoint(lon, lat) for lon, lat in zip(zone_grid["lon"], zone_grid["lat"])
            ],
            crs="EPSG:4326",
        )

        # Keep only points inside the actual zone geometry
        zone_grid_filtered = zone_grid_gdf[zone_grid_gdf.geometry.within(zone_geom)].copy()
        zone_grid_filtered.drop(columns=["geometry"], inplace=True)

        # Assign zone as cluster
        zone_grid_filtered["cluster"] = zone_num
        zone_grid_filtered["zone"] = zone_id

        print(f"  Grid points: {len(zone_grid_filtered)}")

        if len(zone_grid_filtered) == 0:
            print(f"  Warning: No grid points generated for {zone_id}")
            continue

        all_zones_grids.append(zone_grid_filtered)

        # Use actual zone geometry
        zone_geometries.append(
            {
                "cluster": zone_num,
                "zone": zone_id,
                "name": zone_config.get("name", zone_id),
                "geometry": zone_geom,
                "n_points": len(zone_grid_filtered),
            }
        )

    # Combine all zones
    grid_all_zones = pd.concat(all_zones_grids, ignore_index=True)

    print(f"\n{'─' * 70}")
    print(f"✓ Combined grid: {len(grid_all_zones)} points across {len(zone_geometries)} zones")
    print("\nZone distribution:")
    for zone_id in sorted(grid_all_zones["zone"].unique()):
        count = len(grid_all_zones[grid_all_zones["zone"] == zone_id])
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
            print("Warning: GeoPandas not available - skipping GeoJSON export")
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
        print(f"{'─' * 70}")
        print(f"Zone {zone_id}: {zone_config['name']}")
        print(f"{'─' * 70}")

        # Extract zone number (NO_1 -> 0, NO_2 -> 1, etc.)
        zone_num = int(zone_id.split("_")[1]) - 1

        print(f"  Grid resolution: {zone_config['grid_resolution']}°")
        print(f"  Turbine metadata: {config['height']}m, {config['model']}, {config['capacity']}MW")

        # Create grid for this zone
        zone_grid = create_sampling_points(
            country_bounds=zone_config["bounds"],
            method="grid",
            resolution=zone_config["grid_resolution"],
            add_metadata=True,
            default_height=config["height"],
            default_model=config["model"],
            default_capacity=config["capacity"],
        )

        # Assign zone as cluster
        zone_grid["cluster"] = zone_num
        zone_grid["zone"] = zone_id

        print(f"  ✓ Created {len(zone_grid)} grid points for {zone_id}")

        all_zones_grids.append(zone_grid)

        # Create zone geometry (bounding box)
        zone_geometries.append(
            {
                "cluster": zone_num,
                "zone": zone_id,
                "name": zone_config["name"],
                "geometry": zone_config["bounds"],
                "n_points": len(zone_grid),
            }
        )

    # Combine all zones
    grid_all_zones = pd.concat(all_zones_grids, ignore_index=True)

    print(f"\n{'─' * 70}")
    print(f"✓ Combined grid: {len(grid_all_zones)} points across {len(NORWAY_ZONES)} zones")
    print("\nZone distribution:")
    for zone_id in sorted(grid_all_zones["zone"].unique()):
        count = len(grid_all_zones[grid_all_zones["zone"] == zone_id])
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
            print("Warning: GeoPandas not available - skipping GeoJSON export")
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
    print(
        f"\nGrid resolution: {config['grid_resolution']}° (~{config['grid_resolution'] * 100:.0f} km)"
    )
    print("Representative turbine:")
    print(f"  Hub height: {config['height']} m")
    print(f"  Power curve: {config['model']}")
    print(f"  Capacity: {config['capacity']} MW")

    grid_points = create_sampling_points(
        country_bounds=config["bounds"],
        method="grid",
        resolution=config["grid_resolution"],
        add_metadata=True,
        default_height=config["height"],
        default_model=config["model"],
        default_capacity=config["capacity"],
    )

    print(f"\n✓ Created {len(grid_points)} grid points")

    # Cluster spatially with Voronoi tessellation
    print(f"\nClustering into {config['num_clusters']} regions (Voronoi)...")
    grid_clustered, cluster_geoms = cluster_with_geometries(
        sampling_points=grid_points,
        num_clusters=config["num_clusters"],
        method="kmeans",
        country_code=country,
        cluster_mode="onshore",
        geometry_type="voronoi",
    )

    # Print cluster distribution
    cluster_counts = grid_clustered["cluster"].value_counts().sort_index()
    print("\nCluster distribution:")
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
    print("\nGrid points ready for PyVWF simulation!")
    print(f"Columns: {list(grid_clustered.columns)}")
    print(
        "\nNOTE: every point carries the same synthetic capacity, so the "
        "simulated country aggregate is weighted by land area while the "
        "observation is weighted by installed capacity. Run\n"
        f"  python scripts/region_tools/weight_country_grid_points.py {country}\n"
        "to replace the uniform capacities with real ones from the Global Wind "
        "Power Tracker. Regenerating this file overwrites that weighting."
    )

    return grid_clustered, cluster_geoms
