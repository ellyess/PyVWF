"""Generate cluster geometries and GeoDataFrames from existing turbine-level runs.

This script post-processes existing turbine-level PyVWF runs to generate:
1. Cluster geometries (Voronoi tessellation clipped to country/offshore boundaries)
2. Correction factor GeoDataFrames

For countries: DE, DK (onshore/offshore), UK (onshore/offshore)

Cluster geometries are clipped to actual country shapes from:
- input/regions/country_shapes.geojson (onshore)
- input/regions/offshore_shapes.geojson (offshore)

Uses Voronoi tessellation for complete spatial coverage with no gaps between clusters.
"""

import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPoint, box
import warnings

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from vwf.clustering import cluster_with_geometries
from vwf.export_correction_geodataframes import export_correction_factors_geodataframe


# Configurations
CONFIGS = {
    "DE-onshore": {
        "country": "DE",
        "cluster_mode": "onshore",
        "n_clusters": 500,
        "run_dir": "output/run/DE-onshore-obs_turbine-corrected-calc_z0",
    },
    "DK-onshore": {
        "country": "DK",
        "cluster_mode": "onshore",
        "n_clusters": 1000,
        "run_dir": "output/run/DK-onshore-obs_turbine-corrected-calc_z0",
    },
    "DK-offshore": {
        "country": "DK",
        "cluster_mode": "offshore",
        "n_clusters": 2,
        "run_dir": "output/run/DK-offshore-obs_turbine-corrected-calc_z0",
    },
    "UK-onshore": {
        "country": "UK",
        "cluster_mode": "onshore",
        "n_clusters": 300,
        "run_dir": "output/run/UK-onshore-obs_turbine-corrected-calc_z0",
    },
    "UK-offshore": {
        "country": "UK",
        "cluster_mode": "offshore",
        "n_clusters": 10,
        "run_dir": "output/run/UK-offshore-obs_turbine-corrected-calc_z0",
    },
}


def generate_cluster_geometries_from_turb_info(config_name: str, config: dict):
    """Generate cluster geometries from existing turbine info file.

    Args:
        config_name: Configuration name.
        config: Configuration dictionary.
    """
    print(f"\n{'='*80}")
    print(f"Processing {config_name}")
    print(f"{'='*80}")

    country = config["country"]
    cluster_mode = config["cluster_mode"]
    n_clusters = config["n_clusters"]
    run_dir = Path(config["run_dir"])

    # Load turbine info (this has the actual turbine locations and cluster assignments)
    turb_info_files = list((run_dir / "training" / "simulated-turbines").glob("*_turb_info.csv"))
    if not turb_info_files:
        print(f"  ✗ No turbine info files found in {run_dir}")
        return False

    # Use the training turb_info
    train_turb_info = [f for f in turb_info_files if "train" in f.name]
    if train_turb_info:
        turb_info_path = train_turb_info[0]
    else:
        turb_info_path = turb_info_files[0]

    print(f"  Loading turbine info: {turb_info_path.name}")
    turb_info = pd.read_csv(turb_info_path)

    # Check required columns
    if 'lat' not in turb_info.columns or 'lon' not in turb_info.columns:
        print(f"  ✗ Missing lat/lon columns in {turb_info_path}")
        return False

    print(f"  Loaded {len(turb_info)} turbines")

    # Generate cluster geometries
    # Will use actual country/offshore shapes for clipping
    print(f"  Clustering {len(turb_info)} turbines into {n_clusters} clusters...")

    # Prepare data for clustering function
    sampling_points = turb_info[['lat', 'lon']].copy()
    sampling_points['weight'] = turb_info.get('capacity', 1.0)

    # Perform KMeans clustering with actual country shape clipping
    # This should reproduce the same clusters used during training (random_state=42)
    # Uses Voronoi tessellation for complete spatial coverage with no gaps
    sampling_points_clustered, cluster_geoms = cluster_with_geometries(
        sampling_points=sampling_points,
        num_clusters=n_clusters,
        method="kmeans",
        country_code=country,
        cluster_mode=cluster_mode,
        geometry_type="voronoi"
    )

    print(f"  ✓ Clustered into {sampling_points_clustered['cluster'].nunique()} unique clusters")

    if cluster_geoms is None:
        print(f"  ✗ Failed to generate cluster geometries")
        return False

    print(f"  ✓ Generated {len(cluster_geoms)} cluster geometries")

    # Save cluster geometries
    geom_output_dir = Path("output/cluster_geometries") / country.lower()
    geom_output_dir.mkdir(parents=True, exist_ok=True)

    geom_file = geom_output_dir / f"{country.lower()}_{cluster_mode}_correction_regions_{n_clusters}.geojson"
    cluster_geoms.to_file(geom_file, driver='GeoJSON')
    print(f"  ✓ Saved cluster geometries: {geom_file}")

    # Export correction factors as GeoDataFrame
    print(f"  Exporting correction GeoDataFrame...")

    # Find correction factors file
    factors_file = run_dir / "training" / "correction-factors" / f"{country}_factors_fixed_{n_clusters}.csv"

    if not factors_file.exists():
        print(f"  ✗ Correction factors not found: {factors_file}")
        return False

    # Export as GeoDataFrame
    output_dir = Path("output/correction_geodataframes_turbine") / country.lower()
    output_file = output_dir / f"{country.lower()}_{cluster_mode}_corrections_fixed_{n_clusters}.geojson"

    try:
        gdf = export_correction_factors_geodataframe(
            factors_csv=factors_file,
            cluster_geoms_geojson=geom_file,
            output_path=output_file,
            time_slice="1/1"
        )
        print(f"  ✓ Exported {len(gdf)} clusters to: {output_file}")
        return True
    except Exception as e:
        print(f"  ✗ Error exporting GeoDataFrame: {e}")
        return False


def main():
    """Main execution."""
    print("="*80)
    print("Generating Cluster Geometries from Existing Turbine-Level Runs")
    print("="*80)

    results = {}
    for config_name, config in CONFIGS.items():
        success = generate_cluster_geometries_from_turb_info(config_name, config)
        results[config_name] = success

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    successful = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]

    print(f"\n✓ Successful: {len(successful)}")
    for config in successful:
        print(f"  - {config}")

    if failed:
        print(f"\n✗ Failed: {len(failed)}")
        for config in failed:
            print(f"  - {config}")

    print("\n" + "="*80)
    print(f"Total: {len(successful)}/{len(CONFIGS)} configurations processed successfully")
    print("="*80)


if __name__ == "__main__":
    main()
