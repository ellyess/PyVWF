"""Run Turbine-Level PyVWF Workflow with Cluster Geometry Export.

This script processes countries with turbine-level data and automatically
generates cluster geometries for visualization.

Configuration:
    DE - onshore - 500 clusters
    DK - onshore - 1000 clusters
    DK - offshore - 2 clusters
    UK - onshore - 300 clusters
    UK - offshore - 10 clusters

Prerequisites:
    1. Turbine metadata files in input/turbine_level_data/{country}/
    2. ERA5 reanalysis data available for all countries

Usage:
    # Run all configured countries
    python scripts/run_all_turbine_level.py

    # Run specific countries
    python scripts/run_all_turbine_level.py --configs DE-onshore DK-onshore

    # Dry run (check config without training)
    python scripts/run_all_turbine_level.py --dry-run
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vwf.vwf import PyVWF
from vwf.clustering import cluster_with_geometries
from vwf.export_correction_geodataframes import export_correction_factors_geodataframe
from shapely.geometry import box


# Configuration for each country/cluster mode
TURBINE_LEVEL_CONFIGS = {
    "DE-onshore": {
        "country": "DE",
        "cluster_mode": "onshore",
        "cluster_list": [500],
        "test_year": 2019,
        "metadata_file": "input/turbine_level_data/DE/DE_md.csv",
    },
    "DK-onshore": {
        "country": "DK",
        "cluster_mode": "onshore",
        "cluster_list": [1000],
        "test_year": 2020,
        "metadata_file": "input/turbine_level_data/DK/dk_md.csv",
    },
    "DK-offshore": {
        "country": "DK",
        "cluster_mode": "offshore",
        "cluster_list": [2],
        "test_year": 2020,
        "metadata_file": "input/turbine_level_data/DK/dk_md.csv",
    },
    "UK-onshore": {
        "country": "UK",
        "cluster_mode": "onshore",
        "cluster_list": [300],
        "test_year": 2019,
        "metadata_file": "input/turbine_level_data/UK/uk_md.csv",
    },
    "UK-offshore": {
        "country": "UK",
        "cluster_mode": "offshore",
        "cluster_list": [10],
        "test_year": 2019,
        "metadata_file": "input/turbine_level_data/UK/uk_md.csv",
    },
}


# Country bounding boxes for clipping cluster geometries
COUNTRY_BOUNDS = {
    "DE": box(5.8, 47.2, 15.0, 55.1),  # Germany
    "DK": box(8.0, 54.5, 15.2, 57.8),   # Denmark
    "UK": box(-8.2, 49.8, 2.0, 61.0),   # United Kingdom
}


def load_turbine_metadata(metadata_file: str) -> pd.DataFrame:
    """Load turbine metadata from CSV file.

    Args:
        metadata_file: Path to turbine metadata CSV file.

    Returns:
        DataFrame with turbine locations and metadata.
    """
    # Load metadata file
    df = pd.read_csv(metadata_file)

    # Expected columns: ID, lat, lon, capacity, height, model, type
    # Ensure required columns exist
    required_cols = ['lat', 'lon']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns in {metadata_file}: {missing_cols}")

    return df


def generate_cluster_geometries(
    config_name: str,
    config: dict,
    output_dir: Path,
) -> dict:
    """Generate cluster geometries after training and save as GeoJSON.

    Args:
        config_name: Configuration name (e.g., "DE-onshore").
        config: Configuration dictionary.
        output_dir: Output directory for cluster geometries.

    Returns:
        Dictionary with paths to generated geometry files.
    """
    print(f"\n[{config_name}] Generating cluster geometries...")

    country = config["country"]
    cluster_mode = config["cluster_mode"]
    cluster_list = config["cluster_list"]
    metadata_file = config["metadata_file"]

    try:
        # Load turbine metadata
        turb_info = load_turbine_metadata(metadata_file)

        # Filter by cluster mode if needed
        if cluster_mode == "onshore" and "type" in turb_info.columns:
            turb_info = turb_info[turb_info["type"] == "onshore"]
        elif cluster_mode == "offshore" and "type" in turb_info.columns:
            turb_info = turb_info[turb_info["type"] == "offshore"]

        print(f"  Loaded {len(turb_info)} turbines ({cluster_mode})")

        # Get country bounds for clipping
        country_bounds = COUNTRY_BOUNDS.get(country)
        if country_bounds is None:
            print(f"  Warning: No country bounds defined for {country}, skipping geometry clipping")

        # Generate geometries for each cluster count
        geom_files = {}
        for num_clusters in cluster_list:
            print(f"  Creating {num_clusters} cluster geometries...")

            # Cluster turbines with geometries
            turb_clustered, cluster_geoms = cluster_with_geometries(
                sampling_points=turb_info[['lat', 'lon', 'capacity']].rename(columns={'capacity': 'weight'}),
                num_clusters=num_clusters,
                method="kmeans",
                country_bounds=country_bounds
            )

            if cluster_geoms is None:
                print(f"    Warning: Could not generate cluster geometries (geopandas not installed)")
                continue

            # Save cluster geometries
            geom_output_dir = output_dir / "cluster_geometries" / country.lower()
            geom_output_dir.mkdir(parents=True, exist_ok=True)

            geom_file = geom_output_dir / f"{country.lower()}_{cluster_mode}_correction_regions_{num_clusters}.geojson"
            cluster_geoms.to_file(geom_file, driver='GeoJSON')

            print(f"    ✓ Saved cluster geometries: {geom_file}")
            geom_files[num_clusters] = geom_file

        return geom_files

    except Exception as e:
        print(f"  ✗ Error generating cluster geometries: {e}")
        import traceback
        traceback.print_exc()
        return {}


def export_correction_geodataframes(
    config_name: str,
    config: dict,
    model_output_dir: str,
    cluster_geom_files: dict,
    time_res: str = "fixed",
) -> dict:
    """Export correction factors as GeoDataFrames with cluster geometries.

    Args:
        config_name: Configuration name (e.g., "DE-onshore").
        config: Configuration dictionary.
        model_output_dir: PyVWF model output directory.
        cluster_geom_files: Dictionary mapping cluster count to geometry file path.
        time_res: Temporal resolution (default: "fixed").

    Returns:
        Dictionary with paths to exported GeoDataFrame files.
    """
    print(f"\n[{config_name}] Exporting correction factors as GeoDataFrames...")

    country = config["country"]
    cluster_mode = config["cluster_mode"]

    try:
        exported_files = {}

        for num_clusters, geom_file in cluster_geom_files.items():
            # Find correction factors file
            factors_file = Path(model_output_dir) / "training" / "correction-factors" / f"{country}_factors_{time_res}_{num_clusters}.csv"

            if not factors_file.exists():
                print(f"  Warning: Correction factors not found: {factors_file}")
                continue

            # Export as GeoDataFrame
            output_dir = Path("output/correction_geodataframes_turbine") / country.lower()
            output_file = output_dir / f"{country.lower()}_{cluster_mode}_corrections_{time_res}_{num_clusters}.geojson"

            gdf = export_correction_factors_geodataframe(
                factors_csv=factors_file,
                cluster_geoms_geojson=geom_file,
                output_path=output_file,
                time_slice="1/1" if time_res == "fixed" else None
            )

            exported_files[num_clusters] = output_file
            print(f"  ✓ Exported {len(gdf)} clusters to: {output_file}")

        return exported_files

    except Exception as e:
        print(f"  ✗ Error exporting correction GeoDataFrames: {e}")
        import traceback
        traceback.print_exc()
        return {}


def run_turbine_level_config(
    config_name: str,
    config: dict,
    time_res_list: list[str],
    dry_run: bool = False,
) -> dict:
    """Run complete workflow for a single turbine-level configuration.

    Args:
        config_name: Configuration name (e.g., "DE-onshore").
        config: Configuration dictionary.
        time_res_list: List of temporal resolutions.
        dry_run: If True, only validate config without training.

    Returns:
        Dictionary with results and timing info.
    """
    print("\n" + "=" * 80)
    print(f"Processing {config_name}")
    print("=" * 80)

    start_time = time.time()

    try:
        country = config["country"]
        cluster_mode = config["cluster_mode"]
        cluster_list = config["cluster_list"]
        test_year = config["test_year"]
        metadata_file = config["metadata_file"]

        # Check if metadata file exists
        if not Path(metadata_file).exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        print(f"\n[{config_name}] Configuration:")
        print(f"  Country: {country}")
        print(f"  Cluster mode: {cluster_mode}")
        print(f"  Number of clusters: {cluster_list}")
        print(f"  Test year: {test_year}")
        print(f"  Metadata file: {metadata_file}")

        if dry_run:
            print(f"\n[{config_name}] ✓ Dry run - configuration validated successfully")
            return {
                "config": config_name,
                "status": "dry_run_success",
                "time": time.time() - start_time,
            }

        # Initialize PyVWF model (turbine-level)
        print(f"\n[{config_name}] Initializing PyVWF model...")
        model = PyVWF(
            path="",
            country=country,
            correct=True,
            calc_z0=True,
            cluster_mode=cluster_mode,
            cluster_list=cluster_list,
            time_res_list=time_res_list,
            obs_level="turbine",  # TURBINE-LEVEL
        )

        # Train model
        print(f"\n[{config_name}] Training bias corrections...")
        train_start = time.time()
        model.train(check=False)
        train_time = time.time() - train_start

        # Simulate test year
        print(f"\n[{config_name}] Simulating test year {test_year}...")
        sim_start = time.time()
        model.simulate_cf(test_year)
        sim_time = time.time() - sim_start

        # Generate cluster geometries
        geom_start = time.time()
        cluster_geom_files = generate_cluster_geometries(
            config_name=config_name,
            config=config,
            output_dir=Path("output"),
        )
        geom_time = time.time() - geom_start

        # Export correction factors as GeoDataFrames
        export_start = time.time()
        exported_files = export_correction_geodataframes(
            config_name=config_name,
            config=config,
            model_output_dir=model.directory_path,
            cluster_geom_files=cluster_geom_files,
            time_res=time_res_list[0],  # Use first time resolution
        )
        export_time = time.time() - export_start

        total_time = time.time() - start_time

        print(f"\n[{config_name}] ✓ Completed successfully!")
        print(f"  Training time: {train_time:.2f}s")
        print(f"  Simulation time: {sim_time:.2f}s")
        print(f"  Geometry generation time: {geom_time:.2f}s")
        print(f"  GeoDataFrame export time: {export_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Model output: {model.directory_path}")
        print(f"  Exported GeoDataFrames: {len(exported_files)} files")

        return {
            "config": config_name,
            "status": "success",
            "country": country,
            "cluster_mode": cluster_mode,
            "test_year": test_year,
            "train_time": train_time,
            "sim_time": sim_time,
            "geom_time": geom_time,
            "export_time": export_time,
            "total_time": total_time,
            "output_dir": model.directory_path,
            "exported_geodataframes": list(exported_files.values()),
        }

    except FileNotFoundError as e:
        print(f"\n[{config_name}] ✗ Error: File not found")
        print(f"  {e}")
        return {
            "config": config_name,
            "status": "file_not_found",
            "error": str(e),
            "time": time.time() - start_time,
        }

    except Exception as e:
        print(f"\n[{config_name}] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "config": config_name,
            "status": "error",
            "error": str(e),
            "time": time.time() - start_time,
        }


def print_summary(results: list[dict]):
    """Print summary of all results.

    Args:
        results: List of result dictionaries from run_turbine_level_config().
    """
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successes = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]
    file_errors = [r for r in results if r["status"] == "file_not_found"]
    dry_runs = [r for r in results if r["status"] == "dry_run_success"]

    print(f"\nTotal configurations processed: {len(results)}")
    print(f"  ✓ Successful: {len(successes)}")
    print(f"  ✗ Errors: {len(errors)}")
    print(f"  ⚠ File not found: {len(file_errors)}")
    if dry_runs:
        print(f"  ℹ Dry runs: {len(dry_runs)}")

    if successes:
        print("\n" + "-" * 120)
        print("Successful Configurations:")
        print("-" * 120)
        print(f"{'Config':<20} {'Country':<8} {'Mode':<10} {'Year':<6} {'Train':<10} {'Sim':<10} {'Geom':<10} {'Export':<10} {'Total':<10}")
        print("-" * 120)
        for r in successes:
            print(
                f"{r['config']:<20} "
                f"{r['country']:<8} "
                f"{r['cluster_mode']:<10} "
                f"{r['test_year']:<6} "
                f"{r['train_time']:>8.2f}s  "
                f"{r['sim_time']:>8.2f}s  "
                f"{r['geom_time']:>8.2f}s  "
                f"{r['export_time']:>8.2f}s  "
                f"{r['total_time']:>8.2f}s"
            )

        total_train_time = sum(r["train_time"] for r in successes)
        total_sim_time = sum(r["sim_time"] for r in successes)
        total_geom_time = sum(r["geom_time"] for r in successes)
        total_export_time = sum(r["export_time"] for r in successes)
        total_time = sum(r["total_time"] for r in successes)

        print("-" * 120)
        print(
            f"{'TOTAL':<20} {'':8} {'':10} {'':6} "
            f"{total_train_time:>8.2f}s  "
            f"{total_sim_time:>8.2f}s  "
            f"{total_geom_time:>8.2f}s  "
            f"{total_export_time:>8.2f}s  "
            f"{total_time:>8.2f}s"
        )

    if errors:
        print("\n" + "-" * 80)
        print("Errors:")
        print("-" * 80)
        for r in errors:
            print(f"  {r['config']}: {r['error']}")

    if file_errors:
        print("\n" + "-" * 80)
        print("File Not Found:")
        print("-" * 80)
        for r in file_errors:
            print(f"  {r['config']}: {r['error']}")

    # Output directories and GeoDataFrames
    if successes:
        print("\n" + "-" * 80)
        print("Output Directories:")
        print("-" * 80)
        for r in successes:
            print(f"  {r['config']}: {r['output_dir']}")
            if r.get('exported_geodataframes'):
                print(f"    GeoDataFrames: {len(r['exported_geodataframes'])} files")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Run turbine-level PyVWF workflow with cluster geometry export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(TURBINE_LEVEL_CONFIGS.keys()),
        choices=list(TURBINE_LEVEL_CONFIGS.keys()),
        help="Configurations to process (default: all)",
    )
    parser.add_argument(
        "--time-res",
        nargs="+",
        default=["fixed"],
        choices=["fixed", "season", "bimonth", "month"],
        help="Temporal resolutions (default: fixed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate configurations without training",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue processing other configs if one fails (default: True)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Turbine-Level PyVWF Workflow - Batch Processing")
    print("=" * 80)
    print(f"\nConfigurations: {', '.join(args.configs)}")
    print(f"Time resolutions: {', '.join(args.time_res)}")
    if args.dry_run:
        print("\nℹ DRY RUN MODE - No training will be performed")

    results = []
    total_start = time.time()

    for config_name in args.configs:
        config = TURBINE_LEVEL_CONFIGS[config_name]
        result = run_turbine_level_config(
            config_name=config_name,
            config=config,
            time_res_list=args.time_res,
            dry_run=args.dry_run,
        )
        results.append(result)

        # Stop if error and continue_on_error is False
        if result["status"] == "error" and not args.continue_on_error:
            print(f"\n✗ Stopping due to error in {config_name}")
            break

    total_time = time.time() - total_start

    # Print summary
    print_summary(results)

    print("\n" + "=" * 80)
    print(f"Total execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print("=" * 80)

    # Exit code based on results
    if any(r["status"] == "error" for r in results):
        return 1
    if any(r["status"] == "file_not_found" for r in results):
        print("\n⚠ Some metadata files were not found.")
        return 2

    print("\n✓ All configurations processed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
