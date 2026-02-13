"""Export correction factors as GeoDataFrames with cluster geometries.

This module provides functions to export PyVWF correction factors as
spatial data (GeoDataFrame) with cluster geometries for visualization
and spatial analysis.
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path


def export_correction_factors_geodataframe(
    factors_csv: str | Path,
    cluster_geoms_geojson: str | Path,
    output_path: str | Path,
    time_slice: str | None = None
) -> gpd.GeoDataFrame:
    """Export correction factors as GeoDataFrame with cluster geometries.

    Args:
        factors_csv: Path to correction factors CSV file
            (e.g., "NL_factors_fixed_5.csv")
        cluster_geoms_geojson: Path to cluster geometries GeoJSON file
            (e.g., "nl_correction_regions.geojson")
        output_path: Output path for GeoDataFrame (GeoJSON, GeoPackage, or Shapefile)
        time_slice: Optional time slice filter (e.g., "1/1", "winter")
            If None, includes all time slices

    Returns:
        GeoDataFrame with correction factors and cluster geometries

    Example:
        >>> gdf = export_correction_factors_geodataframe(
        ...     "output/run/NL-all/training/correction-factors/NL_factors_fixed_5.csv",
        ...     "input/country_level_data/grid_points/nl/nl_correction_regions.geojson",
        ...     "output/nl_corrections_spatial.geojson"
        ... )
    """
    # Load correction factors
    factors = pd.read_csv(factors_csv)

    # Load cluster geometries
    cluster_geoms = gpd.read_file(cluster_geoms_geojson)

    # Filter by time slice if specified
    if time_slice is not None:
        # Get time column name (could be 'fixed', 'season', 'bimonth', etc.)
        time_cols = [c for c in factors.columns if c in ['fixed', 'season', 'bimonth', 'month']]
        if time_cols:
            time_col = time_cols[0]
            factors = factors[factors[time_col] == time_slice]

    # Merge factors with geometries on cluster ID
    # Ensure cluster is same type in both dataframes
    factors['cluster'] = factors['cluster'].astype(int)
    cluster_geoms['cluster'] = cluster_geoms['cluster'].astype(int)

    gdf = cluster_geoms.merge(factors, on='cluster', how='left')

    # Save to output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == '.geojson':
        gdf.to_file(output_path, driver='GeoJSON')
    elif output_path.suffix == '.gpkg':
        gdf.to_file(output_path, driver='GPKG')
    elif output_path.suffix == '.shp':
        gdf.to_file(output_path)
    else:
        # Default to GeoJSON
        gdf.to_file(output_path, driver='GeoJSON')

    print(f"✓ Exported {len(gdf)} clusters with correction factors to: {output_path}")

    return gdf


def export_all_country_corrections(
    country: str,
    output_dir: str | Path = "output/correction_geodataframes",
    config_dir: str | Path = "input/country_level_data"
) -> dict[str, gpd.GeoDataFrame]:
    """Export all correction factors for a country as GeoDataFrames.

    Args:
        country: Country code (e.g., "NL", "FR", "BE", "NO")
        output_dir: Directory to save GeoDataFrame outputs
        config_dir: Configuration directory with grid_points and geometries

    Returns:
        Dictionary mapping time_res to GeoDataFrame

    Example:
        >>> gdfs = export_all_country_corrections("NL")
        >>> # Access specific time resolution
        >>> fixed_gdf = gdfs['fixed']
    """
    country_lower = country.lower()
    output_dir = Path(output_dir)
    config_dir = Path(config_dir)

    # Find output directory for this country
    run_dirs = list(Path("output/run").glob(f"{country}-*"))
    if not run_dirs:
        raise ValueError(f"No run directory found for country {country}")

    run_dir = run_dirs[0]
    factors_dir = run_dir / "training" / "correction-factors"

    # Find cluster geometries
    if country == "NO":
        geom_file = config_dir / "grid_points" / country_lower / f"{country_lower}_bidding_zones.geojson"
    else:
        geom_file = config_dir / "grid_points" / country_lower / f"{country_lower}_correction_regions.geojson"

    if not geom_file.exists():
        raise FileNotFoundError(f"Cluster geometries not found: {geom_file}")

    # Find all correction factor files
    factor_files = list(factors_dir.glob(f"{country}_factors_*.csv"))

    if not factor_files:
        raise ValueError(f"No correction factor files found in {factors_dir}")

    # Export each
    gdfs = {}
    for factor_file in factor_files:
        # Extract time_res and n_clusters from filename
        # Format: {country}_factors_{time_res}_{n_clusters}.csv
        parts = factor_file.stem.split('_')
        time_res = parts[2]
        n_clusters = parts[3]

        output_file = output_dir / country_lower / f"{country_lower}_corrections_{time_res}_{n_clusters}.geojson"

        gdf = export_correction_factors_geodataframe(
            factors_csv=factor_file,
            cluster_geoms_geojson=geom_file,
            output_path=output_file,
            time_slice="1/1" if time_res == "fixed" else None
        )

        gdfs[f"{time_res}_{n_clusters}"] = gdf

    return gdfs


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python export_correction_geodataframes.py COUNTRY [OUTPUT_DIR]")
        print("Example: python export_correction_geodataframes.py NL")
        sys.exit(1)

    country = sys.argv[1].upper()
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output/correction_geodataframes"

    print(f"Exporting correction factors for {country} as GeoDataFrames...")
    gdfs = export_all_country_corrections(country, output_dir)

    print(f"\n✓ Exported {len(gdfs)} correction factor geodataframes:")
    for key, gdf in gdfs.items():
        print(f"  - {key}: {len(gdf)} clusters")
