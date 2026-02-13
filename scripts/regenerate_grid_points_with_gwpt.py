"""Regenerate grid points with realistic capacity distribution from Global Wind Power Tracker.

This script:
1. Loads wind farm data from Global Wind Power Tracker Excel file
2. For each grid point, calculates nearby wind farm capacity within a search radius
3. Updates grid points with realistic capacity weights (optionally filtered by year)
4. Maintains grid point structure (lat, lon, height, model, etc.)

Usage:
    # Generate grid points with all current farms
    python scripts/regenerate_grid_points_with_gwpt.py --country NL --radius 50
    
    # Generate year-specific grid points (only farms operational in that year)
    python scripts/regenerate_grid_points_with_gwpt.py --country NL --year 2019 --radius 50
    python scripts/regenerate_grid_points_with_gwpt.py --country FR --year 2020 --radius 75
    
    # Use custom data file
    python scripts/regenerate_grid_points_with_gwpt.py --country NL --data /path/to/tracker.xlsx
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


# Country name mappings for Global Wind Power Tracker data
COUNTRY_NAMES = {
    "NL": "Netherlands",
    "FR": "France",
    "BE": "Belgium",
    "DE": "Germany",
    "DK": "Denmark",
    "UK": "United Kingdom",
    "SE": "Sweden",
    "NO": "Norway",
    "ES": "Spain",
    "PT": "Portugal",
    "IT": "Italy",
    "IE": "Ireland",
}


def load_wind_farms(country: str, data_file: Path, year: Optional[int] = None) -> pd.DataFrame:
    """Load wind farm data from Global Wind Power Tracker, filtered by operational status in year.

    Args:
        country: Country code (e.g., 'NL', 'FR')
        data_file: Path to Global Wind Power Tracker Excel file
        year: Year to filter for (only include farms commissioned by this year and not retired).
              If None, includes all current farms.

    Returns:
        DataFrame with wind farm locations and capacity
    """
    if not data_file.exists():
        print(f"✗ Data file not found: {data_file}")
        return pd.DataFrame()

    country_name = COUNTRY_NAMES.get(country.upper())
    if not country_name:
        print(f"✗ Unknown country code: {country}")
        return pd.DataFrame()

    print(f"Loading Global Wind Power Tracker data for {country}...")
    
    try:
        df = pd.read_excel(data_file, sheet_name="Data")
    except Exception as e:
        print(f"✗ Error reading Excel file: {e}")
        return pd.DataFrame()

    # Filter for country
    country_data = df[df["Country/Area"] == country_name].copy()

    if country_data.empty:
        print(f"⚠ No wind farms found for {country_name}")
        return pd.DataFrame()

    # Extract relevant columns
    wind_farms = country_data[["Latitude", "Longitude", "Capacity (MW)", "Start year", "Retired year"]].copy()
    wind_farms.columns = ["lat", "lon", "capacity", "start_year", "retired_year"]

    # Filter out rows with missing location or capacity
    wind_farms = wind_farms.dropna(subset=["lat", "lon", "capacity"])

    # Filter by year if provided
    if year is not None:
        print(f"  Filtering for operational status in year {year}...")
        
        # Include farms that were commissioned by this year and not retired
        operational_mask = (wind_farms["start_year"].isna() | (wind_farms["start_year"] <= year)) & \
                          (wind_farms["retired_year"].isna() | (wind_farms["retired_year"] > year))
        
        wind_farms = wind_farms[operational_mask].copy()

    print(f"✓ Found {len(wind_farms)} wind farms")
    if wind_farms.empty:
        return pd.DataFrame()
    
    print(f"  Total capacity: {wind_farms['capacity'].sum():.0f} MW")
    if year is not None:
        print(f"  (as of {year})")

    # Keep only location and capacity columns for weight calculation
    return wind_farms[["lat", "lon", "capacity"]]


def calculate_grid_capacity_weights(
    grid_points: pd.DataFrame,
    wind_farms: pd.DataFrame,
    search_radius_km: float = 50.0
) -> pd.Series:
    """Calculate capacity weight for each grid point based on nearby wind farms.

    Args:
        grid_points: DataFrame with grid point locations (lat, lon)
        wind_farms: DataFrame with wind farm locations and capacity
        search_radius_km: Search radius in kilometers

    Returns:
        Series with capacity weight for each grid point
    """
    if wind_farms.empty:
        print("⚠ No wind farms available; returning uniform weights")
        return pd.Series(3.0, index=grid_points.index)

    # Haversine distance calculation
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate distance in km between two lat/lon points."""
        R = 6371  # Earth radius in km
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)

        a = np.sin(delta_lat / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c

    # For each grid point, sum capacity of nearby wind farms
    weights = []

    for idx, gp in grid_points.iterrows():
        distances = haversine_distance(
            gp["lat"], gp["lon"],
            wind_farms["lat"].values, wind_farms["lon"].values
        )

        nearby_mask = distances <= search_radius_km
        nearby_capacity = wind_farms.loc[nearby_mask, "capacity"].sum()

        # Use nearby capacity, or default to 3 MW if no wind farms found
        weights.append(max(nearby_capacity, 3.0))

    return pd.Series(weights, index=grid_points.index)


def regenerate_grid_points(
    country: str,
    output_dir: Path,
    search_radius_km: float = 50.0,
    data_file: Optional[Path] = None,
    year: Optional[int] = None
):
    """Regenerate grid points with realistic capacity distribution.

    Args:
        country: Country code (e.g., 'NL', 'FR')
        output_dir: Output directory for grid points
        search_radius_km: Search radius for nearby wind farms (default: 50 km)
        data_file: Path to Global Wind Power Tracker Excel file
        year: Training year to filter for (generates year-specific grid points).
              If None, uses all current farms.
    """
    country = country.upper()

    # Load original grid points
    grid_file = Path(f"input/country_level_data/grid_points/{country.lower()}/{country.lower()}_grid_points.csv")

    if not grid_file.exists():
        print(f"✗ Grid points file not found: {grid_file}")
        return

    grid_points = pd.read_csv(grid_file)
    print(f"\n✓ Loaded grid points: {len(grid_points)} points")

    # Load wind farm data
    if data_file is None:
        data_file = Path("input/Global-Wind-Power-Tracker-February-2026.xlsx")

    wind_farms = load_wind_farms(country, data_file, year=year)

    # Calculate capacity weights
    if not wind_farms.empty:
        print(f"\nCalculating capacity weights (radius={search_radius_km} km)...")
        grid_points["capacity"] = calculate_grid_capacity_weights(
            grid_points,
            wind_farms,
            search_radius_km
        )
    else:
        print("⚠ Using default 3 MW per grid point (no wind farm data)")
        grid_points["capacity"] = 3.0

    # Save updated grid points
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Include year in filename if provided
    if year is not None:
        output_file = output_dir / f"{country.lower()}_grid_points_{year}.csv"
    else:
        output_file = output_dir / f"{country.lower()}_grid_points.csv"

    grid_points.to_csv(output_file, index=False)

    print(f"\n✓ Regenerated grid points saved to: {output_file}")
    print(f"  Capacity distribution:")
    print(f"    Min: {grid_points['capacity'].min():.2f} MW")
    print(f"    Max: {grid_points['capacity'].max():.2f} MW")
    print(f"    Mean: {grid_points['capacity'].mean():.2f} MW")
    print(f"    Total: {grid_points['capacity'].sum():.0f} MW")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Regenerate grid points with realistic capacity distribution from Global Wind Power Tracker"
    )
    parser.add_argument(
        "--country",
        type=str,
        required=True,
        help="Country code (NL, FR, BE, DE, DK, UK, SE, NO, ES, PT, IT, IE)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=50.0,
        help="Search radius in km for nearby wind farms (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (defaults to input/country_level_data/grid_points/{country}/)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to Global Wind Power Tracker Excel file (default: input/Global-Wind-Power-Tracker-February-2026.xlsx)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Training year to filter for (generates year-specific grid points with operational farms only). If not provided, uses all current farms.",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"input/country_level_data/grid_points/{args.country.lower()}")

    print("=" * 80)
    print("Regenerate Grid Points with Global Wind Power Tracker Capacity Distribution")
    print("=" * 80)

    regenerate_grid_points(
        args.country,
        args.output,
        args.radius,
        data_file=args.data,
        year=args.year
    )

    print("\n" + "=" * 80)
    print("Note: Re-run your country-level training with updated grid point capacities")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
