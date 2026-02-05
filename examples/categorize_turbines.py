"""
Example: Categorizing turbines by onshore/offshore location.

This example demonstrates how to use PyVWF's geospatial utilities
to automatically classify turbines based on their geographic location
relative to onshore and offshore regions.
"""
from pathlib import Path
import pandas as pd
from vwf import add_domain_column, filter_by_domain


def example_basic_categorization():
    """Basic example: categorize turbines from a CSV file."""
    # Load turbine data (must have 'lon' and 'lat' columns)
    turbines = pd.DataFrame({
        'turbine_id': [1, 2, 3, 4, 5],
        'lon': [0.1, 2.5, -1.0, 4.2, 1.8],
        'lat': [51.5, 52.0, 53.5, 51.2, 52.8],
        'capacity_mw': [3.0, 5.0, 8.0, 4.5, 6.0],
    })
    
    # Add domain classification based on GeoJSON regions
    turbines = add_domain_column(
        turbines,
        onshore_geojson=Path('input/regions/country_shapes.geojson'),
        offshore_geojson=Path('input/regions/north_sea_shape.geojson'),
        method='spatial_join',  # Fast method (requires rtree/pygeos)
    )
    
    print("Turbines with domain classification:")
    print(turbines)
    print()
    
    # Filter by domain
    onshore_turbines = filter_by_domain(turbines, "onshore")
    offshore_turbines = filter_by_domain(turbines, "offshore")
    
    print(f"Onshore turbines: {len(onshore_turbines)}")
    print(f"Offshore turbines: {len(offshore_turbines)}")
    
    return turbines


def example_with_existing_csv():
    """Example: Load turbines from CSV and add domain column."""
    # Load from CSV file
    turbines = pd.read_csv('input/turbines.csv')
    
    # Handle different column names
    if 'longitude' in turbines.columns:
        turbines = turbines.rename(columns={'longitude': 'lon', 'latitude': 'lat'})
    
    # Add domain classification
    turbines = add_domain_column(
        turbines,
        onshore_geojson=Path('input/regions/country_shapes.geojson'),
        offshore_geojson=Path('input/regions/north_sea_shape.geojson'),
        lon_col='lon',
        lat_col='lat',
        prefer_onshore=True,  # If point is in both regions, classify as onshore
    )
    
    # Save with domain column
    turbines.to_csv('output/turbines_with_domain.csv', index=False)
    
    # Summary statistics
    print("\nDomain distribution:")
    print(turbines['domain'].value_counts())
    
    # Calculate capacity by domain
    print("\nTotal capacity by domain:")
    print(turbines.groupby('domain')['capacity_mw'].sum())
    
    return turbines


def example_custom_regions():
    """Example: Use custom region definitions for different countries."""
    turbines = pd.read_csv('input/country-data/DE/turbines.csv')
    
    # For Germany, you might have specific onshore/offshore regions
    turbines = add_domain_column(
        turbines,
        onshore_geojson=Path('input/regions/germany_onshore.geojson'),
        offshore_geojson=Path('input/regions/germany_offshore.geojson'),
    )
    
    # Process onshore and offshore separately
    onshore = filter_by_domain(turbines, "onshore")
    offshore = filter_by_domain(turbines, "offshore")
    
    print(f"Germany - Onshore: {len(onshore)}, Offshore: {len(offshore)}")
    
    return turbines


def example_comparison_methods():
    """Example: Compare different categorization methods."""
    import time
    
    turbines = pd.read_csv('input/turbines_large.csv')
    
    # Method 1: Spatial join (faster for large datasets)
    start = time.time()
    turbines1 = add_domain_column(
        turbines.copy(),
        onshore_geojson=Path('input/regions/country_shapes.geojson'),
        offshore_geojson=Path('input/regions/north_sea_shape.geojson'),
        method='spatial_join',
    )
    time1 = time.time() - start
    
    # Method 2: Point-in-polygon (more reliable, but slower)
    start = time.time()
    turbines2 = add_domain_column(
        turbines.copy(),
        onshore_geojson=Path('input/regions/country_shapes.geojson'),
        offshore_geojson=Path('input/regions/north_sea_shape.geojson'),
        method='point_in_polygon',
    )
    time2 = time.time() - start
    
    print(f"Spatial join: {time1:.2f}s")
    print(f"Point-in-polygon: {time2:.2f}s")
    print(f"Speedup: {time2/time1:.1f}x")
    
    # Verify results match
    assert (turbines1['domain'] == turbines2['domain']).all(), "Methods produced different results!"
    
    return turbines1


if __name__ == "__main__":
    print("=" * 60)
    print("PyVWF - Turbine Domain Classification Examples")
    print("=" * 60)
    print()
    
    # Run basic example
    print("Example 1: Basic categorization")
    print("-" * 60)
    example_basic_categorization()
    print()
    
    # Uncomment to run other examples (requires actual data files)
    # example_with_existing_csv()
    # example_custom_regions()
    # example_comparison_methods()
