"""
Example: Export PyVWF Bias Correction to Atlite Cutout Grid

This script demonstrates how to use the export_pyvwf_grid function to
interpolate PyVWF correction factors onto an atlite cutout grid.

The output NetCDF can be used in atlite workflows for bias-corrected
wind power simulations.
"""
from pathlib import Path
from vwf import export_pyvwf_grid


def example_basic_export():
    """
    Basic example: Export correction points to atlite grid.
    """
    print("="*70)
    print("Example 1: Basic PyVWF to Atlite Export")
    print("="*70)
    
    # Define paths
    CUTOUT_PATH = Path("cutouts/europe-2023-sarah3-era5.nc")
    POINTS_CSV = Path("out/correction_points.csv")
    OUTPUT_NC = Path("out/pyvwf_bias_grid.nc")
    
    ONSHORE_GEOJSON = Path("input/regions/country_shapes.geojson")
    OFFSHORE_GEOJSON = Path("input/regions/north_sea_shape.geojson")
    
    # Export to grid
    result = export_pyvwf_grid(
        cutout_nc=CUTOUT_PATH,
        points_csv=POINTS_CSV,
        out_nc=OUTPUT_NC,
        onshore_geojson=ONSHORE_GEOJSON,
        offshore_geojson=OFFSHORE_GEOJSON,
        # Optional: specify which column contains domain info
        domain_col="type",  # or "domain", "onshore_offshore", etc.
        # Optional: specify correction column names
        scalar_col="scalar",
        offset_col="offset",
    )
    
    print(f"\n✓ Exported bias grid to: {result}")
    print("\nOutput NetCDF contains:")
    print("  - scalar: Multiplicative correction factor")
    print("  - offset: Additive correction factor")
    print("  - scalar_onshore, offset_onshore: Onshore corrections")
    print("  - scalar_offshore, offset_offshore: Offshore corrections")
    print("  - is_onshore_aoi, is_offshore_aoi: Region masks")
    
    return result


def example_performance_optimized():
    """
    Example: Optimized settings for large datasets.
    """
    print("="*70)
    print("Example 2: Performance-Optimized Export (Large Dataset)")
    print("="*70)
    
    result = export_pyvwf_grid(
        cutout_nc=Path("cutouts/europe-2023-sarah3-era5.nc"),
        points_csv=Path("out/correction_points.csv"),  # e.g., 50,000+ points
        out_nc=Path("out/pyvwf_bias_grid_optimized.nc"),
        onshore_geojson=Path("input/regions/country_shapes.geojson"),
        offshore_geojson=Path("input/regions/north_sea_shape.geojson"),
        
        # Kriging parameters
        variogram_model="spherical",  # or "linear", "gaussian", "exponential"
        
        # Performance settings
        workers=1,  # Use 1 for macOS to avoid memory issues; increase on Linux
        onshore_thin_if_gt=15000,  # Thin onshore points if > 15k
        onshore_bin_ddeg=0.05,  # Spatial bin size for thinning (degrees)
        
        # Local kriging (much faster than global)
        n_closest_onshore=50,   # Use 50 nearest points for onshore
        n_closest_offshore=80,  # Use 80 nearest points for offshore
    )
    
    print(f"\n✓ Optimized export complete: {result}")
    print("\nPerformance tips:")
    print("  - Spatial thinning reduces onshore points (denser coverage)")
    print("  - Local kriging (n_closest) avoids O(N²) scaling")
    print("  - Use workers=1 on macOS, 2-4 on Linux")
    
    return result


def example_custom_regions():
    """
    Example: Use custom region definitions for a specific country.
    """
    print("="*70)
    print("Example 3: Custom Region Definitions (Germany)")
    print("="*70)
    
    result = export_pyvwf_grid(
        cutout_nc=Path("cutouts/germany-2023-era5.nc"),
        points_csv=Path("out/de_correction_points.csv"),
        out_nc=Path("out/germany_bias_grid.nc"),
        
        # Custom region files for Germany
        onshore_geojson=Path("input/regions/germany_onshore.geojson"),
        offshore_geojson=Path("input/regions/germany_offshore.geojson"),
        
        # Use specific domain column
        domain_col="site_type",  # Could be any column with onshore/offshore info
        
        # Adjust for smaller region
        n_closest_onshore=30,
        n_closest_offshore=50,
    )
    
    print(f"\n✓ Germany-specific export: {result}")
    
    return result


def example_without_domain_column():
    """
    Example: Points CSV has no domain column - automatic classification.
    """
    print("="*70)
    print("Example 4: Automatic Domain Classification")
    print("="*70)
    
    # When points CSV doesn't have a domain/type column,
    # the function will automatically classify points based on
    # their location relative to the GeoJSON regions
    
    result = export_pyvwf_grid(
        cutout_nc=Path("cutouts/europe-2023-sarah3-era5.nc"),
        points_csv=Path("out/points_no_domain.csv"),  # No 'type' column
        out_nc=Path("out/auto_classified_bias_grid.nc"),
        onshore_geojson=Path("input/regions/country_shapes.geojson"),
        offshore_geojson=Path("input/regions/north_sea_shape.geojson"),
        
        # Will trigger automatic classification
        domain_col=None,  # or omit this parameter
    )
    
    print(f"\n✓ Auto-classified export: {result}")
    print("\nAutomatic classification:")
    print("  - Points inside onshore GeoJSON → onshore")
    print("  - Points inside offshore GeoJSON → offshore")
    print("  - Points outside both → dropped with warning")
    print("  - Uses fast spatial join (rtree/pygeos) if available")
    
    return result


def example_different_variograms():
    """
    Example: Compare different kriging variogram models.
    """
    print("="*70)
    print("Example 5: Different Variogram Models")
    print("="*70)
    
    # Different variogram models capture different spatial structures
    variograms = ["spherical", "linear", "gaussian", "exponential"]
    
    for vario in variograms:
        print(f"\nExporting with {vario} variogram...")
        
        result = export_pyvwf_grid(
            cutout_nc=Path("cutouts/europe-2023-sarah3-era5.nc"),
            points_csv=Path("out/correction_points.csv"),
            out_nc=Path(f"out/bias_grid_{vario}.nc"),
            onshore_geojson=Path("input/regions/country_shapes.geojson"),
            offshore_geojson=Path("input/regions/north_sea_shape.geojson"),
            variogram_model=vario,
            workers=1,
        )
        
        print(f"  ✓ Saved: {result}")
    
    print("\nVariogram model selection:")
    print("  - spherical: Good default, smooth interpolation")
    print("  - linear: Simple, less smooth")
    print("  - gaussian: Very smooth, may over-smooth")
    print("  - exponential: Intermediate smoothness")
    
    return variograms


def example_north_sea_focus():
    """
    Example: High-resolution export for North Sea offshore wind.
    """
    print("="*70)
    print("Example 6: High-Resolution North Sea Export")
    print("="*70)
    
    result = export_pyvwf_grid(
        cutout_nc=Path("cutouts/north_sea-2023-era5.nc"),
        points_csv=Path("out/northsea_correction_points.csv"),
        out_nc=Path("out/northsea_bias_grid_hires.nc"),
        
        # North Sea specific regions
        onshore_geojson=Path("input/regions/northsea_countries.geojson"),
        offshore_geojson=Path("input/regions/north_sea_shape.geojson"),
        
        # Higher resolution settings for offshore
        n_closest_offshore=100,  # More neighbors for better accuracy
        variogram_model="gaussian",  # Smooth for offshore
        
        # Less aggressive thinning for smaller region
        onshore_thin_if_gt=20000,
        onshore_bin_ddeg=0.03,  # Finer spatial bins
    )
    
    print(f"\n✓ High-res North Sea export: {result}")
    
    return result


def example_verify_output():
    """
    Example: Verify and inspect the output NetCDF.
    """
    print("="*70)
    print("Example 7: Verify Output NetCDF")
    print("="*70)
    
    import xarray as xr
    
    # First, create the output
    result = export_pyvwf_grid(
        cutout_nc=Path("cutouts/europe-2023-sarah3-era5.nc"),
        points_csv=Path("out/correction_points.csv"),
        out_nc=Path("out/pyvwf_bias_grid_test.nc"),
        onshore_geojson=Path("input/regions/country_shapes.geojson"),
        offshore_geojson=Path("input/regions/north_sea_shape.geojson"),
    )
    
    # Load and inspect
    print(f"\nLoading output: {result}")
    ds = xr.open_dataset(result)
    
    print("\nDataset structure:")
    print(ds)
    
    print("\nVariables:")
    for var in ds.data_vars:
        print(f"  - {var}: {ds[var].dims}, shape={ds[var].shape}")
    
    print("\nCoordinates:")
    print(f"  - x (lon): {len(ds.x)} points")
    print(f"  - y (lat): {len(ds.y)} points")
    
    print("\nScalar correction stats:")
    scalar = ds['scalar']
    print(f"  - Mean: {float(scalar.mean()):.3f}")
    print(f"  - Std: {float(scalar.std()):.3f}")
    print(f"  - Min: {float(scalar.min()):.3f}")
    print(f"  - Max: {float(scalar.max()):.3f}")
    print(f"  - Valid cells: {int((~scalar.isnull()).sum())}")
    
    print("\nOffset correction stats:")
    offset = ds['offset']
    print(f"  - Mean: {float(offset.mean()):.3f}")
    print(f"  - Std: {float(offset.std()):.3f}")
    print(f"  - Min: {float(offset.min()):.3f}")
    print(f"  - Max: {float(offset.max()):.3f}")
    print(f"  - Valid cells: {int((~offset.isnull()).sum())}")
    
    print("\nMetadata:")
    for key, val in ds.attrs.items():
        print(f"  - {key}: {val}")
    
    ds.close()
    
    return result


def example_create_correction_points():
    """
    Example: Create sample correction points CSV for testing.
    """
    print("="*70)
    print("Example 8: Create Sample Correction Points")
    print("="*70)
    
    import pandas as pd
    import numpy as np
    
    # Create synthetic correction points for demonstration
    np.random.seed(42)
    
    # European domain roughly
    n_points = 500
    lon = np.random.uniform(-10, 25, n_points)
    lat = np.random.uniform(45, 60, n_points)
    
    # Synthetic correction factors
    # scalar: typically 0.8 to 1.2 (multiplicative)
    # offset: typically -0.1 to 0.1 (additive)
    scalar = np.random.normal(1.0, 0.1, n_points)
    offset = np.random.normal(0.0, 0.05, n_points)
    
    # Clip to reasonable ranges
    scalar = np.clip(scalar, 0.7, 1.3)
    offset = np.clip(offset, -0.15, 0.15)
    
    # Assign domains (onshore vs offshore)
    # Simplified: offshore if close to coast (lon > 0 and lat < 55)
    domain = ['offshore' if (ln > 0 and ln < 10 and lt > 50 and lt < 58) else 'onshore' 
              for ln, lt in zip(lon, lat)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'lon': lon,
        'lat': lat,
        'scalar': scalar,
        'offset': offset,
        'type': domain,
    })
    
    # Save
    output_path = Path("out/sample_correction_points.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Created {len(df)} sample correction points")
    print(f"  Saved to: {output_path}")
    print(f"\nDistribution:")
    print(f"  Onshore: {(df['type'] == 'onshore').sum()}")
    print(f"  Offshore: {(df['type'] == 'offshore').sum()}")
    print(f"\nScalar range: {df['scalar'].min():.3f} to {df['scalar'].max():.3f}")
    print(f"Offset range: {df['offset'].min():.3f} to {df['offset'].max():.3f}")
    
    print("\nSample data:")
    print(df.head())
    
    return output_path


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("PyVWF to Atlite Export - Example Runs")
    print("="*70 + "\n")
    
    print("This script demonstrates how to export PyVWF correction factors")
    print("to an atlite cutout grid using the export_pyvwf_grid function.")
    print()
    print("Note: Examples are commented out by default.")
    print("      Uncomment the ones you want to run and ensure you have")
    print("      the required input files.")
    print()
    
    # Uncomment the examples you want to run:
    
    # Example 1: Basic export
    # example_basic_export()
    
    # Example 2: Performance optimized
    # example_performance_optimized()
    
    # Example 3: Custom regions
    # example_custom_regions()
    
    # Example 4: Automatic classification
    # example_without_domain_column()
    
    # Example 5: Different variograms
    # example_different_variograms()
    
    # Example 6: North Sea high-res
    # example_north_sea_focus()
    
    # Example 7: Verify output
    # example_verify_output()
    
    # Example 8: Create sample data (always safe to run)
    print("Creating sample correction points for testing...")
    example_create_correction_points()
    
    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Uncomment the examples above to run them")
    print("  2. Adjust file paths to match your data")
    print("  3. Use the output NetCDF in your atlite workflow")
    print("\nFor more information, see:")
    print("  - vwf/atlite_export.py (source code)")
    print("  - README.md (documentation)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except FileNotFoundError as e:
        print(f"\n\nError: File not found - {e}")
        print("\nTip: This example requires specific input files.")
        print("     Either create them or adjust the paths in the script.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nTip: Make sure you have the required dependencies:")
        print("  pip install pandas xarray geopandas pykrige")
