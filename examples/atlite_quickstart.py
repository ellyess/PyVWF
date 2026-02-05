#!/usr/bin/env python3
"""
Quickstart: PyVWF to Atlite Export

This is a minimal, ready-to-run example that creates synthetic data
and demonstrates the complete workflow.

Just run: python examples/atlite_quickstart.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr


def create_synthetic_cutout(path: Path):
    """Create a minimal synthetic atlite cutout for demonstration."""
    print("Creating synthetic cutout...")
    
    # Small European domain
    lon = np.linspace(-5, 15, 40)
    lat = np.linspace(48, 58, 30)
    
    # Create simple dataset (atlite cutout format)
    ds = xr.Dataset(
        coords={
            'x': lon,
            'y': lat,
        }
    )
    
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    print(f"  ✓ Created cutout: {path}")
    
    return path


def create_synthetic_corrections(path: Path, n_points: int = 300):
    """Create synthetic correction points."""
    print(f"Creating {n_points} synthetic correction points...")
    
    np.random.seed(42)
    
    # European coordinates
    lon = np.random.uniform(-5, 15, n_points)
    lat = np.random.uniform(48, 58, n_points)
    
    # Realistic correction factors
    # Scalar: 0.85 to 1.15 (±15% multiplicative correction)
    # Offset: -0.08 to 0.08 (±8% additive correction)
    scalar = np.random.normal(1.0, 0.08, n_points)
    offset = np.random.normal(0.0, 0.03, n_points)
    
    scalar = np.clip(scalar, 0.85, 1.15)
    offset = np.clip(offset, -0.08, 0.08)
    
    # Simple domain classification
    # Offshore: 0 < lon < 10, 50 < lat < 56 (North Sea-ish)
    domain = []
    for ln, lt in zip(lon, lat):
        if 0 < ln < 10 and 50 < lt < 56:
            domain.append('offshore')
        else:
            domain.append('onshore')
    
    df = pd.DataFrame({
        'lon': lon,
        'lat': lat,
        'scalar': scalar,
        'offset': offset,
        'type': domain,
    })
    
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    
    print(f"  ✓ Created corrections: {path}")
    print(f"    Onshore: {(df['type'] == 'onshore').sum()}")
    print(f"    Offshore: {(df['type'] == 'offshore').sum()}")
    
    return path


def create_synthetic_regions(onshore_path: Path, offshore_path: Path):
    """Create simple region GeoJSON files."""
    print("Creating synthetic region files...")
    
    import geopandas as gpd
    from shapely.geometry import box
    
    # Onshore: larger box covering land areas
    onshore_box = box(-5, 48, 10, 58)
    onshore_gdf = gpd.GeoDataFrame(
        {'geometry': [onshore_box]},
        crs='EPSG:4326'
    )
    
    # Offshore: North Sea-like box
    offshore_box = box(0, 50, 10, 56)
    offshore_gdf = gpd.GeoDataFrame(
        {'geometry': [offshore_box]},
        crs='EPSG:4326'
    )
    
    onshore_path.parent.mkdir(parents=True, exist_ok=True)
    offshore_path.parent.mkdir(parents=True, exist_ok=True)
    
    onshore_gdf.to_file(onshore_path, driver='GeoJSON')
    offshore_gdf.to_file(offshore_path, driver='GeoJSON')
    
    print(f"  ✓ Created onshore region: {onshore_path}")
    print(f"  ✓ Created offshore region: {offshore_path}")
    
    return onshore_path, offshore_path


def run_export_example():
    """Run the complete export example."""
    print("\n" + "="*70)
    print("PyVWF to Atlite Export - Quickstart Example")
    print("="*70 + "\n")
    
    # Define paths
    TEMP_DIR = Path("out/quickstart_demo")
    CUTOUT = TEMP_DIR / "synthetic_cutout.nc"
    CORRECTIONS = TEMP_DIR / "correction_points.csv"
    ONSHORE = TEMP_DIR / "onshore_region.geojson"
    OFFSHORE = TEMP_DIR / "offshore_region.geojson"
    OUTPUT = TEMP_DIR / "bias_correction_grid.nc"
    
    # Step 1: Create synthetic input files
    print("STEP 1: Creating synthetic input files")
    print("-" * 70)
    create_synthetic_cutout(CUTOUT)
    create_synthetic_corrections(CORRECTIONS, n_points=300)
    create_synthetic_regions(ONSHORE, OFFSHORE)
    print()
    
    # Step 2: Run the export
    print("STEP 2: Exporting corrections to grid")
    print("-" * 70)
    
    from vwf import export_pyvwf_grid
    
    result = export_pyvwf_grid(
        cutout_nc=CUTOUT,
        points_csv=CORRECTIONS,
        out_nc=OUTPUT,
        onshore_geojson=ONSHORE,
        offshore_geojson=OFFSHORE,
        domain_col='type',
        scalar_col='scalar',
        offset_col='offset',
        variogram_model='spherical',
        workers=1,
        onshore_thin_if_gt=15000,
        onshore_bin_ddeg=0.05,
        n_closest_onshore=50,
        n_closest_offshore=80,
    )
    
    print(f"\n✓ Export complete!")
    print(f"  Output saved to: {result}")
    print()
    
    # Step 3: Inspect the output
    print("STEP 3: Inspecting output NetCDF")
    print("-" * 70)
    
    ds = xr.open_dataset(result)
    
    print(f"\nDataset structure:")
    print(f"  Dimensions: {dict(ds.dims)}")
    print(f"  Coordinates: {list(ds.coords.keys())}")
    print(f"  Variables: {list(ds.data_vars.keys())}")
    
    print(f"\nScalar correction field:")
    scalar = ds['scalar']
    print(f"  Shape: {scalar.shape}")
    print(f"  Mean: {float(scalar.mean()):.4f}")
    print(f"  Std: {float(scalar.std()):.4f}")
    print(f"  Range: [{float(scalar.min()):.4f}, {float(scalar.max()):.4f}]")
    print(f"  Valid cells: {int((~scalar.isnull()).sum())}")
    
    print(f"\nOffset correction field:")
    offset = ds['offset']
    print(f"  Shape: {offset.shape}")
    print(f"  Mean: {float(offset.mean()):.4f}")
    print(f"  Std: {float(offset.std()):.4f}")
    print(f"  Range: [{float(offset.min()):.4f}, {float(offset.max()):.4f}]")
    print(f"  Valid cells: {int((~offset.isnull()).sum())}")
    
    print(f"\nRegion masks:")
    onshore_mask = ds['is_onshore_aoi']
    offshore_mask = ds['is_offshore_aoi']
    print(f"  Onshore cells: {int(onshore_mask.sum())}")
    print(f"  Offshore cells: {int(offshore_mask.sum())}")
    
    ds.close()
    
    # Step 4: How to use in atlite
    print()
    print("STEP 4: Using in atlite workflow")
    print("-" * 70)
    print("""
To use these corrections in your atlite wind power calculations:

1. Load the correction grid:
    ```python
    import xarray as xr
    corrections = xr.open_dataset('out/quickstart_demo/bias_correction_grid.nc')
    ```

2. Apply to your wind power output:
    ```python
    # Assuming 'capacity_factor' is your uncorrected output
    corrected_cf = capacity_factor * corrections['scalar'] + corrections['offset']
    ```

3. Or use domain-specific corrections:
    ```python
    # For onshore regions
    onshore_cf = capacity_factor * corrections['scalar_onshore'] + corrections['offset_onshore']
    
    # For offshore regions
    offshore_cf = capacity_factor * corrections['scalar_offshore'] + corrections['offset_offshore']
    ```

4. The corrections are already masked to appropriate regions,
   so NaN values indicate areas outside the domain.
""")
    
    print("\n" + "="*70)
    print("Example complete!")
    print("="*70)
    print(f"\nAll files saved to: {TEMP_DIR}")
    print("\nKey outputs:")
    print(f"  • Bias grid: {OUTPUT}")
    print(f"  • Input corrections: {CORRECTIONS}")
    print(f"  • Synthetic cutout: {CUTOUT}")
    print("\nYou can now:")
    print("  1. Inspect the NetCDF with xarray")
    print("  2. Visualize with matplotlib/cartopy")
    print("  3. Use in your atlite workflow")
    print()


if __name__ == "__main__":
    try:
        run_export_example()
    except ImportError as e:
        print(f"\n\nError: Missing dependency - {e}")
        print("\nPlease install required packages:")
        print("  pip install pandas numpy xarray geopandas pykrige shapely")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
