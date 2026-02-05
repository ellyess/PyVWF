# PyVWF to Atlite Export - Quick Reference

## Overview

The `export_pyvwf_grid` function interpolates PyVWF bias correction factors (scalar and offset) from scattered turbine locations onto a regular atlite cutout grid using ordinary kriging.

## Basic Usage

```python
from vwf import export_pyvwf_grid

export_pyvwf_grid(
    cutout_nc='cutouts/europe-2023.nc',
    points_csv='out/correction_points.csv',
    out_nc='out/bias_grid.nc',
    onshore_geojson='input/regions/country_shapes.geojson',
    offshore_geojson='input/regions/north_sea_shape.geojson',
)
```

## Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `cutout_nc` | Path | Atlite cutout NetCDF file (defines the output grid) |
| `points_csv` | Path | CSV with correction factors at turbine locations |
| `out_nc` | Path | Output NetCDF path |
| `onshore_geojson` | Path | GeoJSON defining onshore regions |
| `offshore_geojson` | Path | GeoJSON defining offshore regions |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `domain_col` | str | `None` | Column name for onshore/offshore classification |
| `scalar_col` | str | `"scalar"` | Column name for scalar correction |
| `offset_col` | str | `"offset"` | Column name for offset correction |
| `variogram_model` | str | `"spherical"` | Kriging variogram: spherical, linear, gaussian, exponential |
| `workers` | int | `1` | Number of parallel workers (use 1 on macOS) |
| `onshore_thin_if_gt` | int | `15000` | Thin onshore points if exceeding this count |
| `onshore_bin_ddeg` | float | `0.05` | Spatial bin size for thinning (degrees) |
| `n_closest_onshore` | int | `50` | Number of nearest points for onshore kriging |
| `n_closest_offshore` | int | `80` | Number of nearest points for offshore kriging |

## Input Format

### Correction Points CSV

Must contain:
- `lon`, `lat`: Coordinates (or `longitude`, `latitude`)
- `scalar`: Multiplicative correction factor
- `offset`: Additive correction factor

Optional:
- `type` or `domain`: Domain classification ('onshore' or 'offshore')

Example:
```csv
lon,lat,scalar,offset,type
5.123,52.456,1.05,0.02,onshore
3.789,54.321,0.98,-0.01,offshore
```

### Cutout NetCDF

Atlite cutout format with:
- Coordinates: `x` (lon) and `y` (lat)
- Any atlite-compatible cutout will work

### Region GeoJSON

Standard GeoJSON files with:
- Polygon or MultiPolygon geometries
- CRS: EPSG:4326 (or will be converted)

## Output Format

The output NetCDF contains:

| Variable | Description |
|----------|-------------|
| `scalar` | Combined scalar correction (onshore + offshore) |
| `offset` | Combined offset correction (onshore + offshore) |
| `scalar_onshore` | Scalar for onshore regions only |
| `offset_onshore` | Offset for onshore regions only |
| `scalar_offshore` | Scalar for offshore regions only |
| `offset_offshore` | Offset for offshore regions only |
| `is_onshore_aoi` | Boolean mask for onshore area |
| `is_offshore_aoi` | Boolean mask for offshore area |

Dimensions: `(y, x)` matching the cutout grid.

Outside defined regions: `scalar=1.0`, `offset=0.0` (identity correction).

## Usage in Atlite Workflow

```python
import xarray as xr

# Load corrections
corrections = xr.open_dataset('out/bias_grid.nc')

# Apply to capacity factor
cf_uncorrected = ...  # Your atlite CF output
cf_corrected = cf_uncorrected * corrections['scalar'] + corrections['offset']

# Or use domain-specific corrections
cf_onshore = cf_uncorrected * corrections['scalar_onshore'] + corrections['offset_onshore']
cf_offshore = cf_uncorrected * corrections['scalar_offshore'] + corrections['offset_offshore']
```

## Examples

### 1. Basic Export (Default Settings)

```python
export_pyvwf_grid(
    cutout_nc='cutouts/europe-2023.nc',
    points_csv='out/correction_points.csv',
    out_nc='out/bias_grid.nc',
    onshore_geojson='regions/onshore.geojson',
    offshore_geojson='regions/offshore.geojson',
)
```

### 2. High-Performance (Large Dataset)

```python
export_pyvwf_grid(
    cutout_nc='cutouts/europe-2023.nc',
    points_csv='out/correction_points.csv',  # 50k+ points
    out_nc='out/bias_grid_fast.nc',
    onshore_geojson='regions/onshore.geojson',
    offshore_geojson='regions/offshore.geojson',
    workers=1,                   # macOS: 1, Linux: 2-4
    onshore_thin_if_gt=10000,    # Aggressive thinning
    onshore_bin_ddeg=0.1,        # Coarser bins
    n_closest_onshore=30,        # Fewer neighbors
    n_closest_offshore=50,
)
```

### 3. High-Resolution (Small Region)

```python
export_pyvwf_grid(
    cutout_nc='cutouts/north_sea-2023.nc',
    points_csv='out/northsea_corrections.csv',
    out_nc='out/northsea_bias_hires.nc',
    onshore_geojson='regions/northsea_coast.geojson',
    offshore_geojson='regions/north_sea.geojson',
    variogram_model='gaussian',  # Smoother interpolation
    n_closest_onshore=80,        # More neighbors
    n_closest_offshore=100,
    onshore_bin_ddeg=0.03,       # Finer resolution
)
```

### 4. Custom Column Names

```python
export_pyvwf_grid(
    cutout_nc='cutouts/europe-2023.nc',
    points_csv='out/custom_corrections.csv',
    out_nc='out/bias_grid.nc',
    onshore_geojson='regions/onshore.geojson',
    offshore_geojson='regions/offshore.geojson',
    domain_col='site_type',      # Custom domain column
    scalar_col='mult_factor',    # Custom scalar column
    offset_col='add_factor',     # Custom offset column
)
```

### 5. Automatic Domain Classification

```python
# If points_csv has no domain column, points will be
# automatically classified based on GeoJSON regions
export_pyvwf_grid(
    cutout_nc='cutouts/europe-2023.nc',
    points_csv='out/points_no_domain.csv',
    out_nc='out/bias_grid.nc',
    onshore_geojson='regions/onshore.geojson',
    offshore_geojson='regions/offshore.geojson',
    domain_col=None,  # Trigger auto-classification
)
```

## Performance Tips

### Memory & Speed

1. **Use local kriging**: Set `n_closest_onshore/offshore` to avoid O(N²) scaling
2. **Thin large datasets**: Enable thinning with `onshore_thin_if_gt`
3. **Limit workers on macOS**: Use `workers=1` to avoid memory issues
4. **Linux**: Can use `workers=2-4` for speedup

### Accuracy vs Speed

| Priority | Settings |
|----------|----------|
| **Speed** | `n_closest=30`, `onshore_bin_ddeg=0.1`, `variogram='linear'` |
| **Balanced** | `n_closest=50`, `onshore_bin_ddeg=0.05`, `variogram='spherical'` |
| **Accuracy** | `n_closest=100`, `onshore_bin_ddeg=0.03`, `variogram='gaussian'` |

### Large Datasets (>50k points)

```python
export_pyvwf_grid(
    ...,
    onshore_thin_if_gt=10000,
    onshore_bin_ddeg=0.08,
    n_closest_onshore=40,
    n_closest_offshore=60,
    workers=1,
)
```

## Variogram Models

| Model | Characteristics | Best For |
|-------|----------------|----------|
| `spherical` | Smooth, realistic decay | General use (default) |
| `linear` | Simple linear trend | Sparse data |
| `gaussian` | Very smooth | Dense data, offshore |
| `exponential` | Medium smoothness | Varied terrain |

## Troubleshooting

### "Insufficient points to krige"

- Ensure at least 5 points per domain
- Check for NaN values in correction columns
- Verify coordinates are in range

### "Could not find lon/lat coordinates"

- Check cutout has 'x' and 'y' (or 'lon' and 'lat') coordinates
- Use `xarray.open_dataset(cutout_nc)` to inspect

### "No geometries found in GeoJSON"

- Verify GeoJSON file exists and is valid
- Check it contains Polygon/MultiPolygon geometries
- Test with QGIS or geopandas

### Slow performance

- Enable point thinning: reduce `onshore_thin_if_gt`
- Use fewer neighbors: reduce `n_closest_*`
- Try faster variogram: `variogram_model='linear'`

### High memory usage

- Use `workers=1` (especially on macOS)
- Enable aggressive thinning
- Process smaller regions separately

## Advanced Usage

### Separate Onshore/Offshore Processing

```python
# Process domains separately for different settings
export_pyvwf_grid(
    cutout_nc='cutouts/europe-2023.nc',
    points_csv='out/corrections.csv',
    out_nc='out/bias_grid.nc',
    onshore_geojson='regions/onshore.geojson',
    offshore_geojson='regions/offshore.geojson',
    # Different settings per domain
    n_closest_onshore=40,
    n_closest_offshore=100,
    variogram_model='spherical',
)
```

### Multiple Regions

```python
# Export for multiple countries
for country in ['DE', 'FR', 'UK']:
    export_pyvwf_grid(
        cutout_nc=f'cutouts/{country}-2023.nc',
        points_csv=f'out/{country}_corrections.csv',
        out_nc=f'out/{country}_bias_grid.nc',
        onshore_geojson=f'regions/{country}_onshore.geojson',
        offshore_geojson=f'regions/{country}_offshore.geojson',
    )
```

## See Also

- **Quickstart**: `examples/atlite_quickstart.py`
- **Detailed Examples**: `examples/atlite_export_examples.py`
- **Source Code**: `vwf/atlite_export.py`
- **Main README**: `README.md`
