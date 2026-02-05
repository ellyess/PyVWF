# Terrain Data Requirements for PyVWF

This guide explains what terrain data you need for PyVWF's ML-based bias correction and where to get it.

## Quick Summary

For ML-based bias correction, useful terrain features include:

**Essential:**
- Elevation (DEM - Digital Elevation Model)
- Distance to coastline

**Recommended:**
- Slope
- Terrain roughness
- Land cover type

**Advanced:**
- Aspect (terrain orientation)
- Curvature
- Distance to specific features (mountains, water bodies)

## Terrain Features Explained

### 1. Elevation (Required)

**What:** Height above sea level in meters

**Why:** 
- Orographic effects influence wind speed
- Higher elevations often have stronger winds
- Affects atmospheric boundary layer

**Typical range:** -100m (below sea level) to 4000m (mountains)

**Resolution:** 30m-1km depending on region

### 2. Distance to Coast (Recommended)

**What:** Distance from nearest coastline in kilometers

**Why:**
- Land-sea transitions affect wind characteristics
- Offshore/onshore flow patterns
- Surface roughness changes

**Typical range:** 0-500km

### 3. Slope (Recommended)

**What:** Steepness of terrain in degrees

**Why:**
- Flow acceleration/deceleration on hills
- Sheltering effects in valleys
- Wind channeling

**Typical range:** 0-45 degrees

**Computation:** Can be derived from elevation data

### 4. Terrain Roughness (Recommended)

**What:** Standard deviation of elevation in local neighborhood

**Why:**
- Indicates terrain complexity
- Affects turbulence and wind patterns
- Different from surface roughness length (z0)

**Typical range:** 0-200m

**Computation:** Rolling standard deviation of elevation

### 5. Aspect (Optional)

**What:** Direction terrain faces (0-360°, 0=North)

**Why:**
- Prevailing wind interaction with terrain
- Solar heating effects (secondary)
- Exposure to dominant wind directions

**Computation:** Derived from elevation gradients

### 6. Curvature (Optional)

**What:** Concavity/convexity of terrain

**Why:**
- Flow convergence/divergence
- Accelerated flow on convex features
- Sheltering in concave areas

**Computation:** Second derivative of elevation

### 7. Land Cover (Advanced)

**What:** Surface type (forest, urban, water, agriculture, etc.)

**Why:**
- Surface roughness length proxy
- Different drag characteristics
- Seasonal variations (vegetation)

**Categories:** 10-20 classes typically

## Data Sources

### Global Datasets (Free)

#### 1. SRTM (Shuttle Radar Topography Mission)

**Best for:** Elevation data globally

**Coverage:** 60°N to 56°S (most populated areas)

**Resolution:** 
- SRTM-1: ~30m (1 arc-second)
- SRTM-3: ~90m (3 arc-second)

**Format:** GeoTIFF

**Download:**
- https://earthexplorer.usgs.gov/
- https://dwtkns.com/srtm30m/ (easy interface)

**PyVWF use:**
```python
import rasterio
import xarray as xr

# Read SRTM
with rasterio.open('srtm_tile.tif') as src:
    elevation = src.read(1)
    lon = ...  # extract from bounds
    lat = ...
    
# Convert to xarray
ds = xr.Dataset({
    'elevation': (['lat', 'lon'], elevation)
}, coords={'lat': lat, 'lon': lon})
ds.to_netcdf('terrain.nc')
```

#### 2. ETOPO (Global Relief Model)

**Best for:** Global coverage including bathymetry (offshore)

**Coverage:** Global

**Resolution:** 
- ETOPO 2022: 15 arc-second (~450m)
- ETOPO 1: 1 arc-minute (~1.8km)

**Format:** NetCDF (ready for PyVWF!)

**Download:** https://www.ncei.noaa.gov/products/etopo-global-relief-model

**PyVWF use:** Can use directly if coordinates match

#### 3. EU-DEM (European Digital Elevation Model)

**Best for:** High-resolution elevation in Europe

**Coverage:** Europe (EEA39 countries)

**Resolution:** 25m

**Format:** GeoTIFF

**Download:** https://land.copernicus.eu/imagery-in-situ/eu-dem

#### 4. GEBCO (General Bathymetric Chart of the Oceans)

**Best for:** Offshore/bathymetry data

**Coverage:** Global oceans

**Resolution:** 15 arc-second (~450m)

**Format:** NetCDF

**Download:** https://www.gebco.net/data_and_products/gridded_bathymetry_data/

**Use for:** Distance to coast, offshore elevation

#### 5. Copernicus Land Cover

**Best for:** Land cover classification

**Coverage:** Global

**Resolution:** 100m

**Format:** GeoTIFF

**Download:** https://land.copernicus.eu/global/products/lc

**Classes:** Water, trees, grassland, crops, built-up, bare, etc.

#### 6. OpenStreetMap Coastlines

**Best for:** Computing distance to coast

**Coverage:** Global

**Format:** Shapefile, GeoJSON

**Download:** 
- https://osmdata.openstreetmap.de/data/coastlines.html
- https://www.naturalearthdata.com/ (Natural Earth coastlines)

**PyVWF use:**
```python
from vwf.ml_correction import add_distance_to_coast

df = add_distance_to_coast(
    df,
    coastline_geojson='coastlines.geojson',
)
```

### Regional High-Resolution Options

#### Europe
- **EU-DEM**: 25m elevation
- **Copernicus Land Monitoring**: Various products
- **CORINE Land Cover**: Detailed land use

#### United States
- **USGS 3DEP**: 10m or better elevation
- **NLCD**: 30m land cover

#### UK
- **OS Terrain**: 5m-50m elevation
- **CEH Land Cover**: 25m land cover

## Recommended Workflow

### Step 1: Download Elevation Data

**For Europe (Denmark example):**
```bash
# Option A: SRTM (global, 90m)
# Download tiles covering Denmark from earthexplorer.usgs.gov

# Option B: EU-DEM (Europe only, 25m - better!)
# Download from Copernicus Land portal

# Option C: ETOPO (quick start, coarser)
wget https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/bedrock/grid_registered/netcdf/ETOPO1_Bed_g_gmt4.grd.gz
```

### Step 2: Process to NetCDF

**If you have GeoTIFF:**
```python
import rasterio
import xarray as xr
import numpy as np

# Read GeoTIFF
with rasterio.open('eu_dem_denmark.tif') as src:
    elevation = src.read(1)
    transform = src.transform
    
    # Get coordinates
    height, width = elevation.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    lon, lat = rasterio.transform.xy(transform, rows, cols)
    lon = np.array(lon)[0, :]
    lat = np.array(lat)[:, 0]

# Create xarray Dataset
ds = xr.Dataset(
    {'elevation': (['lat', 'lon'], elevation)},
    coords={'lat': lat, 'lon': lon}
)

# Add attributes
ds['elevation'].attrs = {
    'units': 'meters',
    'long_name': 'Elevation above sea level',
    'source': 'EU-DEM 25m'
}

# Save
ds.to_netcdf('terrain_denmark.nc')
```

### Step 3: Compute Derived Features

```python
from vwf.ml_correction import compute_terrain_derivatives

# Load elevation
ds = xr.open_dataset('terrain_denmark.nc')

# Compute derivatives
terrain = compute_terrain_derivatives(
    ds['elevation'],
    resolution_deg=0.25,  # Grid resolution in degrees
)

# Now terrain contains:
# - elevation, slope, aspect, roughness, curvature
terrain.to_netcdf('terrain_denmark_full.nc')
```

### Step 4: Add Distance to Coast

```python
import pandas as pd
from vwf.ml_correction import add_distance_to_coast

# Load your correction points
corrections = pd.read_csv('correction_points.csv')

# Download coastline from Natural Earth or OSM
# https://www.naturalearthdata.com/downloads/10m-physical-vectors/

# Add distance feature
corrections = add_distance_to_coast(
    corrections,
    coastline_geojson='coastlines.geojson',
)

# Now corrections has 'distance_to_coast_km' column
```

## Minimal Setup (Quick Start)

If you don't have terrain data yet, you can start with just coordinates:

```python
from vwf.ml_correction import train_correction_model

# Your correction points with just lon, lat, scalar, offset
corrections = pd.read_csv('correction_points.csv')

# Train using just coordinates (less accurate but works)
model = train_correction_model(
    corrections,
    target_col='scalar',
    feature_cols=['lon', 'lat'],  # Minimal features
    model_type='random_forest',
)
```

Then add terrain features later for better performance.

## Full Setup (Recommended)

### For Denmark Example:

```bash
# 1. Download EU-DEM for Denmark
#    https://land.copernicus.eu/imagery-in-situ/eu-dem
#    Select tiles covering Denmark (approximately 55-58°N, 8-13°E)

# 2. Download coastline
wget https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_coastline.zip
unzip ne_10m_coastline.zip
```

### Process terrain data:

```python
import xarray as xr
import rasterio
import geopandas as gpd
from vwf.ml_correction import compute_terrain_derivatives, add_distance_to_coast

# 1. Load elevation (assuming you have GeoTIFF)
with rasterio.open('eu_dem_denmark.tif') as src:
    elevation = src.read(1)
    # ... extract coordinates (see Step 2 above)

# Create dataset
ds = xr.Dataset(
    {'elevation': (['lat', 'lon'], elevation)},
    coords={'lat': lat, 'lon': lon}
)

# 2. Compute derivatives
terrain = compute_terrain_derivatives(ds['elevation'], resolution_deg=0.25)
terrain.to_netcdf('input/terrain/denmark_terrain.nc')

# 3. For your correction points, add distance to coast
corrections = pd.read_csv('out/correction_points.csv')
corrections = add_distance_to_coast(
    corrections,
    coastline_geojson='ne_10m_coastline.shp',
)
corrections.to_csv('out/correction_points_with_terrain.csv', index=False)
```

### Use in ML training:

```python
from vwf.ml_correction import create_feature_matrix, train_correction_model

# Create feature matrix (extracts terrain at each point)
features = create_feature_matrix(
    corrections,
    terrain_nc='input/terrain/denmark_terrain.nc',
    coastline_geojson='ne_10m_coastline.shp',
)

# Train model
model = train_correction_model(
    features,
    target_col='scalar',
    model_type='random_forest',
)

# See feature importance
print(model['feature_importance'])
```

## File Structure

Recommended organization:

```
your_project/
├── input/
│   ├── terrain/
│   │   ├── denmark_terrain.nc          # Elevation + derivatives
│   │   ├── europe_terrain.nc           # Full Europe
│   │   └── north_sea_bathymetry.nc     # Offshore
│   ├── regions/
│   │   ├── coastlines.geojson          # For distance calc
│   │   ├── country_shapes.geojson      # Onshore boundaries
│   │   └── offshore_shapes.geojson     # Offshore boundaries
│   └── land_cover/
│       └── corine_land_cover.nc        # Optional
└── out/
    └── correction_points.csv            # Your bias corrections
```

## NetCDF Format Requirements

Your terrain NetCDF should have:

```python
<xarray.Dataset>
Dimensions:    (lat: 100, lon: 200)
Coordinates:
  * lat        (lat) float64 55.0 55.1 ... 57.9 58.0
  * lon        (lon) float64 8.0 8.1 ... 12.9 13.0
Data variables:
    elevation  (lat, lon) float32 ...    # meters
    slope      (lat, lon) float32 ...    # degrees
    aspect     (lat, lon) float32 ...    # degrees
    roughness  (lat, lon) float32 ...    # meters
    curvature  (lat, lon) float32 ...    # dimensionless
```

PyVWF will automatically:
- Interpolate to your correction point locations
- Handle coordinate name variations (x/y vs lon/lat)
- Extract features needed for training

## Common Issues

### "Could not find coordinates"
- Ensure NetCDF has 'lon' and 'lat' (or 'x' and 'y')
- Check with: `xarray.open_dataset('terrain.nc')`

### "No data at point locations"
- Check coordinate ranges overlap
- Verify coordinate order (some data is lat/lon vs lon/lat)

### "Memory error with large terrain files"
- Use chunking: `ds.chunk({'lat': 100, 'lon': 100})`
- Or extract only your region of interest first

### "CRS mismatch"
- Ensure terrain data is in EPSG:4326 (WGS84)
- Reproject if needed: `gdalwarp -t_srs EPSG:4326`

## Performance Tips

1. **Match resolution to your needs:**
   - Turbine-level: 100m-1km sufficient
   - Grid-level: Match cutout resolution

2. **Pre-extract your region:**
   ```python
   # Extract only Denmark region
   ds = xr.open_dataset('europe_terrain.nc')
   ds_dk = ds.sel(lat=slice(54, 58), lon=slice(8, 13))
   ds_dk.to_netcdf('denmark_terrain.nc')
   ```

3. **Use appropriate data types:**
   - Elevation: float32 (not float64)
   - Land cover: int16
   - This saves memory and disk space

## Next Steps

1. **Download elevation data** for your region
2. **Process to NetCDF** format
3. **Compute derivatives** (slope, roughness, etc.)
4. **Download coastlines** for distance calculation
5. **Test with PyVWF ML module**

See `examples/ml_terrain_correction.py` for complete examples!

## Resources

**Data portals:**
- USGS EarthExplorer: https://earthexplorer.usgs.gov/
- Copernicus: https://land.copernicus.eu/
- Natural Earth: https://www.naturalearthdata.com/
- GEBCO: https://www.gebco.net/

**Processing tools:**
- **rasterio**: Read/write GeoTIFF in Python
- **GDAL**: Convert between formats
- **xarray**: Work with NetCDF
- **geopandas**: Vector data (coastlines)

**Tutorials:**
- Rasterio docs: https://rasterio.readthedocs.io/
- Xarray tutorial: https://xarray-contrib.github.io/xarray-tutorial/
