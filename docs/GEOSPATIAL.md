# Geospatial Utilities: Categorizing Turbines

> Extracted from the project README to keep the top-level page focused. This
> reference covers the onshore/offshore classification helpers.

PyVWF includes utilities to automatically categorize turbines as onshore or offshore based on their geographic location relative to region definitions (GeoJSON files).

## Basic Usage

```python
import pandas as pd
from vwf import add_domain_column, filter_by_domain

# Load turbine data with lat/lon coordinates
turbines = pd.read_csv('input/turbines.csv')

# Automatically categorize based on GeoJSON regions
turbines = add_domain_column(
    turbines,
    onshore_geojson='input/regions/country_shapes.geojson',
    offshore_geojson='input/regions/north_sea_shape.geojson',
)

# The 'domain' column now contains 'onshore', 'offshore', or 'unknown'
print(turbines['domain'].value_counts())

# Filter by domain
onshore_turbines = filter_by_domain(turbines, "onshore")
offshore_turbines = filter_by_domain(turbines, "offshore")
```

## Advanced Options

```python
# Use faster spatial join method (requires rtree or pygeos)
turbines = add_domain_column(
    turbines,
    onshore_geojson='regions/onshore.geojson',
    offshore_geojson='regions/offshore.geojson',
    method='spatial_join',  # Default, fast
    prefer_onshore=True,     # If point in both regions, use onshore
)

# Or use point-in-polygon (slower but more reliable)
turbines = add_domain_column(
    turbines,
    onshore_geojson='regions/onshore.geojson',
    offshore_geojson='regions/offshore.geojson',
    method='point_in_polygon',
)

# Handle custom column names
turbines = add_domain_column(
    turbines,
    onshore_geojson='regions/onshore.geojson',
    offshore_geojson='regions/offshore.geojson',
    lon_col='longitude',
    lat_col='latitude',
)
```

For more examples, see:
- `examples/categorize_turbines.py` - Geospatial classification examples
