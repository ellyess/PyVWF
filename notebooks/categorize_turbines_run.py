import pandas as pd
from vwf import add_domain_column, filter_by_domain

# Load your turbines
turbines = pd.read_csv('turbines.csv')  # Must have 'lon' and 'lat' columns

# Categorize them
turbines = add_domain_column(
    turbines,
    onshore_geojson='input/regions/country_shapes.geojson',
    offshore_geojson='input/regions/north_sea_shape.geojson',
)

# Now turbines has a 'domain' column with 'onshore'/'offshore'/'unknown'
onshore = filter_by_domain(turbines, "onshore")
offshore = filter_by_domain(turbines, "offshore")