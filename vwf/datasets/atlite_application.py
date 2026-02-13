import atlite
import xarray as xr

# Load corrections
corrections = xr.open_dataset('output/atlite_corrections/NL_bimonthly_corrections.nc')

# Load your atlite cutout
cutout = atlite.Cutout(
    path="europe-2023.nc",
    module="era5",
    x=slice(3.0, 8.0),  # Netherlands
    y=slice(50.5, 54.0),
    time="2023-01-01",
)

# Apply bimonthly corrections to wind speeds
ds = cutout.data
months = ds['time'].dt.month

# Map to bimonthly periods
period_map = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3,
              7: 4, 8: 4, 9: 5, 10: 5, 11: 6, 12: 6}
periods = xr.DataArray([period_map[m] for m in months.values],
                       dims='time', coords={'time': ds.time})

# Interpolate corrections to cutout grid
corrections_interp = corrections.interp(lat=ds.lat, lon=ds.lon, method='nearest')

# Apply corrections
wnd100m_corrected = ds['wnd100m'].copy()
for period in range(1, 7):  # 6 bimonthly periods
    mask = periods == period
    scalar = corrections_interp.scalar.sel(period=period)
    wnd100m_corrected = xr.where(mask, ds['wnd100m'] * scalar, wnd100m_corrected)

# Use corrected winds in capacity factor calculation
cutout.data['wnd100m'] = wnd100m_corrected
capacity_factors = cutout.wind(...)