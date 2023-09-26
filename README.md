# The Python Virtual Wind Farm (PyVWF) model
---
`PyVWF` is a Python rewrite of the [VWF model](https://github.com/renewables-ninja/vwf/tree/master) developed by Iain Staffell. The wind energy simulations on [Renewables.ninja](https://www.renewables.ninja/) are based on the VWF model.

---

## Functionality
This model has the ability to produce bias corrected power output from wind farms based on reanalysis data. The model has two main methods in calculating wind speed:
- `method_1 : w(h) = A * np.log(h / z)` (based on original `vwf` model)
- `method_2 : w(h) = v(100m) * ln(h/z0)/ln(100m/z0), where z0 is surface roughness`

If using MERRA-2 `method_1` is the only viable method currently. If using ERA-5 either method can be used. I recommend only using ERA-5 and `method_2` for results as this is the route we wish for this model to go.

## Setup

### Download reanalysis wind speed data
First, download the necessary input  reanalysis data data:
- NASA's [MERRA-2 reanalysis](https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/), specifically the [SLV tables](http://dx.doi.org/10.5067/VJAFPLI1CSIV) (M2T1NXSLV).
The easiest (and most wasteful) option is to use the 'Online archive' link and download the complete files (approx 400 MB per day).  Alternatively you could use the various subsetting tools available, selecting the DISPH, U2M, V2M, U10M, V10M, U50M and V50M variables, and whatever region and time period you are interested in.  Good luck, they rarely function correctly!
- ECMWF's [ERA-5 reanalysis](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form), for `method_2` we required `100m u-component of wind`, `100m v-component of wind` and `Forecast surface roughness`.


