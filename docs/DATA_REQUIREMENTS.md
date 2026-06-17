# Data Requirements

> Extracted from the project README. This reference lists the input data PyVWF
> expects and how to obtain the reanalysis winds.

PyVWF expects the following input data types.

## Required Inputs

|Data|Format|Description|
|---|---|---|
|Reanalysis wind data|NetCDF|ERA5 wind components (e.g. u100, v100)|
|Turbine metadata|CSV|Location, capacity, hub height, turbine model|
|Observed generation|CSV|Time series of wind generation or capacity factor|
|Power curves|CSV|Wind speed to power conversion|

The files you should provide are:

- Observation data for all training years placed in `input/country-data/observation/`. Example files are in the repository.
- Reanalysis data for all training years and test years in `data/era5/<country>/<test/train>/`
- Turbine metadata which contains information such as the height, latitude, longitude, turbine ID, turbine model and capacity placed in `data/turb_info/`. An example is provided, plan to make this file easier to create.
- Wind turbine power curves in a .csv file with model names in each column providing the power output with respect to wind speed. Due to proprietary data used in our curve file an example of the format is shown in `input/power_curves.csv`

### Download reanalysis wind speed data

Download the necessary input ERA-5 data (Years in a period can be downloaded separately or together as they will be joined. Ensure training data is separate to validation):

- ECMWF's [ERA-5 reanalysis](https://cds-beta.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download), the required variables are either:
  - 100m u-component of wind, 100m v-component of wind, 10m u-component of wind and 10m v-component of wind (surface roughness is calculated instead and is more accurate).
  - 100m u-component of wind, 100m v-component of wind and Forecast surface roughness. 
