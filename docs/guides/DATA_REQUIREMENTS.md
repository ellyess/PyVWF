# Data Requirements

This reference lists the input data PyVWF expects. For the full download and
directory-layout guide, see the **Input data** section of the top-level
[README](../../README.md#input-data), which this page complements. Paths are
defined centrally in [`src/vwf/config.py`](../../src/vwf/config.py) (`PyVWFPaths`).

## Required inputs

| Data | Format | Description | Location |
|---|---|---|---|
| Reanalysis winds | NetCDF | ERA5 wind components (`u100`, `v100`, and `u10`/`v10` or `fsr`) | `input/era5/EU/*.nc` |
| Turbine metadata | CSV | ID, location, capacity, hub height, rotor diameter, model | `input/observations/turbine/<CC>/` |
| Observed generation | CSV | Monthly generation per turbine, or national series for country-level | `input/observations/turbine/<CC>/` or `input/observations/country/observations/<cc>/` |
| Power curves | CSV | Wind speed to power, one column per model | `input/reference/power_curves.csv` |
| Turbine models | CSV | Manufacturer, model, capacity, diameter, power density | `input/reference/models.csv` |

`<CC>` is the upper-case country code, `<cc>` the lower-case form.

## ERA5 reanalysis winds

Download from ECMWF's Copernicus Climate Data Store:
[ERA5 hourly single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download).
Required variables are either:

- 100m u- and v-components of wind plus 10m u- and v-components of wind (PyVWF
  derives surface roughness from the shear, which is more accurate), or
- 100m u- and v-components of wind plus forecast surface roughness (`fsr`).

Place all NetCDF files in `input/era5/EU/`. They are loaded with a single `*.nc`
glob and combined by coordinates, so filenames are free and all years share one
folder. Years within a period can be downloaded separately or together; the
training and test split is applied in code by time selection, not by directory.

## Turbine and power-curve data

The repository ships no turbine metadata or observations; these datasets are
typically proprietary, so you supply your own in the layout above. The shipped
`power_curves.csv` and `models.csv` are the open turbine curve library
(NREL/turbine-models, BSD-3-Clause, DOI 10.11578/dc.20210112.1,
VWF-smoothed), redistributable and documented per column in
`power_curves_provenance.csv`; see `input/README.md` for provenance details.
Do not assume redistribution rights for data obtained elsewhere.
