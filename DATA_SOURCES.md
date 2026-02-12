# Data Sources Documentation

This document provides information about the data sources used in PyVWF, particularly for the Denmark example datasets included in the repository.

## Denmark Data

Denmark (DK) is used as the primary example throughout PyVWF documentation and quickstart scripts. The following sections describe the sources of this data.

### Turbine Metadata

**File:** `input/country-data/DK/observations/DK_md.csv`

**Source:** The turbine metadata is derived from the **Danish Energy Agency's Master Data Register** via the **GSRN (Global Location Number System for Energy)** identifier system.

**Data Provider:** [Energinet](https://www.energinet.dk/) (Danish Transmission System Operator)

**Key Information:**
- **GSRN Identifier:** Each turbine has a unique Global System of Registration Number (GSRN) identifier
- **Coordinates:** UTM 32 Euref89 coordinate system (X east, Y north)
- **Coordinate Source:** SDFE2018 (Agency for Data Supply and Infrastructure, Denmark - formerly Styrelsen for Dataforsyning og Effektivisering)
- **Fields Include:**
  - Turbine identifier (GSRN)
  - Date of original connection to grid
  - Capacity (kW)
  - Rotor diameter (m)
  - Hub height (m)
  - Manufacturer and Model
  - Geographic coordinates
  - Local authority information
  - Type of location (Land/Hav - Onshore/Offshore in Danish)
  - Distribution company installation number

**Access:** Wind turbine data in Denmark is publicly available through:
- [Energinet's Open Data Portal](https://www.energinet.dk/data)
- [Danish Energy Agency's Master Data Register](https://ens.dk/)

### Generation Observation Data

**Files:** 
- `input/country-data/DK/observations/Denmark_2015.xlsx`
- `input/country-data/DK/observations/Denmark_2016.xlsx`
- `input/country-data/DK/observations/Denmark_2017.xlsx`
- `input/country-data/DK/observations/Denmark_2018.xlsx`
- `input/country-data/DK/observations/Denmark_2019.xlsx`
- `input/country-data/DK/observations/Denmark_2020.xlsx`

**Source:** Wind generation observation data for individual turbines.

**Data Provider:** Likely from [Energinet](https://www.energinet.dk/) operational data or the Danish TSO's metering systems. The GSRN system is used for settlement and metering in the Danish electricity market.

**Format:** Excel spreadsheets containing monthly or hourly generation data for individual wind turbines, indexed by GSRN identifier.

**Training Period Used in Examples:** 2015-2019

**Test Period Used in Examples:** 2020

**Additional Reference Files:**
- `anlaeg.xlsx` - Historical wind farm registry data
- `maanedsdata_2002_2017.xlsx` - Monthly data archive (2002-2017)
- `match_turb_dk.xlsx` - Turbine matching reference file

### Country-Level Aggregated Data

**File:** `input/country-data/northsea_country_generation.csv`

**Source:** [Eurostat](https://ec.europa.eu/eurostat) energy statistics

**Classification:** SIEC (Standard International Energy Product Classification)

**Content:** Aggregated wind generation output at the country level for multiple North Sea countries, used for validation and comparison purposes.

## Reanalysis Data (ERA5)

**Source:** [ECMWF's ERA5 Reanalysis](https://cds-beta.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)

**Provider:** European Centre for Medium-Range Weather Forecasts (ECMWF)

**Access:** Available through the Copernicus Climate Data Store (CDS)

**Required Variables:**
- 100m u-component of wind (eastward)
- 100m v-component of wind (northward)
- 10m u-component of wind (eastward)
- 10m v-component of wind (northward)

**OR alternatively:**
- 100m u-component of wind
- 100m v-component of wind
- Forecast surface roughness

**Spatial Resolution:** 0.25° × 0.25° (approximately 31 km × 31 km at mid-latitudes)

**Temporal Resolution:** Hourly

**Usage in PyVWF:** ERA5 data should be downloaded and placed in:
- Training data: `input/era5/<COUNTRY>/train/`
- Test data: `input/era5/<COUNTRY>/test/`

**License:** ERA5 data is available under the [Copernicus License](https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf)

## Power Curves

**File:** `input/power_curves.csv`

**Source:** Wind turbine manufacturer specifications

**Note:** Due to proprietary nature of detailed power curves, only a format example is provided in the repository. Users should obtain power curves from:
- Turbine manufacturer datasheets
- [The Wind Power Database](https://www.thewindpower.net/)
- [NREL's Turbine Database](https://www.nrel.gov/wind/)
- Research publications and technical reports

**Format:** CSV file with wind speed (m/s) as rows and turbine model names as columns, providing power output or capacity factor at each wind speed.

## Data Usage and Attribution

When using PyVWF for research or publications, please ensure proper attribution of data sources:

### For Denmark Data:
- **Turbine Metadata:** Acknowledge Energinet and the Danish Energy Agency
- **Generation Data:** Acknowledge Energinet (if applicable)
- **Example Citation:**
  > "Wind turbine metadata and generation data for Denmark were obtained from Energinet's Open Data Portal and the Danish Energy Agency's Master Data Register."

### For ERA5 Data:
- **Citation:** Hersbach, H., et al. (2020): ERA5 hourly data on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/cds.adbb2d47

### For PyVWF Methodology:
- The bias correction methodology is based on Staffell, I., & Pfenninger, S. (2016). Using bias-corrected reanalysis to simulate current and future wind power output. Energy, 114, 1224-1239.

## Preparing Your Own Data

If you want to use PyVWF with data from other countries or regions, you'll need to prepare similar datasets:

1. **Turbine Metadata CSV** with minimum fields:
   - Turbine ID
   - Latitude, Longitude
   - Hub height (m)
   - Capacity (MW or kW)
   - Turbine model name (matching power curve data)
   - Type (onshore/offshore) - optional

2. **Observation Data** (CSV or Excel):
   - Time-indexed generation data
   - Columns for each turbine (using turbine IDs)
   - Units: Capacity factor (0-1) or actual generation (MW)

3. **ERA5 Reanalysis Data** (NetCDF):
   - Downloaded for your region and time period
   - See `ATLITE_EXPORT_GUIDE.md` for download instructions

4. **Power Curves** (CSV):
   - One column per turbine model
   - Wind speed vs. power output or capacity factor

## Data Privacy and Confidentiality

Some generation data may be commercially sensitive or subject to data protection regulations. Users should:
- Ensure they have appropriate licenses and permissions to use the data
- Respect data privacy regulations (e.g., GDPR in Europe)
- Acknowledge data providers as required by data use agreements
- Not redistribute proprietary data without permission

## Questions and Support

For questions about:
- **Denmark data specifically:** Contact [Energinet](https://www.energinet.dk/) or the [Danish Energy Agency](https://ens.dk/)
- **PyVWF usage:** See README.md or contact the repository maintainer
- **ERA5 data access:** See the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)

## References

1. Staffell, I., & Pfenninger, S. (2016). Using bias-corrected reanalysis to simulate current and future wind power output. Energy, 114, 1224-1239.
2. Hersbach, H., et al. (2020). The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, 146(730), 1999-2049.
3. Energinet Open Data Portal: https://www.energinet.dk/data
4. Danish Energy Agency Master Data Register: https://ens.dk/
5. Copernicus Climate Data Store: https://cds.climate.copernicus.eu/

---

**Last Updated:** February 2026

**Maintained by:** PyVWF Contributors

**Note:** Data sources and access methods may change over time. Please verify current data availability and access procedures with the respective data providers.
