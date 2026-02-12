# Data Sources

This document describes the sources of input data used in PyVWF, particularly for turbine metadata and generation observations.

## Denmark (DK)

### Turbine Metadata

**File:** `input/country-data/DK/observations/DK_md.csv`

**Source:** Danish Energy Agency (Energistyrelsen) and Energinet

The turbine metadata contains information from Denmark's official wind turbine registry. Key identifiers and data sources include:

- **GSRN Numbers**: Global Service Relation Numbers (GSRN) are unique identifiers assigned by Energinet (the Danish transmission system operator) to each grid-connected energy installation in Denmark. These numbers are used to track energy production and are part of Denmark's national energy data infrastructure.

- **Spatial Coordinates**: UTM 32 Euref89 coordinate system, sourced from SDFE2018 (Styrelsen for Dataforsyning og Effektivisering - the Danish Agency for Data Supply and Efficiency, formerly known as Kortforsyningen).

- **Technical Specifications**: Includes capacity (kW), rotor diameter (m), hub height (m), manufacturer, and model information maintained in Denmark's master data system (Stamdata).

**Data Fields:**
- Turbine identifier (GSRN)
- Date of original connection to grid
- Capacity (kW)
- Rotor diameter (m)
- Hub height (m)
- Manufacture
- Model
- Local authority information
- Type of location (Land/Hav - onshore/offshore)
- Cadastral information
- UTM coordinates (X east, Y north in UTM 32 Euref89)
- Origin of coordinates (SDFE2018 or Stamdata)

**Access:** This data can be accessed through:
- [Energinet's Data Hub](https://energinet.dk/)
- [Danish Energy Agency's Master Data Register](https://ens.dk/ansvarsomraader/vindenergi/registre-vindmoeller)
- [eNerginet Stamdataregister](https://www.energidataservice.dk/)

### Generation Data

**Files:** `input/country-data/DK/observations/Denmark_YYYY.xlsx` (where YYYY = 2015-2020)

**Source:** Energinet / Danish Energy Data Service

Monthly generation data for individual wind turbines identified by their GSRN numbers. This data represents the actual measured electricity production from each turbine aggregated on a monthly basis.

**Historical Data:**
- `maanedsdata_2002_2017.xlsx`: Historical monthly data spanning 2002-2017
- `anlaeg.xlsx`: Plant/facility information
- `match_turb_dk.xlsx`: Matching table between turbine identifiers

**Data Format:**
- Columns: GSRN identifier followed by 12 monthly columns (January through December)
- Units: Typically in kWh or MWh (check specific file headers)
- Time Period: Individual annual files for 2015-2020

**Access:** Monthly generation data can be obtained from:
- [Energy Data Service API](https://www.energidataservice.dk/): Official open data platform for Danish energy system data
- Energinet's transparency platform

### Data Quality and Usage Notes

1. **GSRN Identifiers**: Some turbines may have incomplete data marked with "Ukendt" (Unknown) for manufacturer or model information.

2. **Coordinates**: Most turbines have coordinates from SDFE2018, but some may only have "Stamdata" as the source, which may indicate lower precision or missing coordinates.

3. **Turbine Types**: The "Type of location" field distinguishes between:
   - "Land" (Onshore)
   - "Hav" (Offshore)

4. **Data Coverage**: The metadata file contains turbines connected to the grid at various dates, so not all turbines will have generation data for all years.

### References

- [Energinet - Danish TSO](https://energinet.dk/)
- [Danish Energy Agency (Energistyrelsen)](https://ens.dk/)
- [Energy Data Service](https://www.energidataservice.dk/)
- [SDFE - Agency for Data Supply and Efficiency](https://sdfe.dk/)

## Germany (DE)

**Files:** `input/country-data/DE/observations/`

**Source:** To be documented. Please refer to the German data sources when this information becomes available.

## United Kingdom (UK)

**Files:** `input/country-data/UK/observations/`

**Source:** To be documented. Please refer to the UK data sources when this information becomes available.

## Reanalysis Data (ERA5)

**Source:** ECMWF ERA5 Reanalysis

PyVWF uses ERA5 reanalysis data from the European Centre for Medium-Range Weather Forecasts (ECMWF).

**Required Variables:**
- 100m u-component of wind (u100)
- 100m v-component of wind (v100)
- 10m u-component of wind (u10) - optional, for roughness calculation
- 10m v-component of wind (v10) - optional, for roughness calculation
- Forecast surface roughness (optional, if not calculating from wind components)

**Access:** 
- [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
- Dataset: `reanalysis-era5-single-levels`

**Citation:**
Hersbach, H., Bell, B., Berrisford, P., et al. (2020). The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, 146(730), 1999-2049.

## Data Usage and Attribution

When using PyVWF with these data sources in academic publications, please:

1. **Cite the data sources** appropriately based on their terms of use
2. **Acknowledge Energinet** for Danish wind generation data
3. **Cite ERA5** as described in the Copernicus Climate Data Store
4. **Reference PyVWF** in your methodology section

For commercial use, please review the data licensing terms from each provider, as some data sources may have restrictions on commercial applications.

## Contributing Data Sources

If you have information about data sources for other countries or updates to existing documentation, please:

1. Open an issue on the PyVWF GitHub repository
2. Submit a pull request with updated documentation
3. Include relevant links and references to official data sources

Last updated: 2026-02-12
