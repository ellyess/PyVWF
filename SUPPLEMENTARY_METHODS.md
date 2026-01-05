# PyVWF: Python Virtual Wind Farm Model

**Supplementary Software Description**

## S1. Purpose and Scope

PyVWF is a Python-based research framework developed to **process, bias-correct, and validate wind speed and wind power time series** derived from atmospheric reanalysis data against observed wind power generation.

The framework is designed for **reproducible scientific analysis** of wind resource representation and correction, with a particular focus on:

- Daily to monthly temporal resolution
    
- Turbine-level to aggregated regional or national scales
    
- Statistical bias correction of reanalysis-based wind speeds
    

PyVWF is intended for **methodological research and validation studies**, rather than operational forecasting.

---

## S2. Methodological Overview

The computational workflow implemented in PyVWF consists of the following steps:

1. **Ingestion of reanalysis wind data** (e.g. ERA5 wind components)
    
2. **Spatial mapping** between turbine locations and reanalysis grid points
    
3. **Vertical extrapolation** of wind speed to turbine hub height
    
4. **Statistical bias correction** using observed wind generation data
    
5. **Wind-to-power conversion** via turbine-specific power curves
    
6. **Validation and aggregation** of simulated generation
    

Each step is implemented as a modular component to allow independent testing and substitution.

---

## S3. Input Data Requirements

### S3.1 Reanalysis Data

- Format: NetCDF
    
- Typical source: ERA5
    
- Required variables:
    
    - Horizontal wind components at reference height(s) (e.g. u100, v100)
        
    - Optional: surface roughness or near-surface wind fields, depending on extrapolation method
        
- Temporal resolution: hourly (aggregated internally)
    

### S3.2 Turbine Metadata

- Format: CSV
    
- Required fields (minimum):
    
    - Turbine identifier
        
    - Latitude, longitude
        
    - Hub height [m]
        
    - Rated capacity [MW]
        
    - Turbine model or power curve identifier
        

### S3.3 Observed Generation Data

- Format: CSV
    
- Temporal resolution: daily or higher
    
- Values:
    
    - Generation [MWh] or capacity factor [-]
        
- Used exclusively for bias correction and validation
    

### S3.4 Power Curves

- Format: CSV
    
- Structure:
    
    - Wind speed bins [m s⁻¹]
        
    - Power output or capacity factor for each turbine model
        

---

## S4. Directory Structure

A typical directory layout is:

```text
PyVWF/
├── data/
│   ├── era5/
│   │   └── wind_*.nc
│   ├── turbine_info/
│   │   └── turbines.csv
│   └── observations/
│       └── generation.csv
├── input/
│   └── power_curves.csv
├── vwf/
│   ├── processing/
│   ├── bias/
│   ├── validation/
│   └── utils/
├── notebooks/
├── environment.yaml
└── README.md
```

Paths are configurable within the codebase.

---

## S5. Computational Environment

All numerical results are reproducible using the Conda environment specified in `environment.yaml`.

### S5.1 Environment Definition

```yaml
name: pyvwf
channels:
  - conda-forge
dependencies:
  - python>=3.10
  - pandas=2.2.2
  - dask=2025.11.0
  - xarray=2024.9.0
  - utm=0.7.0
  - scipy=1.14.1
  - scikit-learn=1.5.2
  - matplotlib=3.9.2
  - seaborn=0.13.2
  - openpyxl=3.1.5
  - netcdf4=1.7.1
  - bottleneck=1.4.0
```

### S5.2 Environment Creation

```bash
conda env create -f environment.yaml
conda activate pyvwf
```

---

## S6. Core Algorithms

### S6.1 Wind Speed Calculation

Horizontal wind speed is derived from reanalysis wind components as:

[  
U = \sqrt{u^2 + v^2}  
]

where (u) and (v) are the zonal and meridional wind components, respectively.

### S6.2 Hub-Height Extrapolation

Wind speeds are extrapolated from the reference height to turbine hub height using configurable vertical scaling methods (e.g. logarithmic or power-law profiles). The chosen method and parameters must be explicitly documented in any study using PyVWF.

### S6.3 Bias Correction

Bias correction is performed using **statistical correction factors** derived from observed generation data over a defined training period. The framework supports linear and regression-based approaches.

Bias correction is applied to wind speed time series prior to power conversion.

### S6.4 Wind-to-Power Conversion

Corrected wind speeds are converted to power output or capacity factor using turbine-specific power curves. No wake interaction or curtailment modelling is included.

---

## S7. Validation and Aggregation

Simulated generation can be validated against observations at:

- Individual turbine level
    
- Aggregated regional or national level
    

Validation metrics (e.g. bias, RMSE, correlation) are computed externally or via user-defined analysis scripts.

---

## S8. Assumptions and Limitations

- ERA5 spatial resolution limits turbine-level accuracy
    
- Bias correction is empirical and non-physical
    
- Wake effects and availability losses are not modelled
    
- Results are sensitive to power curve selection and observation quality
    

These limitations should be considered when interpreting results.

---

## S9. Reproducibility Statement

All results generated with PyVWF are reproducible provided:

- The same input datasets are used
    
- The Conda environment is created from `environment.yaml`
    
- Training and validation periods are explicitly documented
    

Randomness is not introduced unless explicitly enabled by the user.

---

## S10. Availability

The PyVWF source code is publicly available at:

```
https://github.com/ellyess/PyVWF
```

---

## S11. Author and Maintenance

Developed and maintained by **Ellyess Benmoufok**.

Questions, issues, or methodological discussions should be raised via the GitHub issue tracker.
