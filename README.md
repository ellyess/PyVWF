# The Python Virtual Wind Farm (PyVWF) model

---
PyVWF is a research-oriented Python framework for processing, bias-correcting, and simulating wind resources and wind power generation using reanalysis data (e.g. ERA5), turbine metadata, and observed generation data. The novelty of this model comes from the bias correction process used to improve the simulations from ERA-5. The simulated wind time-series can be both corrected and uncorrected.

`PyVWF` is a Python rewrite of the [VWF model](https://github.com/renewables-ninja/vwf/tree/master) developed by Iain Staffell. The wind energy simulations on [Renewables.ninja](https://www.renewables.ninja/) are based on the VWF model. 

---

## Overview

PyVWF supports the following workflow:

1. **Ingest wind reanalysis data** (e.g. ERA5 wind fields)
2. **Interpolate wind speeds to turbine hub height**
3. **Apply bias correction** using observed generation data
4. **Convert wind speeds to power / capacity factor** via turbine power curves
5. **Validate simulated output** against observations

The framework is intended for **daily to monthly** analysis at **turbine, regional, or national scale**.

---

## Key Features

- ERA5-based wind speed processing
- Hub-height extrapolation with configurable methods
- Statistical bias correction of wind speeds
- Power curve–based generation modelling
- Modular, research-friendly Python codebase
- Version-pinned environment for reproducibility

---

## Installation

PyVWF uses a fully pinned Conda environment to ensure reproducibility across systems.

### 1. Clone the repository

```bash
git clone https://github.com/ellyess/PyVWF.git
cd PyVWF
```

### 2. Create the environment

```bash
conda env create -f environment.yaml
conda activate pyvwf
```

### 3. Verify installation

```bash
python -c "import pandas, xarray, scipy; print('Environment OK')"
```

---

## Data Requirements

PyVWF expects the following input data types.

### Required Inputs

|Data|Format|Description|
|---|---|---|
|Reanalysis wind data|NetCDF|ERA5 wind components (e.g. u100, v100)|
|Turbine metadata|CSV|Location, capacity, hub height, turbine model|
|Observed generation|CSV|Time series of wind generation or capacity factor|
|Power curves|CSV|Wind speed to power conversion|

## Setup

### Download reanalysis wind speed data

First, download the necessary input ERA-5 data (Years in a period can be downloaded separately or together as they will be joined. Ensure training data is separate to validation):

- ECMWF's [ERA-5 reanalysis](https://cds-beta.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download), the required variables are either:
  - 100m u-component of wind, 100m v-component of wind, 10m u-component of wind and 10m v-component of wind (surface roughness is calculated instead and is more accurate).
  - 100m u-component of wind, 100m v-component of wind and Forecast surface roughness. 

### Inputting data required for the model to run

The files you should provide are:

- Observation data for all training years placed in `input/country-data/observation/`. Example files are in the repository.
- Reanalysis data for all training years and test years in `data/era5/<country>/<test/train>/`
- Turbine metadata which contains information such as the height, latitude, longitude, turbine ID, turbine model and capacity placed in `data/turb_info/`. An example is provided, plan to make this file easier to create.
- Wind turbine power curves in a .csv file with model names in each column providing the power output with respect to wind speed. Due to proprietary data used in our curve file an example of the format is shown in `input/power_curves.csv`

### Setup the PyVWF model

With the required libraries installed and input files in place, minor modifications of paths may be required.

### Basic Usage

A typical workflow consists of:

1. Loading ERA5 wind data using `xarray`
2. Mapping turbines to grid points
3. Extrapolating wind speed to hub height
4. Applying bias correction
5. Converting wind speed to power output

---

## Assumptions & Limitations

- Bias correction is **statistical**, not physical
- Accuracy depends on the quality and representativeness of observed data
- ERA5 spatial resolution may limit turbine-level accuracy
- Wake effects are not explicitly modelled
- Power curve selection strongly influences results

These assumptions should be considered when interpreting results.

---

## Reproducibility

- All dependencies are **version-pinned**
- Deterministic methods are used where possible
- Results should be reproducible across systems using `environment.yaml`

For published work, we recommend citing the repository and documenting:

- ERA5 data version
- Bias correction training period
- Power curve sources

---

## Citation

If you use PyVWF in academic work, please cite the repository.  
A `CITATION.cff` file may be added in future releases.

---

## Contributing

Contributions are welcome, especially:

- Documentation improvements
- Additional bias correction methods
- Validation case studies
- Performance optimisations

Please open an issue to discuss changes before submitting a pull request.

## CREDITS & CONTACT

The PyVWF code is developed by Ellyess F. Benmoufok. You can email me via benmoufok.ellyess@gmail.com.

The original VWF code this is based on is developed by Iain Staffell.  You can try emailing them at i.staffell@imperial.ac.uk

PyVWF is part of the [Renewables.ninja](https://renewables.ninja) project, developed by Stefan Pfenninger and Iain Staffell.  Use the [contacts page](https://www.renewables.ninja/about) there.