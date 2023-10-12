# The Python Virtual Wind Farm (PyVWF) model
---
`PyVWF` is a Python rewrite of the [VWF model](https://github.com/renewables-ninja/vwf/tree/master) developed by Iain Staffell. The wind energy simulations on [Renewables.ninja](https://www.renewables.ninja/) are based on the VWF model.

---

## Functionality
This model has the ability to produce bias corrected power output from wind farms based on reanalysis data. The model calculates wind speed with this equation:
- `w(h) = v(100m) * ln(h/z0)/ln(100m/z0), where z0 is surface roughness`


## Installation:
* Clone the repository from Github:
  * Using command line:
  `git clone https://github.com/EllyessB/PyVWF.git`
  * Downloading the repository as a .zip
* Installing all the requirements by using command line:
 `pip install -r requirement.txt`


## Setup
### Download reanalysis wind speed data
First, download the necessary input ERA-5 data:
- ECMWF's [ERA-5 reanalysis](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form), the required variables are: the 100m u-component of wind, 100m v-component of wind and Forecast surface roughness.

### Inputing data required for model to run
The files you should provide are:
- Observation data for all training years placed in `data/observation`. Example files are in the respository.
- Reanalysis data for all training years and test years in `data/reanalysis/<set_used>/`
- Turbine metadata which contains information such as the height, latitude, longitude, turbine ID, turbine model and capacity placed in `data/turb_info/`. An example is provided, plan to make this file easier to create.
- Wind turbine power curves in a .csv file with model names in each column providing the power output with respect to wind speed. This exists already in `data/turb_info/`, extra curves can be added, these are found through the manufaturer.

### Setup the VWF model
With the required libraries installed and input files inplace, minor modification of paths may be required.

## Usage
Open `run.py` and edit the parameters you wish to use and save. 

Run the file using `python run.py`, this should work if all paths and input files are inplace. Run time will vary depending on number of clusters required to train and size of the area etc. If model has been trained for a number of clusters, running this again will skip the training.


## CREDITS & CONTACT

The PyVWF code is devloped by Ellyess F. Benmoufok. You can email me via benmoufok.ellyess@gmail.com.

The original VWF code this is based on is developed by Iain Staffell.  You can try emailing them at i.staffell@imperial.ac.uk

PyVWF is part of the [Renewables.ninja](https://renewables.ninja) project, developed by Stefan Pfenninger and Iain Staffell.  Use the [contacts page](https://www.renewables.ninja/about) there.

