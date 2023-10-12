title: 'PyVWF: A Bias Corrected Wind Power Simulator'
tags:
  - Python
  - energy system model
  - wind
authors:
  - name: Ellyess F. Benmoufok
    equal-contrib: true
    affiliation: "1"
affiliations:
 - name: Department of Earth Science and Engineering, Imperial College London, London SW7 2AZ, UK
   index: 1
date: 22 September 2023
bibliography: paper.bib

---

# Summary

With climate change having such a significant impact on the world, countries are turning to renewable energy resources. Among the renewable energy resources, wind energy is abundant and widely distributed; the global installed wind power capacity has increased greater than four-fold from 180.9 GW in 2010 to 823.5GW in 2021 [@irena2022]. Unlike traditional energy sources, wind is inherently stochastic and non-stationary, and there is a need for accurate models and tools to predict the wind resource power output.

`PyVWF` is an open Python software package designed to make scientific-quality weather and energy data available to a wider community.  It takes historical weather data and converts it into potential wind power output at given points for wind turbine placement. The wind speeds are bias-corrected to ensure a higher accuracy than directly converting the historical weather data. 

# Statement of need

A popular product used to run simulations of the hourly power output from wind and solar power plants located anywhere in the world is renewables.ninja (RN) [@staffell2016, pfenninger2016]. Wind speeds are converted into power output using the `vwf` (Virtual Wind Farm) model by Iain Staffell [@staffell2016].

Currently, most studies use reanalysis data (particularly ERA-5 [@era5] or MERRA-2 [@merra2]) [@kiss2009,kubik2013] to simulate wind power for energy system modeling. However, it has been found that the direct simulation outputs from reanalysis data suffer from up to ± 50\% of bias [@staffell2016].

`vwf` calculates hourly power output at any given location in the world using reanalysis data from MERRA-2. The novelty of this model is the bias-correction process, which is capable of achieving a R-squared score of 0.95 for national hourly power output of 23 European countries [@staffell2016]. The bias correction is done using factors on a country-wide basis. It has already been used in a number of scientific publications [@pennock2023, @zhang2023, @fliegner2022] showing there is a significant importance to ensure this model is maintained and improved.

The `vwf` code is difficult to use locally, thus is primarily used through the API, renewables.ninja [@staffell2016, pfenninger2016]. This factor led to the development of similar models such as `atlite` [@atlite] which provide easier use for non-commercial use.

`PyVWF` is a complete rework of the `vwf` model, written to make non-commercial usage more accessible while keeping the novel bias-correction which the similar models didn't implement. It is written in a streamline workflow using Python, to provide an open and extensible library for further development. Python has an abundance of packages (e.g. `xarray`, `dask` and `geopandas`) that are desirable for easy processing of the reanalysis data and the potential to parallelize to perform well on even large datasets.

In `PyVWF` the capability of the code has been expanded to use ERA-5 reanalysis data. A notable addition from this change is the improved resolution of the data allows for the bias correction factors to be calculated at a finer spatial resolution than country-wide. Another notable addition to this release is the bias correction factors are now calculated on a monthly basis, allowing for bi-monthly and seasonal factors to more accurately bias correct. 


# Acknowledgements
