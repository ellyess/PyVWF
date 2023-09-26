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

With climate change having such a significant impact on the world, countries are turning to renewable energy resources. Among the renewable energy resources, wind energy is abundant and widely distributed; the global installed wind power capacity has increased greater than four-fold from 180.9 GW in 2010 to 823.5GW in 2021. Unlike traditional energy sources, wind is inherently stochastic and non-stationary, and there is a need for accurate models and tools to predict the wind resource power output.

Renewables.Ninja (RN) allows you to run simulations of the hourly power output from wind and solar power plants located anywhere in the world. Wind speeds are converted into power output using the `vwf` (Virtual Wind Farm) model by Iain Staffell.

`vwf` calculates hourly power output at any given location in the world using reanalysis data from MERRA-2. The novelty of this model is the bias-correction process, that is capable of achieving a R-squared score of 0.95 for national hourly power output of 23 European countries. The bias correction is done using factors on a country-wide basis.



# Statement of need

`PyVWF` is a Python package for generating bias corrected wind power output at a given location using reanalysis data. The `vwf` model was designed to make scientific-quality weather and energy data available to a wider community. A key use has been as an input into energy system models, such as `TIMES` and `PyPSA`.

In `PyVWF` the capability of the code has been expanded to use ERA-5 reanalysis data, as well as MERRA-2. A notable addition from this change is the improved resolution of the data allows for the bias correction factors to be calculated at a finer spacial resolution than country-wide. Another notable addition to this release is the bias correction factors are now calculated on a monthly basis, allowing for bi-monthly and seasonal factors to more accurately bias correct. The `vwf` package has been completely reworked into a streamline workflow using Python, orginally written in R. This was to enable further improvements and functionality as much of the open-source research in this field is done in Python (PyPSA etc)

