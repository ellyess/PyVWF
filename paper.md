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

With climate change having such a significant impact on the world, a recent pledge at COP 28 to triple the world's installed renewable energy capacity by 2030 has been called. Among these, wind energy is abundant and widely distributed; the global installed wind power capacity has increased more than four-fold from 181 GW in 2010 to 899 GW in 2022 [@irena2023]. With the expected rapid uptake of new wind farms, it is crucial to understand the wind power output accurately. Unlike traditional energy sources, wind is inherently stochastic and non-stationary, and there is a need for accurate models and tools to predict the power output from the wind resource.

`PyVWF` is an open Python software package designed to make scientific-quality weather and energy data available to a wider community.  It takes historical weather data and converts it into potential wind power output at requested locations that correspond with wind turbine placement. The wind speeds are bias corrected to ensure a higher accuracy than directly converting the historical weather data. 

# Statement of need

A popular product used to produce simulations of the hourly power output from wind and solar power plants located anywhere in the world is Renewables.Ninja (RN) [@staffell2016, pfenninger2016]. In the case of wind power, atmospheric flow speeds are converted into power output using the `vwf` (Virtual Wind Farm) model [@staffell2016].

Currently, most studies use reanalysis data (particularly ERA-5 [@era5] or MERRA-2 [@merra2]) [@kiss2009,kubik2013] to simulate wind power for energy system modelling. However, it has been found that the direct simulation outputs of capacity factors derived from reanalysis data can suffer from up to ± 50\% of bias [@staffell2016].

`vwf` calculates hourly power output at any given location using reanalysis data from MERRA-2. A key novelty of this model is the bias correction process that it employs which is capable of achieving a R-squared score of 0.95 for the national scale hourly power output of 23 European countries [@staffell2016]. The bias correction is performed using factors on a country-wide basis. It has already been used in several scientific publications [@pennock2023, @zhang2023, @fliegner2022] showing there is a significant importance to ensuring this model is maintained and improved.

The `vwf` code exhibits challenges to use locally and, thus is primarily accessed via the API, Renewables.Ninja[@staffell2016, pfenninger2016]. Noting this factor, the development of similar models such as `atlite` [@atlite] which provide easier usage for non-commercial use.

The `PyVWF` model is a complete rework of the `vwf` model, written to make non-commercial usage more accessible while keeping the novel bias correction that similar models do not incorporate. It is written in a streamlined workflow using Python, to provide an open and extensible library for further development. Python has an abundance of packages (e.g. `xarray`, `dask` and `geopandas`) that are desirable for easy processing of the weather reanalysis data and the potential to be parallelised to perform well on even large datasets.

`PyVWF` provides accessibility to the training process for the bias correction factors on top of the base simulation function that uses already derived bias correction factors. The capability of the code has been expanded to use ERA-5 reanalysis data providing an improved resolution of the reanalysis data from 0.5° x 0.625° to  0.25° x 0.25°. A notable addition possible from these changes is the bias correction factors can be calculated at finer spatial resolutions than the existing national scale factors, through the use of spatial clustering. Another notable implementation of this release is the option to add time dependency to the calculation of the bias correction factors to more accurately correct seasonal trends.

# Core Functionality

## Simulating Capacity Factor
- Obtain wind speed data: Acquire surface roughness $z_{0}$, eastward $u$ and northward $v$ wind speed components at 100m above ground at each ERA-5 grid point. 
- Derive wind speed magnitude: Derive wind speed magnitude at 100m, $w_{100m}$,  from the $u$ and $v$ wind components at each grid point.
- Height extrapolation: Derive the wind speed at hub height assuming the log wind profile [@holmesWindLoading2017]:
- $$w_{sim}(h)=w_{100m}\frac{\ln(h / z_{0})}{\ln(100 / z_{0})},$$
    where wind speed $w_{\textnormal{sim}}$ is a function of the height $h$ and the surface roughness $z_{0}$.
- Location interpolation: Linearly interpolate speeds to the specific geographic coordinates of each wind turbine, using Python's `xarray.DataArray.interp`.
- Convert wind speed to capacity factor: Using smoothed manufacturers' power curves, convert wind speed $w_{sim}$ to capacity factors $CF_{sim}$.

## Bias Correction 
Bias correction is applied on wind speeds from the reanalysis data as we assume this is where the error comes from rather than the power conversion. Wind speeds are corrected using the following scheme adapted from [@staffell2016]:
$$w_{corrected} = \alpha w_{uncorrected} + \beta,$$
where $w_{\textnormal{uncorrected}}$ is the uncorrected wind speed from the ERA-5 renalysis data. The training process to find the multiplicative scalar $\alpha$ and the linear offset $\beta$, involves the following steps:
- Compute the simulated CF $CF_{sim}$ using $w_{uncorrected}$ and calculate the error factor $\varepsilon_{CF}$ from the mean (calculated over desired spatiotemporal resolution) $CF_{sim}$ and observed CF $CF_{obs}$ via $\varepsilon_{cf}=\frac{CF_{obs}}{CF_{sim}}$;
- Derive the multiplicative scalar using the bias factor (note [@staffellUsingBiascorrected2016] used $\alpha=0.6\varepsilon_{CF}+0.2$ calculated per country to allow for generalisation of the method to be applied to other locations where only the long-run mean observed CF is known and reduce over-fitting. This is based on the observation of reanalysis tending to overestimate variability in wind. This has been modified for this study as we are deriving $\alpha$ at varying spatial resolutions and are assessing the performance without generalising.): $\alpha=\varepsilon_{cf}$;
- Perform an iterative process on \ref{eq3} to find $\beta$ such that $CF_{sim}=CF_{obs}$.


# Acknowledgements
