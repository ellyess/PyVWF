---
title: 'PyVWF: An open Python framework for bias-corrected wind power simulation from reanalysis data'
tags:
  - Python
  - wind energy
  - bias correction
  - reanalysis
  - energy system modelling
  - capacity factor
authors:
  - name: Ellyess F. Benmoufok
    orcid: 0009-0000-0337-5690
    affiliation: "1"
  - name: Simon C. Warder
    orcid: 0000-0003-0445-4280
    affiliation: "1"
  - name: Matthew D. Piggott
    orcid: 0000-0002-7526-6853
    affiliation: "1"
affiliations:
  - name: Department of Earth Science and Engineering, Imperial College London, London SW7 2AZ, UK
    index: 1
date: 6 July 2026
bibliography: paper.bib
---

# Summary

`PyVWF` (the Python Virtual Wind Farm) is an open, research-oriented framework
that converts atmospheric reanalysis data into bias-corrected wind power
generation. It re-implements, in a modular and extensible Python codebase, the
Virtual Wind Farm (VWF) methodology that underpins the wind simulations on
[Renewables.ninja](https://www.renewables.ninja) [@staffell2016; @pfenninger2016].
Given gridded reanalysis winds (e.g. ERA5 [@era5] or MERRA-2 [@merra2]), turbine
metadata, smoothed manufacturer power curves, and observed generation, `PyVWF`
extrapolates wind to hub height and interpolates to turbine locations, learns
per-cluster scale-and-offset corrections by comparing simulated and observed
capacity factors, applies these corrections to the ERA5 wind speeds, and
reconverts the corrected winds to capacity factor. Unlike API-only or
general-purpose reanalysis-to-power tools, it exposes the full **training**
workflow for the bias-correction factors and lets researchers compute them at
finer spatial and temporal resolution than the conventional national-scale
factors, through configurable spatial clustering and temporal grouping. The
package implements the granular bias-correction method of @benmoufok2024.
Experimental gridded interpolation and machine-learning extensions are
maintained on a separate development branch.

# Statement of need

Continental-scale energy system models such as `PyPSA-Eur` [@pypsaeur] depend
on wind power time series derived from reanalysis, but reanalysis-derived
capacity factors can carry biases of up to ±50 % [@staffell2016]. These biases can propagate through non-linear power
conversion and spatial aggregation and, when employed within energy system
models, lead to misleading conclusions about generation mix, transmission
investment, and system cost. The original VWF model corrects this bias but is
closed and primarily accessible only through the Renewables.ninja API. Open
tools address only part of this: general-purpose converters such as `atlite`
[@atlite] omit observation-based correction entirely, and validated global
products such as ETHOS.RESKit [@reskit] apply a correction that is national in
resolution, built on a single global wind-speed curve, and shipped as a fixed
calibration rather than a retrainable pipeline. `PyVWF` fills that specific gap
with a peer-reviewed method [@benmoufok2024] that extends the validated VWF
approach [@staffell2016] to ERA5 and to sub-national, seasonal resolution: a
transparent, reproducible research instrument for the correction step, with
distributional (not only mean-error) diagnostics, letting researchers compute,
inspect, and retrain corrections at arbitrary spatial and temporal
granularity. Research extensions on a separate development branch interpolate
corrections onto a regular grid and export them for use with
`atlite`/`PyPSA-Eur` workflows, supporting resource assessment and sensitivity
studies across scales. The framework targets energy systems researchers,
climate scientists, and power system analysts who need transparent,
reproducible, and calibrated wind resource simulations.

# Functionality

`PyVWF` supports a turbine-, regional-, or national-scale workflow. Throughout,
a *cluster* is a group of nearby turbines or grid points that share one set of
bias-correction parameters:

- **Wind processing.** Hub-height extrapolation via a log wind profile using
  ERA5 surface roughness (or roughness derived from 10 m/100 m wind shear), and
  bilinear interpolation to turbine coordinates.
- **Bias-correction training.** Per-cluster, per-time-slice multiplicative
  scalar and additive offset factors (`w_corrected = α·w + β`) learned by
  comparing simulated and observed capacity factors [@benmoufok2024], with
  user-specific spatial clustering and temporal grouping (fixed, seasonal,
  bimonthly, monthly).
- **Distributional diagnostics.** Capacity-factor histograms, empirical CDFs,
  and quantile-quantile plots via `vwf.viz` that complement conventional
  mean-error metrics.
- **Research extensions (development branch).** Interpolation of point
  corrections onto a regular ERA5 grid (nearest neighbour, IDW, RBF, kriging)
  with export to NetCDF for `atlite`/`PyPSA-Eur`, and machine-learning
  prediction of corrections from terrain and environmental features. These are
  research extensions maintained on a separate development branch.

The package ships with an automated `pytest` suite that runs on synthetic data
(no large reanalysis downloads required) and continuous integration, and a
`conda` environment for reproducibility. The granular bias-correction method
has been applied in peer-reviewed studies of European and UK wind resources
[@benmoufok2024; @wang2026].

# State of the field

The open landscape spans general-purpose converters that apply no
observation-based correction (`atlite` [@atlite]), the closed and largely
dormant VWF code behind Renewables.ninja [@staffell2016; @pfenninger2016], and
validated global products such as ETHOS.RESKit [@reskit], which pairs a
measurement-trained wind-speed correction with a national capacity-factor
calibration. RESKit is the stronger choice for a ready-made global assessment.
`PyVWF`'s contribution is complementary and peer-reviewed: it implements the
granular bias-correction method of @benmoufok2024, which extends the validated
VWF approach of @staffell2016 to ERA5 and to sub-national and seasonal
resolution, and exposes that correction's *training* step as an open,
inspectable, retrainable pipeline, unlike the national, single-global-curve,
fixed-product calibrations above.

# AI disclosure

AI-assisted tools were used to help with software refactoring, test scaffolding,
and editing portions of this manuscript for clarity. All technical content,
claims, methods, and citations were reviewed and verified by the authors, who
take full responsibility for the final software and text.

# Acknowledgements

The original VWF model was developed by Iain Staffell, and Renewables.ninja by
Stefan Pfenninger and Iain Staffell. The authors thank collaborators on the
underlying bias-correction research [@benmoufok2024].

# References