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
date: 17 July 2026
bibliography: paper.bib
---

# Summary

`PyVWF` (the Python Virtual Wind Farm) is an open-source Python framework
that converts atmospheric reanalysis data into bias-corrected wind power
generation. It re-implements, in a modular and extensible Python codebase, the
Virtual Wind Farm (VWF) methodology that underpins the wind simulations on
[Renewables.ninja](https://www.renewables.ninja) [@staffell2016; @pfenninger2016].
Given gridded reanalysis winds (ERA5 [@era5]; the original VWF used
MERRA-2 [@merra2]), turbine metadata, smoothed power curves, and observed
generation, `PyVWF`
extrapolates wind to hub height and interpolates to turbine locations, learns
per-cluster scale-and-offset corrections by comparing simulated and observed
capacity factors, applies these corrections to the ERA5 wind speeds, and
reconverts the corrected winds to capacity factor. Unlike API-only or
general-purpose reanalysis-to-power tools, it exposes the full **training**
workflow for the bias-correction factors and lets researchers compute them at
finer spatial and temporal resolution than the conventional national-scale
factors, through configurable spatial clustering and temporal grouping. The
package implements the granular bias-correction method of @benmoufok2024. A
redistributable open power-curve library [@turbinemodels] ships with the
package (see *Functionality*), so the workflow runs out of the box; users
supply their own curve library for manufacturer-specific production work.
Experimental gridded interpolation and machine-learning extensions are
maintained on a separate development branch.

# Statement of need

Continental-scale energy system models such as `PyPSA-Eur` [@pypsaeur] depend
on wind power time series derived from reanalysis, but reanalysis-derived
capacity factors can carry biases of up to ±50 % [@staffell2016]. These biases can propagate through non-linear power
conversion and spatial aggregation and, when employed within energy system
models, lead to misleading conclusions about generation mix, transmission
investment, and system cost. The original VWF model corrects this bias, but
its correction is accessible only through the closed Renewables.ninja API and
cannot be inspected or retrained. Existing *open* tools cover only part of the
problem: general-purpose converters such as `atlite` [@atlite] omit
observation-based correction entirely, while validated global products such as
ETHOS.RESKit [@reskit] ship a fixed national-resolution calibration built on a
single global wind-speed curve, not a retrainable pipeline. `PyVWF` fills this
gap. It implements a peer-reviewed method
[@benmoufok2024] that extends the validated VWF approach [@staffell2016] to
ERA5 and to sub-national, seasonal resolution, and it reports distributional
diagnostics in addition to mean error. Corrections can be computed at whatever
spatial and temporal granularity the observation record supports, then
inspected and retrained as that record grows.
Research extensions on a separate development branch interpolate
corrections onto a regular grid and export them for use with
`atlite`/`PyPSA-Eur` workflows, supporting resource assessment and sensitivity
studies across scales. The framework is intended for researchers in energy
systems and climate science, and for power system analysts, who need calibrated
wind resource simulations whose correction step they can reproduce and examine.

# Functionality

`PyVWF` supports a turbine-, regional-, or national-scale workflow, in which a
*cluster* is a group of nearby turbines or grid points sharing one set of
bias-correction parameters. Wind is extrapolated to hub height via a log wind
profile using ERA5 surface roughness (or roughness derived from 10 m/100 m wind
shear) and bilinearly interpolated to turbine locations. Per-cluster,
per-time-slice scale-and-offset factors (`w_corrected = α·w + β`) are then
learned by comparing simulated and observed capacity factors [@benmoufok2024],
with user-configurable spatial clustering and temporal grouping (fixed,
seasonal, bimonthly, monthly). Observed generation and site metadata are
supplied through `ObservationSource` adapters resolved via a registry, so a new
region is added by writing an adapter rather than editing the pipeline.
Turbine-level adapters (code, not data) for Denmark, Germany, and the United
Kingdom ship with the package; the observation datasets themselves are
user-supplied. The `vwf.viz` module is designed so that a learned correction
can be examined directly rather than applied unseen. It provides distributional
diagnostics (capacity-factor histograms, empirical CDFs, and quantile-quantile
plots), maps of the learned factors, and error-versus-cluster-count curves for
hyperparameter selection.

A redistributable open power-curve library of 69 real machines and 7 composites
from the NREL/turbine-models archive [@turbinemodels] (BSD-3-Clause), smoothed
by an independent reproduction of the VWF method [@staffell2016], ships with the
package, so an automated `pytest` suite runs the full workflow on synthetic data
in under a minute with no reanalysis download. The granular bias-correction
method has been applied in peer-reviewed studies of European and UK wind
resources [@benmoufok2024; @wang2026].

# State of the field

Open tools in this area include the general-purpose converter `atlite`
[@atlite], which applies no observation-based correction, and validated global
products such as ETHOS.RESKit [@reskit], which pairs a measurement-trained
wind-speed correction with a national capacity-factor calibration; the VWF code
behind Renewables.ninja [@staffell2016; @pfenninger2016] remains closed and
largely dormant. For a ready-made global assessment, RESKit is the stronger
choice. `PyVWF` is complementary. It implements the peer-reviewed granular
bias-correction method of @benmoufok2024, which adapts the VWF approach of
@staffell2016 to ERA5 and refines it to sub-national and seasonal resolution,
and it exposes the training step of that correction as a pipeline the user can
inspect and retrain; the calibrations above ship as fixed national products.

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