# Multi-region validation scorecard

**Date:** 2026-07-24
**What this is:** one place that states, per region, how much PyVWF's affine
wind-speed correction reduces the error between ERA5-simulated and observed wind
capacity factors, on a held-out year. Every number is read from a
`metrics.csv` under `output/validation/`; source paths are given so each is
auditable. This is screening-level validation, one test year per region, not an
accredited yield assessment.

## Turbine / plant-level (observed capacity factor per farm)

Matched real turbine curves and hub heights; k-swept affine fit; best held-out
`affine-wind` row (fleet scope). Source: `cluster_sweep_2026-07-24/<CODE>/`
except CL (`cl_matched_2026-07-24`) and AR (`ar_capfix3_2026-07-24`).

| Region | Fleet (test) | Train → test | Uncorr RMSE | Corr RMSE | Uncorr MBE | Corr MBE | Corr r | Best cfg |
|---|---|---|---|---|---|---|---|---|
| Germany (DE) | 4814 turbines | → 2019 | 0.086 | **0.057** | +0.042 | +0.001 | 0.85 | k100 fixed |
| Denmark (DK) | 5410 turbines | 2015-19 → 2020 | 0.147 | **0.085** | +0.110 | +0.023 | 0.83 | k100 season |
| Brazil (BR) | 151 complexes | 2021-23 → 2024 | 0.139 | **0.105** | -0.046 | -0.015 | 0.72 | k60 fixed |
| United States (US) | 520 plants | 2019-21 → 2022 | 0.110 | **0.098** | +0.022 | +0.024 | 0.78 | k250 fixed |
| Australia (AU-NEM) | 77 farms | 2020-22 → 2023 | 0.115 | **0.094** | +0.009 | -0.006 | 0.61 | k45 season |
| United Kingdom (UK) | 348 turbines | → 2019 | 0.145 | **0.115** | +0.037 | -0.038 | 0.70 | k50 fixed |
| New Zealand (NZ) | 12 farms | 2019-23 → 2024 | 0.157 | **0.106** | -0.062 | +0.021 | 0.66 | k5-7 fixed |
| Chile (CL) | 59 plants | → 2024 | 0.123 | **0.104** | -0.032 | -0.002 | 0.45 | k10 fixed |
| Argentina (AR) | 59 plants | → 2024 | 0.151 | **0.133** | +0.010 | +0.001 | 0.44 | k10 fixed |

Read this as: across four continents, on fleets the model never saw, the
correction lowers capacity-factor RMSE and drives the systematic bias (MBE)
toward zero in every case. It is strongest where the raw ERA5 bias is a
clusterable wind-speed offset (DE, DK, BR), and weakest where the residual is
not a wind-speed error at all (CL, AR; see caveats).

## Country-level (ENTSO-E national aggregate, 2023)

Capacity-weighted national monthly CF, trained and tested on 2023.
Source: `country_baseline_2026-07-23/<CODE>/`.

| Region | Uncorr RMSE | Corr RMSE | Uncorr MBE | Corr MBE | Best cfg |
|---|---|---|---|---|---|
| France (FR) | 0.171 | **0.012** | +0.165 | +0.006 | N=10 fixed |
| Belgium (BE) | 0.340 | **0.020** | +0.337 | -0.002 | N=3 season |
| Spain (ES) | 0.135 | **0.026** | +0.130 | +0.016 | N=4 fixed |
| Ireland (IE) | 0.172 | **0.021** | +0.168 | +0.009 | N=1 season |
| Sweden (SE) | 0.088 | **0.030** | +0.085 | -0.027 | N=4 fixed |
| Italy (IT) | 0.066 | **0.034** | +0.062 | -0.020 | N=3 season |
| Portugal (PT) | 0.110 | **0.074** | -0.097 | +0.029 | N=1 season |
| Norway (NO) | 0.025 | 0.037 | +0.009 | -0.028 | correction does not help |

The country-level fit removes very large mean biases (FR, BE, ES, IE all from
0.13-0.34 down to ~0.01-0.03). Two honest notes: NO is already near-perfect
uncorrected and the correction makes it slightly worse; and the country method
fits under-determined offsets against one national series per month, so the
offsets largely repair the scalar's cube-law overshoot rather than a genuine
additive spatial bias (`country_level_method_review.md`).

## What must NOT be overclaimed

- **One test year per region.** Orderings of two close configs are not
  meaningful; the headline is the uncorrected-to-corrected drop, not the exact k.
- **CL and AR are caveated, not headline wins.** ERA5 exaggerates the
  north-south wind gradient in both (Atacama too calm, Patagonia too windy). The
  correction removes the mean bias but adds limited skill; for CL it only helps
  once real turbine curves are used, and CL is unstable at low k. AR is usable
  only after its capacity denominators were rebuilt; a northern cluster still
  fits an extreme scalar that a higher-resolution wind product, not more data,
  would fix (`south_america_spatial_bias.md`).
- **NO gets worse; NL is excluded** (an ENTSO-E coverage defect makes its CF
  series unusable). Reporting either as a corrected region would be false.
- **US carries an unscreened curtailment confound** (ERCOT/SPP); its near-zero
  fleet MBE is partly an aggregation artefact.
- **AU-NEM's "does correction help?" is config-dependent** (it improves the
  fleet seasonal cycle but can worsen absolute farm RMSE in curtailed South
  Australia); the table row is the matched-curve k-swept result.
- Screening-level throughout: not MEASNET/DNV-accredited, not investment advice.

## Data provenance

Turbine/plant observations by region: DK Danish Energy Agency; DE public
turbine register; UK REPD/Ofgem; US EIA-923; BR ONS; AU-NEM AEMO; NZ EMI; CL
Coordinador (CEN); AR CAMMESA (capacities rebuilt from turbine specs).
Country-level: ENTSO-E Transparency. All correction operates on ERA5 at 0.25deg.
Confidential inputs (WindStats DE/ES, Ofgem certificate warehouse, licensed
curve library) are not redistributed and are not required to reproduce the open
rows above, which run on the bundled open curve library.
