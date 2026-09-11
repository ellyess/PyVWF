# Multi-region validation scorecard

Per region, how much PyVWF's affine wind-speed correction reduces the error
between ERA5-simulated and observed capacity factors on a held-out year. Every
number is read from a `metrics.csv` under `output/validation/`, with the source
path given so each is auditable. Screening-level validation, one test year per
region, not an accredited yield assessment.

All rows were produced by PyVWF v0.4.0 at commit `41462e9` from a clean tree on
2026-08-24, one region per process. Runs are in
`output/validation/refresh_2026-08-24/<CODE>/`, outside the repository. Each row
was run from the single-configuration file committed under
`configs/regions/scorecard/` (`<code>_k<N>.toml` or `<code>_country.toml`), which
fixes the cluster count and time slice the row reports. For the eight
country-level regions that file is identical to the maintained
`configs/regions/<code>.toml`. For the nine turbine-level regions it is not: the
maintained configs carry a cluster sweep, and for seven of the nine (all but DK
and UK) that sweep does not contain the reported cluster count, so re-running the
maintained config does not reproduce the row.

Each evaluate manifest's `trained_from` records a temporary path that no longer
exists. The link to `train-refresh/` was verified on 2026-09-11: re-running every
evaluation at the same commit, against the surviving training directories,
reproduced all seventeen `metrics.csv` files byte for byte.

Seven rows (DE, DK, UK, US, BR, AU-NEM, NZ) were run with a licensed curve
library that is not redistributable, identified by sha256 in each manifest.
Four of them (DE, DK, UK, US) simulate 74% to 90% of their capacity on curves
only that library contains, and are not reproducible by a third party. The
other three (BR, AU-NEM, NZ) simulate only on curves the open library also
contains: re-run on the open library on 2026-09-11, each reproduced its
`metrics.csv` byte for byte (`output/validation/open_library_check_2026-09-11/`).
The remaining ten (CL, AR and the eight country-level regions) were run on the
bundled open library. The country-level grid points name Vestas models that the
open library does not contain, so every unit in those eight rows fell back to a
single default curve, the open library's first column:
`2019COE_DW100_100kW_27.6`, a 100 kW distributed-wind turbine.

**Other brand** is the share of each row's fitted training fleet, by capacity,
simulated on another manufacturer's curve. **Reference curve** is the share on a
research reference design or generic composite curve, which is never the unit's
own machine; on the open library it is often the only kind available, so read
it as a fact about the library rather than about the matching. **Unverifiable**
is the share where either side cannot be identified, so the match cannot be
checked in either direction. Unit counts and the largest mismatched pairs are in
`output/validation/curve_resolution_backfill_2026-09-11/cross_manufacturer_audit.csv`,
produced by `scripts/analysis/curve_match_audit.py`.

Every turbine-level row except BR is matched on specific power, by one of three
routes:

- **DE, DK and UK:** `add_models` at load time, a fuzzy manufacturer match then
  nearest specific power.
- **US, NZ, CL and AR:** `assign_curves_from_library` at processing time,
  nearest specific power within a rating band, with no manufacturer step.
- **AU-NEM:** a specific-power class.

BR assigns one uniform curve to every complex, and nothing records its
manufacturers, so its match shares are n/a rather than zero.

The two mismatch columns therefore measure how far each row rests on the
assumption that specific power fixes a curve's shape. Whether held-out skill
survives that assumption is not assessed here. Largest case: 13.4% of DE
capacity is Vestas turbines on Gamesa curves.

The rule is strict and names brands, not lineages. A Bonus turbine on a
Siemens curve counts as other brand. A GE plant on the DOE reference curve of a
GE 1.5 MW machine counts as a reference curve; that is 6.6% of US capacity.

Every model key in every turbine-level row has a curve in its
`power_curves.csv`, so nothing there was substituted. That is a different table
from `models.csv`, which the audit reads for each model's manufacturer: 11 AU-NEM
farms (14.3% of capacity) carry keys with a curve in `power_curves.csv` and no
row in `models.csv`, because the open library's normalized composites are
curve-only by design; the audit identifies them from the open library's
provenance file as reference curves. The country-level table's **Substituted**
column is the share of the evaluated fleet whose model key has no curve in
`power_curves.csv` at all. Per-row records are in each row's
`curve_resolution.csv` under the same folder.

## Turbine / plant-level (observed capacity factor per farm)

Matched real turbine curves and hub heights; k-swept affine fit; best held-out
`affine-wind` row, fleet scope.

| Region | Fleet (test) | Train → test | Uncorr RMSE | Corr RMSE | Uncorr MBE | Corr MBE | Corr r | Best cfg | Other brand | Reference curve | Unverifiable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Germany (DE) | 4814 turbines | 2015-18 → 2019 | 0.086 | **0.057** | +0.042 | +0.001 | 0.85 | k100 fixed | 40.0% | 8.9% | 0.0% |
| Denmark (DK) | 5410 turbines | 2015-19 → 2020 | 0.147 | **0.085** | +0.110 | +0.023 | 0.83 | k100 season | 11.6% | 0.6% | 23.0% |
| Brazil (BR) | 151 complexes | 2021-23 → 2024 | 0.139 | **0.105** | -0.046 | -0.015 | 0.72 | k60 fixed † | n/a | n/a | 100.0% |
| United States (US) | 520 plants | 2019-21 → 2022 | 0.110 | **0.097** | +0.022 | +0.024 | 0.79 | k250 fixed † | 48.3% | 22.0% | 1.1% |
| Australia (AU-NEM) | 77 farms | 2020-22 → 2023 | 0.115 | **0.094** | +0.009 | -0.006 | 0.61 | k45 season | 2.8% | 84.5% | 4.5% |
| United Kingdom (UK) | 348 turbines | 2015-18 → 2019 | 0.145 | **0.115** | +0.037 | -0.038 | 0.70 | k50 fixed | 21.8% | 7.5% | 0.0% |
| New Zealand (NZ) | 12 farms | 2019-23 → 2024 | 0.157 | **0.106** | -0.062 | +0.021 | 0.66 | k7 fixed | 41.9% | 47.3% | 0.0% |
| Chile (CL) | 59 plants | 2021-23 → 2024 | 0.123 | **0.105** | -0.026 | -0.002 | 0.43 | k10 fixed † | 3.5% | 91.6% | 0.0% |
| Argentina (AR) | 59 plants | 2021-23 → 2024 | 0.151 | **0.133** | +0.010 | +0.001 | 0.44 | k10 fixed † | 0.2% | 96.7% | 0.0% |

**† The fit behind this row is degenerate.** `fit_quality` run against the exact
factors file each row reports, with the calibrated bounds (scalar in 0.2 to 3.0,
offsets required to converge):

| Region | Config | Max scalar | Implausible scalars | Failed offsets |
|---|---|---|---|---|
| Chile (CL) | k10 fixed | **80.23** | 3 | **1** |
| United States (US) | k250 fixed | **46.39** | 5 | 0 |
| Argentina (AR) | k10 fixed | **15.53** | 1 | 0 |
| Brazil (BR) | k60 fixed | **4.82** | 2 | 0 |

The other five are clean: DE 2.79, AU-NEM 2.64, UK 1.86, NZ 1.81, DK 1.15, all
inside the ceiling with no failed offsets, as is every country-level fit below.
These figures travel in `metrics.csv` automatically, so a future run cannot hide
them.

Chile is the worst and is documented in `method-scalar-bounds.md`, which
shows no `min_cluster_size` setting satisfies all three of its gates: raising it
to 3 clears the failed offset and still beats uncorrected (0.1069 against
0.1226) but leaves a scalar of 39.3, and raising it to 5 collapses the
correction entirely (0.2366 against an uncorrected 0.1226). The same document
reports the exported Chile field flagging 717 of 2,697 grid cells, 26.6%, as
degenerate. A corrected RMSE of 0.104 that contains a wind scalar of 80 is not a
result to quote without this context.

Read this as: across four continents, on fleets the model never saw, the
correction lowers capacity-factor RMSE and drives the systematic bias (MBE)
toward zero in every case. It is strongest where the raw ERA5 bias is a
clusterable wind-speed offset (DE, DK, BR), and weakest where the residual is
not a wind-speed error at all (CL, AR; see caveats).

That reading holds for the five clean rows. For the four daggered ones the
aggregate improved while individual clusters did not: an RMSE that improves
while the fit contains a scalar of 80 is telling you the fleet average absorbed
the damage, not that the correction is sound everywhere it was applied. The
per-cluster factors from those four rows should not be used, and the gridded
exports built from them carry a `degenerate` layer for exactly this reason.

## Country-level (ENTSO-E national aggregate, 2023)

Capacity-weighted national monthly CF, held-out 2023, bundled open curve library
throughout.

| Region | Uncorr RMSE | Corr RMSE | Uncorr MBE | Corr MBE | Best cfg | Substituted |
|---|---|---|---|---|---|---|
| France (FR) | 0.171 | **0.012** | +0.165 | +0.006 | N=10 fixed | 100% |
| Belgium (BE) | 0.340 | **0.020** | +0.337 | -0.002 | N=3 season | 100% |
| Spain (ES) | 0.135 | **0.026** | +0.130 | +0.016 | N=4 fixed | 100% |
| Ireland (IE) | 0.172 | **0.021** | +0.168 | +0.009 | N=1 season | 100% |
| Sweden (SE) | 0.088 | **0.030** | +0.084 | -0.027 | N=4 fixed | 100% |
| Italy (IT) | 0.066 | **0.034** | +0.062 | -0.020 | N=3 season | 100% |
| Portugal (PT) | 0.110 | **0.074** | -0.097 | +0.029 | N=1 season | 100% |
| Norway (NO) | 0.034 | 0.039 | +0.024 | -0.030 | correction does not help | 100% |

The country-level fit removes very large mean biases (FR, BE, ES, IE all from
0.13-0.34 down to ~0.01-0.03). Two honest notes: NO is already close to
unbiased uncorrected (RMSE 0.034) and the correction makes it worse (0.039); and the country method
fits under-determined offsets against one national series per month, so the
offsets largely repair the scalar's cube-law overshoot rather than a genuine
additive spatial bias (`method-country-level.md`).

## What must NOT be overclaimed

- **Four of the nine turbine-level rows rest on degenerate fits** (CL, US, AR,
  BR). The aggregate metric is real; the underlying per-cluster factors are not
  usable. Chile carries a fitted wind scalar of 80.23 with one offset that never
  converged, and the United States carries 46.39. Neither is visible in the
  skill metric, which is the point.
- **An aggregate that barely moves does not mean the fit did not move.** A
  change of fitting code moved the United States' headline RMSE by 0.0005 while
  more than doubling its worst fitted scalar, from 20.54 to 46.39. Check
  `fit_quality`, not the metric, when judging whether a correction is sound.
- **One test year per region.** Orderings of two close configs are not
  meaningful; the headline is the uncorrected-to-corrected drop, not the exact k.
- **CL and AR are caveated, not headline wins.** ERA5 exaggerates the
  north-south wind gradient in both (Atacama too calm, Patagonia too windy). The
  correction removes the mean bias but adds limited skill; for CL it only helps
  once real turbine curves are used, and CL is unstable at low k. The k=10 row
  reported above is itself degenerate, so "use k=1 or k>=8" is not sufficient
  guidance: check `fit_quality` rather than the cluster count. AR is usable
  only after its capacity denominators were rebuilt; a northern cluster still
  fits an extreme scalar that a higher-resolution wind product, not more data,
  would fix (`region-south-america.md`).
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
