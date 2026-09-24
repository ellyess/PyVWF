# Australia/NEM synthesis: what generalises, what doesn't, and what a global method needs

**Reproduction record, added 2026-09-18.** Drivers:
`scripts/studies/method-generalisation/export_au_grid_netcdf.py`,
`scripts/studies/method-generalisation/au_nem_validation.ipynb`. Until
2026-09-18 they were in `examples/notebooks/` and `scripts/region_tools/`, the
paths any command below uses; `scripts/studies/README.md` maps each old path to
its new one. Numbers: the gridded export,
`output/validation/AU-NEM/au_nem_grid.nc`, records commit `8da53a6` in its
attributes. The notebook records none: its outputs are stripped, and the
harness runs it makes write their own manifests. The notebook's bundled inputs
moved with it, to `data/` beside it.

**Date:** 2026-07-16
**Scope:** what the Australia/NEM validation says generalises across regions, what does
not, and what a global method would need. The closing document for that work.

**Correction notice, 2026-09-24: every corrected Australia/NEM figure in this
document is withdrawn, including the DK → AU transfer.** `correct_wind_speed`
rebuilt the turbine axis in sorted ID order and then attached each unit's model
key and capacity by position, in the fleet's own order. A fleet whose IDs were
not already sorted, and which used more than one model key, therefore ran its
corrected simulation on other units' curves and capacities. Fixed in
`a94670c`. A transfer applies its collapsed factors through the same function,
on the target fleet.

**Withdrawn:**

- the Australia/NEM column's corrected figures in the central table (absolute
  RMSE 0.104 → 0.120, cycle tracking 0.072 → 0.065), which come from
  `region-au-nem.md`, whose notice of the same date withdraws them;
- the Australia half of the central result, which rests on those figures;
- the DK → AU transfer (+32% RMSE, MBE −0.024 → −0.093), because its target
  fleet is AU-NEM, unsorted on 11 model keys;
- the statements that rest on the withdrawn Australia figures: the −10.9% and
  −18.3% under *The evidence discipline*, the verdict under *Dual-stack
  robustness*, the false-in-AU statement under *"Corrected = improved"*, and
  seasonal factors beating fixed ones on cycle tracking in Australia under
  *Open questions*.

Each is marked where it stands. The corrected values are not known, because
the runs behind this document were not retained.

**Standing:** the Denmark column, whose fleet `add_models` sorted by ID
before any simulation, in the code this document ran on as now; the AU → DK
transfer, whose target fleet is DK; the collapsed scalars of both transfers,
which training produces before the defective step; every uncorrected figure;
the regression validation, region-as-config, the ingest pattern and the
uniform-curve amplitude check, none of which uses a corrected simulation of an
unsorted multi-key fleet. The gridded export `au_nem_grid.nc` applies its
factors on the grid and does not call `correct_wind_speed`; the validation note
embedded in it quotes a `metrics.csv` figure that has not been checked.
*[Correction to this notice, 2026-09-24: the file carries no validation note
and quotes no `metrics.csv` figure. It was written by
`export_au_grid_netcdf.py`, not by the harness exporter that embeds one. Its
data stands: the corrected wind applies training-produced factors to grid
cells, and its capacity factor uses one generic curve. Its `summary` attribute
does repeat the withdrawn claim that the correction compresses South
Australia's cycle toward observation. The file is left as written; the script
no longer writes that claim.]*

**A transfer re-measured.** No transfer run behind this document was
retained. Both transfers were re-run from the scorecard rows' training runs
(`refresh_2026-09-20`: DK k100 season, trained 2015-19; AU-NEM k45 season,
trained 2020-22), from a clean tree on the licensed library as input root,
before the fix (`b6bdfe4`) and after it (`56b293e`). These are not the runs
this document reports, so they do not reproduce its figures: the AU-NEM test
fleet here is 77 farms with an uncorrected MBE of +0.009, not −0.024.

DK → AU, test year 2023, 77 farms, 920 farm-months:

| Variant | RMSE before | RMSE after | MAE before | MAE after | MBE before | MBE after | r before | r after |
|---|---|---|---|---|---|---|---|---|
| uncorrected | 0.1153 | 0.1153 | 0.0943 | 0.0943 | +0.0089 | +0.0089 | 0.463 | 0.463 |
| transfer from DK k100 season | 0.1174 | **0.1128** | 0.0971 | **0.0900** | −0.0526 | **−0.0469** | 0.448 | **0.438** |

AU → DK, test year 2020, 5,410 turbines, 64,090 turbine-months: its
`metrics.csv` is byte-identical before and after the fix (uncorrected RMSE
0.1482, transferred 0.1606). That is the control: its target fleet is sorted,
so the fix must not move it.

Data: `output/c1_turbine_order_2026-09-24/transfer/` (`before/` and `after/`).

Closing document for the Australia/NEM validation. The per-result documents
behind it are `method-harness-regression.md` (does the refactored pipeline still
reproduce known-good results?) and `region-au-nem.md` (the seasonal
finding and its absolute-skill cost).

## The central result: correction value tracks the STRUCTURE of the bias

The branch's finding, assembled from both hemispheres:

| | Denmark | Australia/NEM |
|---|---|---|
| ERA5 level bias (MBE) | **+0.121** (level-dominated) | −0.024 (near-unbiased) |
| ERA5 shape error | small | **over-amplified SA cycle** |
| Affine correction, absolute RMSE | 0.160 → 0.099 (**−38%**) | 0.104 → 0.120 (+16%) |
| Affine correction, cycle tracking | improves | 0.072 → 0.065 (**−11%**, the pre-specified gate) |

*[Withdrawn 2026-09-24: the Australia/NEM column's two corrected rows, and the
Australia half of the paragraph below; the Denmark column stands. See the
notice of that date.]*

The affine-in-wind correction earns its keep **where the reanalysis bias is
level-dominated** (DK: it wins on every metric) and **compresses the
seasonal shape where the bias is shape-dominated** (AU: the pre-specified
cycle gate passes on both curve libraries). It cannot improve absolute
skill where there is no level bias to remove, and its level machinery then
adds farm-level noise (AU: +16% RMSE, reported in full in
`region-au-nem.md`). This is one result seen from two sides, not a success
and a failure: the method improves the component of the bias that exists.
For anyone applying correction to a new region, the actionable version is:
**diagnose the bias structure first** (level vs shape decomposition of
uncorrected-vs-observed; cheap, needs only monthly aggregates), and expect
level-dominated regions to gain broadly, shape-dominated regions to gain on
seasonal profiles only. One line of future work follows and is deliberately
not pursued here: a shape-only correction variant (unit mean scalar per
cluster) is the obvious formulation for level-unbiased regions. It is named,
not proposed or evaluated, because iterating the method on this evaluation
data would be post-hoc tuning of the kind the phase boundary closed.

## Transfer: corrections are regional bias fingerprints

The same fact from the other direction. Under the pre-agreed semantics
(capacity-weighted collapse to one factor pair per season, uniform
application, season-NAME matching so AU winter factors land on the target's
winter months):

- **AU → DK is gracefully useless** (+4% RMSE over uncorrected; native DK
  correction: −38%). AU's clusters carry *opposing* corrections (scalars
  0.78–2.4) that cancel to near-identity in the collapse (0.98–1.10): a
  region without a common level bias has nothing to port.
- **DK → AU is bounded-harmful** (+32% RMSE; MBE −0.024 → −0.093). DK's
  pull-down factors (collapsed scalars 0.72–0.84) are the right medicine for
  DK's over-blown reanalysis and the wrong medicine for near-unbiased AU.
  *[Withdrawn 2026-09-24: the +32% RMSE and the corrected MBE; the collapsed
  scalars stand. A re-run from different training runs is in the notice of
  that date.]*
- Degradation is graceful in both directions (bounded, sign-consistent, no
  pathologies), a publishable negative result: correction factors encode a
  region's specific reanalysis-bias fingerprint, not portable physics.
  *[Withdrawn 2026-09-24: the DK → AU half of this statement, which rests on
  the withdrawn transfer figures; see the notice of that date.]*

## What generalises (validated)

- **The pipeline.** The regression validation: the harness reproduces the
  legacy method bit-for-bit
  (max abs diff 0.000e+00) on DK/DE/NL/FR (turbine-level, postcode-located,
  and country-level joint-offset paths) against real curves and data, with
  both methodology preconditions (PYVWF_INPUT honoured on both sides; main
  deterministic against itself) established first.
- **Region-as-config.** Australia needed one observation adapter and one
  TOML file; no rewrites. Explicit season-month lists killed the
  NH-hardcoding hazard by construction (four hardcoded sites found and
  closed; the mirrored-hemisphere test proves month-matched application is
  worse than no correction).
- **The ingest pattern.** Market-time archives straddling UTC months →
  reduce-to-partials-then-finalise, with commissioning and
  registered-capacity masks (42/3,455 farm-months) keeping CF denominators
  clean. Verified fast path == slow path frame-identically.
- **The evidence discipline.** Pre-specified gates; must-distinguish tests
  (a check that cannot fail is not a check; the planned 0-360 longitude
  wrap test turned out to be provably vacuous for Australia and was
  replaced by a latitude-flip equivalence test); conservative headline
  numbers with
  robustness analyses alongside (all-farms −10.9% as the claim, far-north
  exclusion −18.3% as support). *[Withdrawn 2026-09-24: both figures; see the
  notice of that date.]*
- **Dual-stack robustness.** The seasonal-cycle verdict and its regional
  pattern hold on the licensed curve library and on a fully-open library with a
  different matching strategy (confounded by design; divergence was a
  hard-stop condition and did not occur). Curve choice is level-not-shape
  (normalized-cycle r ≥ 0.9989 across real curves). *[Withdrawn 2026-09-24:
  the verdict and its regional pattern, which rest on corrected figures; the
  uncorrected curve comparison stands. See the notice of that date.]*

## What does not generalise

- **Correction parameters** across regions (the transfer result above).
- **Cluster counts and granularity**: k-means with a geographic outlier
  distorts *neighbouring* clusters, not just its own (the n=1 far-north
  cluster degraded NSW's clustering; its exclusion improved a region it was
  never in). Cluster configuration is per-region tuning, not a constant.
- **The uniform-curve shortcut**: rejected for AU after the field-coverage
  check; the synthetic placeholder curves specifically distort seasonal
  amplitude (+0.15–0.46) and are unusable for seasonal claims.
- **"Corrected = improved"**: unqualified, that claim is false in AU on
  absolute skill. Language must track the diagnosed structure. *[Withdrawn
  2026-09-24: the AU evidence for this; see the notice of that date.]*

## What a genuinely global method needs

1. **A bias-structure diagnosis step** ahead of correction choice: level vs
   shape decomposition per region, from monthly aggregates.
2. **Curtailment-aware observations** in high-penetration markets. SA (the
   region carrying the AU finding) is the NEM's most curtailed, curtailment
   is seasonal, and resource bias vs curtailment-driven seasonality cannot
   be separated without semi-dispatch data. The tracking claim survives;
   attribution does not. *[Withdrawn 2026-09-24: the tracking claim, by
   `region-au-nem.md`'s notice of that date; the attribution limit stands.]*
3. **Hub-height and turbine-model metadata** as first-class inputs: AU ran
   on a uniform 100 m default and a three-tier curve compilation (44.1% of
   capacity OEM-confirmed); Europe's per-turbine registries are the
   exception globally, not the rule.
4. **Registered-capacity histories** everywhere (the AU DUDETAIL mask
   pattern), or ramping fleets inject spurious sub-annual signal.
5. **Redistributable curve and reference data** for reproducibility: the
   open-library run shows the full stack can be open without changing the
   verdict. *[Withdrawn 2026-09-24: the verdict this compares; see the notice
   of that date.]*

## Open questions

**Answered here.** Where the affine correction holds (level-dominated
regions) and where it stops (shape-dominated ones), which is the central
result above. Seasonal factors beat fixed ones on cycle tracking in
Australia and roughly tie in Denmark. *[Withdrawn 2026-09-24: the Australia
comparison and the answered status of the central result; see the notice of
that date.]* The hemisphere and season-definition
question is settled by explicit month lists. The ERA5 longitude and
latitude-ordering hazards are closed with tests. The NEMWeb access path is
resolved.

**Closed by decision, not by evidence.** Non-linear correction variants
were ruled out of this work to avoid tuning the method on the same data
used to evaluate it; the shape-only variant named in the central result
stays one line of future work. External validation anchors were deferred:
Renewables.ninja for Australia would be circular, and OpenNEM is the
preferred anchor if one is ever run. Yawong is excluded and stated as a
limitation.

**Still open, with what would unblock each.**

- **Country-level joint-offset identifiability.** Needs a synthetic
  ground-truth experiment. Untouched here, because the regression
  validation established only that the refactor reproduces the old
  behaviour, not that the behaviour is well-posed.
- **Whether seasonal factors help directionally**, not just on cycle
  tracking.
- **Systematic pooling curves**: how skill varies as observations are
  pooled across sites. The Denmark-versus-Germany postcode comparison is
  the cheap first probe, since the two regions differ mainly in how
  precisely turbines are located.
- **A formal synthetic-versus-real curve attribution.** The dual-library
  run covers what the demonstration needs; the formal two-way comparison
  was not run.
- **Hub-height and vintage covariates.** Blocked on height data. Australia
  runs on an all-default 100 m height, so it cannot be tested there.
- **Separating curtailment from resource bias in South Australia.** Needs
  semi-dispatch-cap data.

## What this work produced

The regression validation (bit-for-bit on four regions), a synthetic
Southern-Hemisphere ground-truth test through the full pipeline, the
data-integrity preconditions, the dual-library seasonal validation with its
absolute-skill cost reported in full, a validation notebook that runs on the
open stack, a gridded corrected-wind and capacity-factor NetCDF export, and
this synthesis with the transfer table. Real curves and raw market and
reanalysis data never entered committed state or CI.
