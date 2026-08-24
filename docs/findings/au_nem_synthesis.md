# Australia/NEM synthesis: what generalises, what doesn't, and what a global method needs

Closing document for the Australia/NEM validation. The per-result documents
behind it are `harness_regression.md` (does the refactored pipeline still
reproduce known-good results?) and `au_seasonal_cycle.md` (the seasonal
finding and its absolute-skill cost).

## The central result: correction value tracks the STRUCTURE of the bias

The branch's finding, assembled from both hemispheres:

| | Denmark | Australia/NEM |
|---|---|---|
| ERA5 level bias (MBE) | **+0.121** (level-dominated) | −0.024 (near-unbiased) |
| ERA5 shape error | small | **over-amplified SA cycle** |
| Affine correction, absolute RMSE | 0.160 → 0.099 (**−38%**) | 0.104 → 0.120 (+16%) |
| Affine correction, cycle tracking | improves | 0.072 → 0.065 (**−11%**, the pre-specified gate) |

The affine-in-wind correction earns its keep **where the reanalysis bias is
level-dominated** (DK: it wins on every metric) and **compresses the
seasonal shape where the bias is shape-dominated** (AU: the pre-specified
cycle gate passes on both curve libraries). It cannot improve absolute
skill where there is no level bias to remove, and its level machinery then
adds farm-level noise (AU: +16% RMSE, reported in full in
`au_seasonal_cycle.md`). This is one result seen from two sides, not a success
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
- Degradation is graceful in both directions (bounded, sign-consistent, no
  pathologies), a publishable negative result: correction factors encode a
  region's specific reanalysis-bias fingerprint, not portable physics.

## What generalises (validated on this branch)

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
  exclusion −18.3% as support).
- **Dual-stack robustness.** The seasonal-cycle verdict and its regional
  pattern hold on the licensed curve library and on a fully-open library with a
  different matching strategy (confounded by design; divergence was a
  hard-stop condition and did not occur). Curve choice is level-not-shape
  (normalized-cycle r ≥ 0.9989 across real curves).

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
  absolute skill. Language must track the diagnosed structure.

## What a genuinely global method needs

1. **A bias-structure diagnosis step** ahead of correction choice: level vs
   shape decomposition per region, from monthly aggregates.
2. **Curtailment-aware observations** in high-penetration markets. SA (the
   region carrying the AU finding) is the NEM's most curtailed, curtailment
   is seasonal, and resource bias vs curtailment-driven seasonality cannot
   be separated without semi-dispatch data. The tracking claim survives;
   attribution does not.
3. **Hub-height and turbine-model metadata** as first-class inputs: AU ran
   on a uniform 100 m default and a three-tier curve compilation (44.1% of
   capacity OEM-confirmed); Europe's per-turbine registries are the
   exception globally, not the rule.
4. **Registered-capacity histories** everywhere (the AU DUDETAIL mask
   pattern), or ramping fleets inject spurious sub-annual signal.
5. **Redistributable curve and reference data** for reproducibility: the
   open-library run shows the full stack can be open without changing the
   verdict.

## Open questions

**Answered here.** Where the affine correction holds (level-dominated
regions) and where it stops (shape-dominated ones), which is the central
result above. Seasonal factors beat fixed ones on cycle tracking in
Australia and roughly tie in Denmark. The hemisphere and season-definition
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
