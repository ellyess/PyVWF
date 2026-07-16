# D2 synthesis — what generalises, what doesn't, and what a global method needs

Closing document for the Australia/NEM validation (D1–D5 on the
`multi-region-validation` branch). Full decision history: the validation
log (`validation_progress.md`, checkpoints 1–23); per-result documents:
`d1_regression.md`, `pillar_a_au.md`.

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
cycle gate passes on both curve libraries) — but it cannot improve absolute
skill where there is no level bias to remove, and its level machinery then
adds farm-level noise (AU: +16% RMSE, reported in full in
`pillar_a_au.md`). This is one result seen from two sides, not a success
and a failure: the method improves the component of the bias that exists.
For anyone applying correction to a new region, the actionable version is:
**diagnose the bias structure first** (level vs shape decomposition of
uncorrected-vs-observed — cheap, needs only monthly aggregates), and expect
level-dominated regions to gain broadly, shape-dominated regions to gain on
seasonal profiles only. One line of future work follows and is deliberately
not pursued here: a shape-only correction variant (unit mean scalar per
cluster) is the obvious formulation for level-unbiased regions — named,
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
- Degradation is graceful in both directions — bounded, sign-consistent, no
  pathologies — a publishable negative result: correction factors encode a
  region's specific reanalysis-bias fingerprint, not portable physics.

## What generalises (validated on this branch)

- **The pipeline.** D1: the harness reproduces the legacy method bit-for-bit
  (max abs diff 0.000e+00) on DK/DE/NL/FR — turbine-level, postcode-located,
  and country-level joint-offset paths — against real curves and data, with
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
  (a check that cannot fail is not a check — see the pillar C re-scope,
  where the planned 0–360 wrap test was provably vacuous for AU and the
  lat-flip equivalence replaced it); conservative headline numbers with
  robustness analyses alongside (all-farms −10.9% as the claim, far-north
  exclusion −18.3% as support).
- **Dual-stack robustness.** The pillar A verdict and its regional pattern
  hold on the licensed curve library and on a fully-open library with a
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

1. **A bias-structure diagnosis step** ahead of correction choice — level vs
   shape decomposition per region, from monthly aggregates.
2. **Curtailment-aware observations** in high-penetration markets: SA — the
   region carrying the AU finding — is the NEM's most curtailed, curtailment
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

## Open-questions triage (the once-only triage mandated at checkpoint 8)

**Answered by this branch:** RQ1 in its practical form (where affine holds:
level-dominated regions; its limit: shape-dominated ones — the central
result); RQ2's seasonal half (seasonal factors beat fixed on cycle tracking
in AU; roughly tie in DK); the hemisphere/season design question (explicit
month lists, adopted); the ERA5 longitude/lat-ordering hazards (closed with
tests); the NEMWeb access-path question (resolved, HTTPS live).

**Closed by ruling:** RQ5 non-linear variants (gated at the phase boundary;
the shape-only variant stays one named line of future work); pillar D
external anchors (deferred; RN-AU noted as circular, OpenNEM preferred if
ever run); Yawong (excluded, stated limitation); US/Brazil regions
(deferred off-branch).

**Open, with what unblocks each:** RQ4 country-level joint-offset
identifiability (needs the synthetic-ground-truth experiment; untouched —
D1 validated only reproduction); RQ2's directional half; RQ3 systematic
pooling curves (the DK-vs-DE postcode diagnostic from checkpoint 3 remains
the cheap first probe); RQ6 formal synthetic-vs-real attribution (the
dual-library run covers the demo-relevant content; the formal two-way
comparison was not run); RQ7 hub-height/vintage covariates (blocked on
height data; AU all-default heights make it impossible there today);
curtailment separation in SA (needs semi-dispatch-cap data); the JOSS
synthetic→open curve replacement (parked post-AU, checkpoint 19 — the
strongest candidate for immediate follow-up).

## Deliverables shipped

D1 regression validation (PASS, bit-for-bit, 4 regions) · pillar B
synthetic SH ground truth (3 pins) · pillar C data-integrity preconditions
(re-scoped honestly, PASS) · pillar A dual-library seasonal validation
(gate PASS ×2, absolute skill reported in full) · D3 notebook (open-stack,
headlessly verified, licence-diligenced data bundle) · D4 gridded NetCDF
(exclusion-run factors, loud provenance) · D5 this synthesis + the transfer
table. All work on `multi-region-validation`; real curves and raw
market/reanalysis data never entered committed state or CI.
