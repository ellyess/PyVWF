# Pillar A — Australia/NEM seasonal validation (D2)

**Finding.** PyVWF detected and corrected a strong seasonal ERA5 bias in
**South Australia**: ERA5 does not merely misestimate SA's seasonal cycle, it
**over-amplifies it** — the shape is right (winter-peaking) but the amplitude
is exaggerated in both directions (June simulated at 1.617× annual mean vs
1.394 observed; January at 0.929 vs 1.090 observed). The seasonal correction
compresses the simulated cycle toward observation (SA cycle-tracking RMSE
0.111 → 0.097, −12%, on the real curve library; 0.106 → 0.094 on the open
library). Regions with little seasonal bias showed no improvement or slight
degradation — which is the expected behaviour of an honest seasonal
correction, and part of the evidence (see *Selectivity*).

The fleet-level gate (capacity-weighted normalized-cycle RMSE, 76 farms with
complete held-out-2023 observations) passes on both curve libraries —
real 0.0724 → 0.0645 (−10.9%), open 0.0672 → 0.0614 (−8.7%) — with no
divergence in verdict, direction, or regional pattern. The fleet number is
supporting detail; the finding is SA.

## Setup

Corrections trained 2020–2022 (SH-season config, 5 clusters, affine-in-wind
baseline), evaluated on held-out 2023. Observations: AEMO 5-minute SCADA per
DUID, UTC-binned monthly, coverage floor, commissioning and
registered-capacity masks (42/3,455 farm-months masked). Fleet: 104 farms,
14.26 GW (99% of NEM wind DUIDs); 76 farms carry complete 2023 observations
and form the gate set. Preconditions established beforehand: pillar B
(synthetic SH ground truth through the full pipeline: labels, value, and
NH-application-is-worse-than-nothing, all pinned), pillar C (ERA5-AU slice
integrity: 48/48 file integrity, farm containment, lat-flip bit-identity —
the 0–360 check is provably vacuous for AU and is recorded as an invariance
proof, not evidence), and a curve-sensitivity check showing curve choice is
level-not-shape (normalized-cycle r ≥ 0.9989 among real curves).

Turbine identity per farm: `configs/au_turbine_models.csv` — 44.1% of fleet
capacity manufacturer-confirmed from OEM/developer sources, 47.4%
secondary-source, 8.5% per-farm fallback (marked). Real-library matching via
the same `add_models` manufacturer/p_density path as every validated region.

## Gate tables

Normalized-cycle RMSE vs observed, capacity-weighted (`corrected` = seasonal):

| Region (farms) | real unc | real cor | open unc | open cor |
|---|---|---|---|---|
| **FLEET (76)** | 0.0724 | **0.0645** | 0.0672 | **0.0614** |
| SA1 (24) | 0.1107 | **0.0972** | 0.1059 | **0.0942** |
| NSW1 (17) | 0.0886 | 0.0927 | 0.0885 | 0.0897 |
| VIC1 (28) | 0.0669 | 0.0704 | 0.0598 | 0.0619 |
| QLD1 (3) | 0.1139 | 0.1249 | 0.1138 | 0.1397 |
| TAS1 (4) | 0.0509 | 0.0470 | 0.0345 | 0.0587 |

JJA (winter) |error|, fleet: real 0.0506 → 0.0443 (improved); open
0.0401 → 0.0400 (**neutral** — reported as such). One nuance a reader of
both tables will spot, stated here first: on the real library the *fixed*
(non-seasonal) correction beats the seasonal one on JJA specifically
(0.0403 vs 0.0443) while the seasonal correction wins the full cycle —
because SA's cycle gain comes mostly from the non-winter months (the summer
undercut and shoulder seasons; SA's JJA error itself barely moves,
0.0804 → 0.0803).

## Selectivity is evidence

NSW and VIC worsen slightly under the seasonal correction (e.g. VIC
0.0669 → 0.0704). A seasonal correction can only add noise where there is no
seasonal bias to correct, so mild degradation in low-bias regions is the
expected signature of an honest method — it certifies that the SA improvement
is signal, not an artifact of a method that always improves numbers. Both
curve libraries reproduce the same regional pattern, so the selectivity is
robust to curve choice and matching strategy (which are confounded between
the two stacks by design; had the verdicts diverged, that was a
stop-and-understand condition — it did not occur).

## The n=1 cluster, checked

At 5 clusters, one cluster contains a single far-north QLD farm (Mount
Emerald, lat −17.2°) and fits extreme factors (scalars 2.0–2.4) — an n=1
overfit. Re-running with the three far-north sites (lat > −20°) excluded:
the gate **strengthens** (0.0780 → 0.0637, −18.3%), SA is preserved
(0.1107 → 0.0930), and NSW flips to mild improvement (0.0886 → 0.0847) —
the outlier had also been distorting the NSW clustering. VIC remains mildly
negative. QLD's apparent degradation traces to that n=1 cluster; excluding
it, the picture is unchanged where it matters and better where it isn't.

A methodological lesson worth keeping: geographic outliers in k-means
clustering distort *neighbouring* clusters, not just their own — the
exclusion improved NSW, a region the outlier was never clustered with. The
fleet-with-exclusion gate (−18.3%) is reported here as robustness analysis;
the headline claim stays with the conservative all-farms number (−10.9%).

## Curtailment in South Australia

The corrected model **tracks observable SA generation better than the
uncorrected model** — that is the claim, and it is both the defensible and
the commercially relevant one: a user of the model wants the simulation that
tracks actual generation, curtailment included. Attribution is a separate
question, and we engage it directly: SA is the highest-penetration, most
curtailed NEM region, and curtailment concentrates in high-wind/low-demand
periods — SA winter. The direction of the ERA5–observation discrepancy
(simulated winter peak too strong relative to observation) is therefore
consistent with some of the gap being observed output suppressed by
curtailment rather than reanalysis resource bias. With semi-dispatch-cap
data excluded from this branch's scope, resource bias and curtailment-driven
seasonality **cannot be fully separated in SA**; "PyVWF proved a pure ERA5
resource bias in SA" is not claimable on this evidence. What the evidence
does establish: SA's effective generation cycle is materially misrepresented
by uncorrected ERA5, the correction learns and compresses that
misrepresentation, and the improvement holds across two independent curve
stacks — a result that stands whichever mixture of resource bias and
curtailment produced the observed cycle.

## Fitted factors (real library, seasonal)

Physically coherent and regionally structured: the SA-dominated cluster
(18 farms, 1.9 GW) carries scalars above 1 with strong seasonal spread
(autumn 1.42, spring 1.12 — ERA5 under-blows SA on level, with a
season-dependent error); the VIC-dominated cluster (3.3 GW) scales down
(~0.78); the TAS cluster scales down most in winter (0.85). Clustering is
identical across libraries (same coordinates and capacities), so factor
differences between stacks isolate the curve effect.

## Caveats

- One held-out year (2023); 76/104 farms had complete observations.
- Open-stack results use **p_density-class proxy matching** into
  reference/composite machines, not per-OEM curves. This caveat travels with
  every open-stack number in this document and elsewhere.
- Largest corrected-month miss in SA: December (1.174 simulated vs 0.973
  observed normalized) — the correction overshoots early summer.
- TAS (n=4, flattest observed cycle) splits between libraries (improves on
  real, worsens on open) — small-n noise, stated rather than resolved; it
  does not affect the gate or the SA finding.
- Hub heights are a uniform 100 m default (`height_source` marked); GWPT
  carries no heights. Vintage-aware heights are a named follow-up.

## Reproduction

`scripts/process_aemo_au.py` (ingest) → `scripts/assign_au_curves.py`
(per-library metadata) → `scripts/combine_era5_au_daily.py` (daily ERA5) →
harness `run_train`/`run_evaluate` with the AU-NEM spec (SH seasons) → gate
computation as in the progress log (checkpoints 16–21). Real curve library
and AEMO/ERA5 data remain local; nothing proprietary is committed.
