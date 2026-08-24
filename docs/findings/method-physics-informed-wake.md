# E9: a physically-shaped wake term. Right diagnosis, wrong remedy.

Registered in `method-physics-informed-prespecification.md` addendum 8 before implementing, with
three predictions. Two passed, the decisive one failed, and the term is **not
adopted**. What it establishes is worth more than a small accuracy gain would
have been.

## What was built

The residual anatomy found the model over-predicting by **+0.054 capacity
factor** in the densest bin of local capacity density (above 0.75 MW/km2)
against roughly +0.004 in the middle bins. The efficiency term's dependence on
density was too weak at the top end. The fix adds a shape, not flexibility:

    eta = eta_base(offshore, hub height, 50 km density) * 1 / (1 + c * D)

`D` is local capacity density in MW/km2 within 10 km; `c >= 0` is one global
coefficient, the only free number the term adds. The 10 km density is withheld
from `eta_base` so the learned and physical routes cannot offset.

Hyperbolic rather than the `exp(-cD)` first sketched: observed densities reach
9 MW/km2, where an exponential steep enough to matter at the median (0.14)
collapses the efficiency entirely. Deep-array losses saturate towards an
asymptote set by the momentum the boundary layer can supply; they do not decay
to nothing.

## The three predictions

**P1 -- the dense-bin residual falls: CONFIRMED, strongly.**

| capacity density (MW/km2) | mean residual, no wake | with wake |
|---|---|---|
| 0.00-0.045 | +0.0059 | +0.0222 |
| 0.045-0.080 | +0.0048 | +0.0200 |
| 0.080-0.138 | +0.0003 | +0.0143 |
| 0.138-0.244 | +0.0036 | +0.0178 |
| 0.244-0.747 | −0.0019 | +0.0058 |
| **0.747-9.04** | **+0.0542** | **+0.0275** |
| **span across bins** | **0.0561** | **0.0217** |

Span relative to the residual's own spread falls from 0.726 to 0.283, a 61%
cut. The diagnosis was right: the dense-fleet over-prediction IS a density
effect, and this shape removes most of it.

**P2 -- in-region RMSE unchanged within 0.001: CONFIRMED, barely.** Pooled
0.07795 to 0.07886, +0.00091. It passes only because the tolerance allowed
"unchanged"; the true direction is slightly worse. Only the UK improves
(−0.0040); DE +0.0078, US +0.0053, BR +0.0048, DK +0.0038.

**P3 -- zero-shot transfer improves in at least 3 of 5: FAILED, 0 of 5.**

| holdout | best config (D) | D + wake | change |
|---|---|---|---|
| BR | 0.1030 | 0.1073 | +0.0043 |
| DE | 0.0678 | 0.0820 | **+0.0142** |
| DK | 0.0951 | 0.1105 | **+0.0154** |
| UK | 0.1363 | 0.1381 | +0.0018 |
| US | 0.1001 | 0.1006 | +0.0005 |

Mean skill **+0.338 to +0.235**. Not a marginal loss: the term costs roughly
what nine-region training and MLP heads together had won.

## Two explanations proposed and rejected

**"The efficiency head is undoing it."** The first run left the 10 km density in
`eta_base`, contrary to what addendum 8 specified, so the model had a learned
route and a physical route to the same effect. That was a genuine deviation and
was fixed. It changed almost nothing: span 0.0219 to 0.0217, pooled RMSE 0.07870
to 0.07886. **The gap was real and was not the explanation.**

**"Capacity density is a geolocation artefact."** German rows are postcode
centroids and British rows are farm points, so their densities might be inflated
relative to fleets with true coordinates. Measured across nine regions, the rank
correlation between geolocation resolution and median density is **+0.067** --
essentially none. Germany has the second-worst resolution and among the LOWEST
densities. **Rejected.**

The fitted coefficient is not unstable either: `c` = 0.165-0.174 km2/MW across
all five holdouts, seed spread ~0.001. The term is estimated consistently and
transfers badly anyway.

## What it does establish

The best remaining explanation is one this workstream had already written down
as a limitation, and this is the first hard evidence for it: **the observation
unit differs by region, and so does how much wake loss the target already
contains.** A plant-level or complex-level capacity factor is an array average
with internal wakes already inside it; a single-turbine capacity factor is not.
A single global coefficient cannot be correct for a target that in Denmark and
Germany excludes internal wakes and in the United States and Brazil includes
them. The physics is right; the target is not consistent enough across regions
to carry a global physical coefficient for it.

That is a data statement, not a modelling one, and it is falsifiable: fitting
`c` separately by observation unit, or harmonising the targets to a common
aggregation level, should recover the transfer. Neither is done here.

## Disposition

- The term is **implemented, tested (7 tests) and off by default.** It is not
  in any carried-forward configuration.
- `method-physics-informed-evaluation.md` ranked it the top improvement lead. That ranking was
  based on in-region residual structure, and the lesson is that **residual
  structure ranks candidates by size, not by transferability**. The next two
  leads -- profile curvature at the hub-height extremes, and directional terrain
  exposure -- should be judged against transfer from the start, not against the
  in-region residual that suggested them.
