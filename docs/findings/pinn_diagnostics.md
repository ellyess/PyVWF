# Diagnostics D0-D4: why the correction does not generalise

Measurements on artefacts already on disk (the canonical post-fix train runs at
commit `8a032d6`, k=100 for DK/DE/UK/US and k=60 for BR). No model is tuned
here and no gate is claimed; gates for the modelling work are fixed in
`pinn_prespecification.md`. Scripts: `scripts/pinn/d[0-4]_*.py`.

Two of the five diagnostics falsified the hypothesis they were written to test.
Both are reported.

---

## D0. The published transfer metric answers a question nobody can act on

`ml_transfer_retest.md` scores transfer as sklearn `r2_score` on the per-cluster
scalar. That denominator is the **holdout region's own mean** -- a number you do
not have when you arrive in an unseen region. The choice a practitioner faces is
"apply a transferred correction, or apply none", so the baseline that decides it
is the identity correction (scalar 1, offset 0).

Same targets, same RandomForest(100/10/10), same seeds, both denominators:

| Holdout | R² vs holdout mean (published) | skill vs no correction | RMSE identity | RMSE ML |
|---|---|---|---|---|
| BR | +0.187 | **+0.325** | 0.892 | 0.733 |
| DE | −0.016 | **+0.129** | 0.264 | 0.247 |
| DK | −0.226 | **+0.759** | 0.241 | 0.118 |
| UK | −0.197 | −0.120 | 0.293 | 0.310 |
| US | −1.049 | −0.774 | 0.623 | 0.830 |
| | **1/5 positive** | **3/5 positive** | | |

The pooled training-region mean -- transfer with no model at all -- scores
**0/5**, so the model is doing something real: it predicts region-specific
levels, not a constant.

DK is the clearest case. Its scalars sit at 0.784 ± 0.107, so the holdout mean
is an extremely strong predictor (hence R² = −0.226) while the identity is far
away (0.784 vs 1.0). The model gets DK's **level** roughly right and its
**within-region spread** wrong. R² against the holdout mean scores only the part
it fails at.

**This does not overturn the published finding**, which is correct on its own
terms: within-region structure does not transfer. It does mean the headline
"correction factors are not predictable in unseen regions" is metric-dependent,
and that both numbers belong in any restatement of it. Neither is decisive:
both are proxies in scalar space, and the deciding test is capacity-factor
error end to end (gate P1).

---

## D1. Sub-grid terrain: hypothesis falsified as posed, replaced by a better variable

Predicted, in advance: the scalar should rise with **elevation excess above the
ERA5 cell mean** (linear speed-up theory), consistently across regions.

Correlation of scalar with elevation excess `h_excess`:

| | BR | DE | DK | UK | US |
|---|---|---|---|---|---|
| pearson | +0.66 | **−0.29** | +0.13 | +0.01 | −0.06 |
| p | 1e-8 | 0.004 | 0.19 | 0.89 | 0.57 |

Region-inconsistent, and the wrong sign in DE. **The prediction fails.**

The same run measured `cell_relief`, the elevation range within the 0.25° cell:

| | BR | DE | DK | UK | US | pooled |
|---|---|---|---|---|---|---|
| pearson | +0.76 | +0.70 | −0.08 | +0.51 | +0.44 | **+0.55** |
| p | 1e-12 | 1e-15 | 0.45 | 1e-7 | 4e-6 | 9e-38 |

Positive, significant and same-signed in 4 of 5 regions; DK is null and DK is
flat (cell relief 78 ± 29 m against 320-362 m elsewhere), so it has almost no
relief variance to correlate with.

The mechanism is not "the turbine is high". It is "the cell contains terrain the
reanalysis cannot see". The largest US scalar (4.25, central Washington) sits
**142 m below** its cell mean while its cell spans 1083 m of relief. Developers
site turbines on the windy features inside a cell; relief measures how much
within-cell variation exists for siting to exploit, and elevation excess does
not. This was **not** the hypothesis under test, so it is a lead, not a result.

---

## D2. The affine correction is rank-1, but that is not what breaks transfer

`pearson(scalar, offset)` = **−0.999** (DE), −0.996 (UK, US), −0.992 (DK),
−0.836 (BR). Regressing offset on scalar recovers a pivot speed:

| Region | pivot w_p (m/s) | r² of the ridge | level = a·w_p + b |
|---|---|---|---|
| BR | 2.62 | 0.698 | 4.24 ± 1.78 |
| DE | 4.10 | 0.997 | 3.97 ± **0.065** |
| DK | 4.22 | 0.984 | 3.99 ± **0.065** |
| UK | 3.95 | 0.991 | 3.93 ± 0.106 |
| US | 3.81 | 0.991 | 4.02 ± 0.225 |

Every cluster's correction passes through essentially the same point, near
`w' = w ≈ 4 m/s`. The fitted family is a **rotation about the turbine cut-in
speed**: below cut-in no power is made, so the objective cannot constrain the
correction there and the fit pivots about it. There is one real parameter, the
gain, and the offset is read off the ridge.

Predicting two collinear targets independently looked like a defect worth
fixing. Constraining the offset to the ridge (`b = c − w_p·a`, ridge constants
fitted on training regions only) scored, in wind-speed space over 4-20 m/s:

| Holdout | skill, offset free | skill, offset on ridge |
|---|---|---|
| BR | +0.288 | +0.287 |
| DE | +0.129 | +0.100 |
| DK | +0.762 | +0.760 |
| UK | −0.166 | −0.137 |
| US | −0.885 | −0.898 |

**No gain.** The RF's two predictions are already correlated, because they are
fitted to correlated targets from the same features, so the errors already
cancel. Hypothesis tested and closed.

---

## D3. Feature scale was wrong; fixing it does not fix transfer

The published features are terrain derivatives on a 3-pixel window of a 30
arc-second grid: about **90 m** around the centroid. The bias arises at the
scale of the ERA5 cell, about **28 km**.

Pre-specified prediction: a pure-physiography multi-scale set (position, spread
and relief at 1/5/28/84 km, no longitude or latitude, so it cannot encode region
identity) beats SET_A in ≥3/5 regions **and** improves both failing regions, US
and UK.

Wind-space LORO transfer skill (>0 beats leaving ERA5 alone):

| Holdout | SET_A (published) | MULTI | SET_A+MULTI | MULTI+hub height |
|---|---|---|---|---|
| BR | +0.288 | +0.121 | +0.208 | +0.124 |
| DE | +0.129 | **+0.574** | +0.493 | +0.567 |
| DK | +0.762 | **+0.786** | +0.780 | +0.797 |
| UK | −0.166 | **−1.539** | −1.518 | −1.895 |
| US | −0.885 | **−0.390** | −0.346 | −0.416 |
| positive | 3/5 | 3/5 | 3/5 | 3/5 |
| mean | +0.025 | −0.090 | −0.077 | −0.165 |

MULTI beats SET_A in 3/5 (DE, DK, US) and roughly halves the US failure, but
**UK collapses from −0.17 to −1.54**. The prediction was a conjunction and it
**fails**. Mean skill falls.

Pooled feature importance confirms the scale diagnosis even so:

| feature | importance |
|---|---|
| relief_28km | 0.193 |
| std_84km | 0.173 |
| z_site | 0.131 |
| std_28km | 0.126 |
| relief_1km | 0.022 |
| std_1km | 0.022 |

The kilometre-scale features the published experiment relied on are worth ~2%
each; the ERA5-cell and orographic scales carry the signal. **Scale was
mis-specified, and correcting it is not sufficient.**

---

## D4. Part of what is being predicted is estimation noise

Empirical variogram of the per-cluster scalar. Two clusters close enough to
share an ERA5 cell should have nearly the same true scalar, so
`gamma(d) = ½·E[(a_i − a_j)²]` tends as `d → 0` to the variance of the
estimation noise -- the nugget.

| Region | n | scalar var | shortest bin | γ/var in that bin | fitted range (km) |
|---|---|---|---|---|---|
| BR | 60 | 0.672 | 20-30 km | **0.062** | 265 |
| DE | 100 | 0.061 | 20-30 km | **0.145** | 774 |
| DK | 100 | 0.012 | 10-20 km | **0.279** | 224 |
| UK | 100 | 0.081 | 20-30 km | **0.587** | 77 |
| US | 100 | 0.339 | **100-150 km** | 0.292 | 186 |

Read the directly measured `γ/var` in the shortest bin, not the fitted nugget:
with no pairs closer than 20 km, the fitted nugget is an extrapolation.

- **BR and DE have strongly spatially structured scalars** (6% and 15% of
  variance already present at 25 km). Their targets are largely learnable.
- **UK does not.** Clusters 20-30 km apart differ by **59% of the region's total
  scalar variance**. At k=100 over 348 turbines, UK clusters hold ~3.5 turbines
  each, and the fitted factors are correspondingly noisy. UK is also one of the
  two transfer failures, and D3 made it worse -- consistent with a model chasing
  noise it cannot see is noise.
- **US cannot be assessed**: its clusters have no pairs closer than 100 km, so
  the nugget is unconstrained and its apparent ceiling of 1.00 is an artefact.
  US's failure is **not** explained by this test.

---

## What the five diagnostics jointly say

1. The transfer failure is **not** a metric artefact alone (D0), **not**
   collinearity (D2), and **not only** feature scale (D3).
2. What transfers is the regional **level**; what does not is **within-region
   structure** (D0), and in at least one region (UK) a large share of that
   structure is estimation noise rather than signal (D4).
3. The one physical variable that tracks the scalar consistently across regions
   is **terrain the reanalysis cannot resolve** (D1, D3), described at the ERA5
   cell scale and above.

Together these indict the **two-stage design** rather than the choice of
regressor. The current method fits free affine factors per cluster, then a
second model regresses those factors on features. Stage one injects estimation
noise (D4), depends on an arbitrary k-means partition, produces a rank-1
parameter pair (D2) whose second parameter is unidentified below cut-in, and
discards every turbine-month into ~100 cluster summaries. Stage two then has to
learn from those summaries.

The alternative that follows is to remove stage one: make the correction a
smooth function of physiography, and fit it **directly to observed generation**
through the physical forward operator, so the supervision is the observation
rather than a noisy intermediate estimate. That is the model specified in
`pinn_prespecification.md` and built next.
