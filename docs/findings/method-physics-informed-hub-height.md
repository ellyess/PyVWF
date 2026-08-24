# D8: what Brazil's uniform 100 m hub height costs, and whether to go and get the real one

Prompted by a concrete failure: the first smoke test of the profile-curvature
change (addendum 9) returned **bit-identical** results for two different profile
forms. Every Brazilian complex sits at exactly 100 m, which is the reference
height of the ERA5 wind field, and at that height every profile returns a ratio
of exactly 1. Brazil is invisible to the entire hub-height question.

## The data is not in this repository

| source | present | carries hub height |
|---|---|---|
| ONS raw (`FATOR_CAPACIDADE-2_*.csv`) | yes | **no** (subsystem, state, connection point, coordinates, generation, capacity) |
| `br_md.csv` | yes | **no** (193 complexes at 100.0, `height_source = "default-uniform"`, `commissioning_date` all NaN) |
| Global Wind Power Tracker | yes | **no** (capacity, coordinates, start year, operator, owner) |
| ANEEL SIGA | **absent** | commissioning, not hub height |
| `configs/curation/*_turbine_specs.csv` | CL (60 rows), AR (65) | **no BR table exists** |

`docs/runbooks/BR.md` already states this and lists vintage-aware per-complex
assignment as a named follow-up, not done.

The machinery to consume such data exists: `apply_turbine_specs.py` joins a
per-plant table (`ID, turbine_count, rotor_diameter_m, hub_height_m`), assigns
real curves by scale and specific power, and records which plants are real and
which remain proxies. Only the table is missing, and building it means the kind
of per-farm research behind the Chilean and Argentine tables, each row carrying
a cited source, for **193 complexes**.

## What it would buy: essentially nothing

Refitting Brazil (MLP heads, 60 epochs) while varying the height assumption.

**Level.** A uniform height swept across the plausible range:

| assumed height | power | shear-log |
|---|---|---|
| 80 m | 0.0912 | 0.0912 |
| 90 m | 0.0918 | 0.0914 |
| 100 m (the default) | 0.0918 | 0.0918 |
| 110 m | 0.0916 | 0.0919 |
| 120 m | 0.0917 | 0.0916 |

Total spread across a 40 m range: **0.0006 RMSE**.

**Heterogeneity**, heights drawn around 100 m with a modern fleet's spread
(sd ~14 m), three draws: RMSE 0.0938, 0.0885, 0.0929, mean **0.0917** against
0.0918 for the uniform default. Cost of assuming uniformity: **−0.0001**, with a
seed-to-seed spread of 0.0028 that is **28 times larger than the effect**.

**Could Brazil then discriminate profile forms?** Mean |power − shear-log|
rises from 0.00019 to 0.00036. Still an order of magnitude below the seed noise.

## Why, and what the test does and does not show

The level arm was close to uninformative **by construction**, and that is worth
stating rather than presenting as a finding. Multiplying every hub height by a
constant multiplies the profile ratio by very nearly a constant, and the
per-site speed-up term `exp(gamma)` multiplies the wind too. The two are
near-degenerate, so the model absorbs a uniform height change almost exactly.
Any level sweep had to come out flat.

The heterogeneity arm is the informative one, and it carries its own limit: it
changes the height INPUT while the observations still reflect the true, unknown
heights. So it measures how much the model's answer depends on the height
assumption, not how much accuracy is lost by getting heights wrong. For the
decision at hand that is the right quantity: if the output barely moves when the
input changes, better input will not move the output much either.

Brazil is also an unusually insensitive case on physical grounds. Its Nordeste
trade-wind regime keeps turbines near or above rated power much of the time,
where the power curve is flat and a few percent of wind speed costs little.

## Recommendation

**Do not build the Brazilian turbine-spec table for this purpose.** 193
complexes of manual research would move the result by less than a thousandth of
an RMSE point, against seed noise nearly thirty times larger, and would not make
Brazil useful for profile physics either.

Two caveats on the scope of that recommendation. It is about **hub height**: the
same table would also supply real turbine MODELS, and `method-generalisation.md`
records curve identity as a separate and larger concern, untested here. And it
is about **Brazil**: a fleet with genuinely varied heights and a wind regime
that spends more time on the steep part of the curve, such as Germany's
40 to 149 m range, is where hub-height quality earns its keep.
