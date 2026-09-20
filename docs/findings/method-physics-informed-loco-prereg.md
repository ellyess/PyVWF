# Leave one country out for the physics-informed model: registered design and gates

**Date:** 2026-09-16 (registered on commit, before any cache for it is built
and before any model is fitted)
**Scope:** whether the physics-informed model simulates accurate monthly
capacity factors in European countries it was never trained on, scored as
national capacity factor on each country's test year, and what part of any
transfer is tied to place rather than to global constants. Terms follow
`CONTEXT.md`. The model is `method-physics-informed.md`, reproduced at
`9faa192` (`method-physics-informed-rerun-prereg.md`).

**The thesis tested generalisation by leaving one country out, on fitted
coefficients, and never in capacity factor.** No interpolation or machine
learning result exists on those folds in capacity-factor space
(`method-loco-interpolation-prereg.md`). This study asks the question the
thesis asked, of the model that fits capacity factor directly, in the unit a
user of the simulation cares about.

**Everything below is fixed before any run exists.** A condition or a gate
added later is labelled post hoc and cannot pass.

## Folds

Twelve countries, the thesis's twelve.

| Fold | Tier | Training years | Test year | Config | Role |
|---|---|---|---|---|---|
| DK | turbine | 2015 to 2019 | 2020 | `scorecard/dk_k100.toml` | gated |
| DE | turbine | 2015 to 2018 | 2019 | `scorecard/de_k100.toml` | gated |
| UK | turbine (farms) | 2015 to 2018 | 2019 | `scorecard/uk_k50.toml` | gated |
| FR | country | 2015 to 2021 | 2023 | `scorecard/fr_country.toml` | gated |
| BE | country | 2015 to 2021 | 2023 | `scorecard/be_country.toml` | gated |
| ES | country | 2015 to 2021 | 2023 | `scorecard/es_country.toml` | gated |
| IE | country | 2017 to 2021 | 2023 | `scorecard/ie_country.toml` | gated |
| IT | country | 2015 to 2021 | 2023 | `scorecard/it_country.toml` | gated |
| NO | country | 2015 to 2021 | 2023 | `scorecard/no_country.toml` | gated |
| SE | country | 2015 to 2021 | 2023 | `scorecard/se_country.toml` | flagged |
| PT | country | 2015 to 2021 | 2023 | `scorecard/pt_country.toml` | flagged |
| NL | country | 2015 to 2021 | 2023 | `pinn_loco/nl_country.toml` | flagged |

Configs are under `configs/regions/`.

**Flagged folds are scored and never trained on, and are outside every gate
count.** Each carries a defect in its observations that would teach every other
fold something false:

- **SE:** its capacity register is derived from its own generation
  ([`STATUS.md`](https://github.com/ellyess/PyVWF/blob/3dbb83b54355290b8a25727847fcca0640909706/STATUS.md)).
- **PT:** its training register is frozen for five years.
- **NL:** its observations never exceed a capacity factor of 0.57, and it has no
  scorecard row. Its study config is `nl.toml` moved to the `EU_2026-09`
  archive with the per-timestep roughness, so its fold reads the same winds as
  the others.

The flagged folds are reported with their defect beside every number.

**A country-level fold's units are the grid points of its national fleet.**
Each training month is aggregated with that year's capacities, from the
per-year grid files, and the test year with 2023's. The observation is the
energy-weighted national monthly capacity factor on both splits. A
turbine-level fold is scored nationally as the capacity-weighted mean of the
units observed that month, simulation and observation over the same units.

## Arms

Every arm, for every fold, uses the same settings:
- five seeds (0, 1, 2, 3, 42) and 60 epochs;
- linear heads and the power-law profile on the measured shear;
- density, wake and bound scale off, off and 1.0;
- the licensed curve library through `input/combined`.

| Arm | Trained on | Heads see |
|---|---|---|
| `uncorrected` | nothing | nothing: ERA5 through the power curves, log law on the roughness |
| `transfer` | the pool, never the fold | terrain and fleet features |
| `features-off` | the same pool | zeros, so each learned quantity is one global constant, while the speed-up keeps its relief pin |
| `in-country` | the fold's own training years | terrain and fleet features |

**`in-country` is the ceiling, not a transfer arm.** **`features-off` is
identical across seeds by construction**, because seeds perturb head weights
that multiply zero inputs. Its seed spread is zero and is reported as such.

**One change from the reproduced configuration, declared here.** The fleet
features drop the capacity density within 10 km. The three kept are:
- capacity density within 50 km;
- the offshore flag;
- hub height.

A country grid point sits about 50 km from its neighbours and carries every
megawatt nearest to it, so its 10 km density measures aggregation rather than a
fleet. A head given it could tell the two data tiers apart and call that a
fleet property, which is the failure the published model removed raw capacity
for.

## The pool

- **Study A, Europe.** A gated fold trains on the other eight gated countries.
  A flagged fold trains on all nine.
- **Study B, world.** The same folds and arms, with US, BR, AR, AU-NEM, CL and
  NZ added to every fold's pool. Their scorecard configs are `us_k250`,
  `br_k60`, `ar_k10`, `au_nem_k45`, `cl_k10` and `nz_k7`.

Study B runs after Study A's full table has been shown, and is fixed here.

## Metrics

**Primary:** for every fold, national monthly capacity-factor RMSE on its test
year, scored on the months every arm can score. MBE, MAE, Pearson r and the
month count are reported beside it, as a scorecard country row reports them.
Per arm, the fold's figure is the mean over seeds.

**Secondary:** for turbine folds, per-unit RMSE on common unit-months, as in
the rerun. It is reported for every arm and gated nowhere.

**Recovery ratio** per fold, from mean squared errors averaged over seeds:
`(MSE_uncorrected - MSE_transfer) / (MSE_uncorrected - MSE_in-country)`. It is
undefined when `in-country` does not beat `uncorrected`.

## Gates, Study A

Read in order. A gate is not read until the one before it has been.

| Gate | Requirement | Outcome |
|---|---|---|
| **G0** records | The cache and study manifests record `git_dirty: false` and the same commit, which contains this document. The library has `power_curves_sha256` `689cfee71dc9e1aa5408cff4e2dbf5c205e30bd5f0b416ba8ee99691391ee00d`. Every cache records zero substituted units and `era5_roughness.applied` of `derived`. Every country cache resolves each year's grid by its own name. **A failure means no metric is read until it is explained.** | **Pass.** Cache and study manifests: `5960efd`, `git_dirty: false`, and the commit contains this document. Library hash matches. No substituted unit, `derived` in all 36 caches, every per-year grid resolved by its own name. |
| **G1** floor | `transfer` national RMSE below `uncorrected` in at least 7 of the 9 gated folds, and in no gated fold above `uncorrected` by more than 10% relative. | **Fail.** Below in 8 of 9; FR above by 78% (0.0362 against 0.0204). |
| **G2** recovery | The recovery ratio is at least 0.5 in more than half of the gated folds where it is defined. Not readable, and reported as such, if fewer than 5 are defined. | **Pass.** Defined in 9 of 9, at least 0.5 in 7 (not UK, FR). |
| **G3** place | `transfer` national RMSE is below `features-off` by more than max(0.002, twice the square root of the mean of the two arms' seed variances) in at least 5 of 9 gated folds. | **Pass.** 7 of 9 (not FR, UK). |

### What each outcome means

- **G1 fails:** the physics-informed model does not beat doing nothing in
  countries it has not seen. That is a negative result for its stated purpose,
  and it is published as one.
- **G2 fails with G1 passing:** transfer helps, but recovers less than half of
  what fitting in the country achieves. A country with observations should
  still fit its own.
- **G3 fails:** whatever transfers is carried by global constants, an average
  efficiency, spread and speed-up, not by anything the terrain or fleet
  features locate. That is consistent with the surviving explanation in
  `method-why-corrections-do-not-transfer.md`, now tested in capacity-factor
  space.
- **G3 passes:** something tied to place transfers across borders, which the
  coefficient-space studies never found. That would be the first evidence
  against the surviving explanation, and it would need Study B and a finer
  wind product before it is claimed.

## Gate, Study B

| Gate | Requirement | Outcome |
|---|---|---|
| **W1** more climates | B's `transfer` national RMSE is below A's by more than max(0.002, twice the square root of the mean of the two runs' seed variances) in at least 5 of 9 gated folds. | **Fail.** 1 of 9 (FR); B worse beyond the margin in 4 (DE, IE, IT, NO). |

B's G1 to G3 are computed and reported, not gated.

## Registered predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | G1 passes. | **Failed**, on FR alone. |
| 2 | G2 fails: in most defined folds, transfer recovers less than half of the in-country gain. | **Failed.** G2 passed. |
| 3 | G3 fails: `features-off` is within the margin of `transfer` in at least 5 of 9 gated folds. | **Failed.** G3 passed, 7 of 9. |
| 4 | Across gated folds, the transfer arm's national RMSE reduction from `uncorrected` rises with the uncorrected arm's absolute national MBE (Spearman above 0). | **Held.** Spearman +0.92 across the nine gated folds. |
| 5 | W1 fails, as the regime-coverage elimination predicts. | **Held.** W1 failed, 1 of 9. |

## What this does not compare against, and why

- **The scorecard's affine country rows** are an in-country reference on a
  different pipeline: a harness target that averages instantaneous ratios, the
  2021 snapshot fleet, and the open library, which substituted a 100 kW curve
  for every grid point. They are quoted beside the table and gated nowhere.
- **The thesis's interpolation and machine-learning leave-one-country-out**
  exist in coefficient space only. Scoring them in capacity factor would need
  the chapter pool refitted on today's pipeline, costed as P1 in
  [`manuscript-chapters-45.md`](https://github.com/ellyess/PyVWF/blob/ce20ead0716ffcbe2ad132d616f14c288cbc679d/docs/design/manuscript-chapters-45.md). That is follow-on work, not a
  comparator here.

## Caveats written in advance

- **One test year per fold,** and three turbine folds test on 2019 or 2020
  while nine country folds test on 2023.
- **A national series is 12 test months.** One month moves a fold's RMSE, and
  the seed spread does not measure that.
- **Country grids carry one curve and one hub height per country,** all typed
  onshore. BE's national series includes offshore generation its grid does not
  represent.
- **Regions are weighted equally in the loss, as published,** but a turbine
  region's loss is over unit-months and a country region's over national months.
  The two statistics differ in scale.
- **Screening-level throughout.**

## Commands

From the repository root, on a clean tree, one process at a time. Each script
refuses a dirty tree.

```bash
PYVWF_INPUT=input/combined PYTHONPATH=src /opt/anaconda3/bin/python -u \
  scripts/pinn/build_cache.py --out output/pinn_loco_2026-09-16/cache \
  --regions DK DE UK FR BE ES IE IT NO SE PT NL US BR AR AU-NEM CL NZ \
  --config DK=configs/regions/scorecard/dk_k100.toml \
  --config DE=configs/regions/scorecard/de_k100.toml \
  --config UK=configs/regions/scorecard/uk_k50.toml \
  --config FR=configs/regions/scorecard/fr_country.toml \
  --config BE=configs/regions/scorecard/be_country.toml \
  --config ES=configs/regions/scorecard/es_country.toml \
  --config IE=configs/regions/scorecard/ie_country.toml \
  --config IT=configs/regions/scorecard/it_country.toml \
  --config NO=configs/regions/scorecard/no_country.toml \
  --config SE=configs/regions/scorecard/se_country.toml \
  --config PT=configs/regions/scorecard/pt_country.toml \
  --config NL=configs/regions/pinn_loco/nl_country.toml \
  --config US=configs/regions/scorecard/us_k250.toml \
  --config BR=configs/regions/scorecard/br_k60.toml \
  --config AR=configs/regions/scorecard/ar_k10.toml \
  --config AU-NEM=configs/regions/scorecard/au_nem_k45.toml \
  --config CL=configs/regions/scorecard/cl_k10.toml \
  --config NZ=configs/regions/scorecard/nz_k7.toml \
  --registration docs/findings/method-physics-informed-loco-prereg.md
```

Study A, with the defaults `--folds` of all twelve, `--pool` of the nine gated
countries and `--flagged SE PT NL`:

```bash
PYVWF_INPUT=input/combined PYTHONPATH=src /opt/anaconda3/bin/python -u \
  scripts/pinn/loco.py --tag europe \
  --cache output/pinn_loco_2026-09-16/cache --out output/pinn_loco_2026-09-16/europe \
  --config DK=configs/regions/scorecard/dk_k100.toml \
  --config DE=configs/regions/scorecard/de_k100.toml \
  --config UK=configs/regions/scorecard/uk_k50.toml \
  --config FR=configs/regions/scorecard/fr_country.toml \
  --config BE=configs/regions/scorecard/be_country.toml \
  --config ES=configs/regions/scorecard/es_country.toml \
  --config IE=configs/regions/scorecard/ie_country.toml \
  --config IT=configs/regions/scorecard/it_country.toml \
  --config NO=configs/regions/scorecard/no_country.toml \
  --config SE=configs/regions/scorecard/se_country.toml \
  --config PT=configs/regions/scorecard/pt_country.toml \
  --config NL=configs/regions/pinn_loco/nl_country.toml \
  --registration docs/findings/method-physics-informed-loco-prereg.md
```

Study B:

```bash
PYVWF_INPUT=input/combined PYTHONPATH=src /opt/anaconda3/bin/python -u \
  scripts/pinn/loco.py --tag world --extra-pool US BR AR AU-NEM CL NZ \
  --cache output/pinn_loco_2026-09-16/cache --out output/pinn_loco_2026-09-16/world \
  --config DK=configs/regions/scorecard/dk_k100.toml \
  --config DE=configs/regions/scorecard/de_k100.toml \
  --config UK=configs/regions/scorecard/uk_k50.toml \
  --config FR=configs/regions/scorecard/fr_country.toml \
  --config BE=configs/regions/scorecard/be_country.toml \
  --config ES=configs/regions/scorecard/es_country.toml \
  --config IE=configs/regions/scorecard/ie_country.toml \
  --config IT=configs/regions/scorecard/it_country.toml \
  --config NO=configs/regions/scorecard/no_country.toml \
  --config SE=configs/regions/scorecard/se_country.toml \
  --config PT=configs/regions/scorecard/pt_country.toml \
  --config NL=configs/regions/pinn_loco/nl_country.toml \
  --config US=configs/regions/scorecard/us_k250.toml \
  --config BR=configs/regions/scorecard/br_k60.toml \
  --config AR=configs/regions/scorecard/ar_k10.toml \
  --config AU-NEM=configs/regions/scorecard/au_nem_k45.toml \
  --config CL=configs/regions/scorecard/cl_k10.toml \
  --config NZ=configs/regions/scorecard/nz_k7.toml \
  --registration docs/findings/method-physics-informed-loco-prereg.md
```

## Cost

| Step | Cost |
|---|---|
| Caches, 18 regions | under half an hour |
| Study A, 12 folds | about six hours, detached |
| Study B | about eight hours, detached |
| Reading the gates and the full tables | the larger share, and not compute |

## Study A run record, 2026-09-17

Filled in after the run. The table and its interpretation are in
`method-physics-informed-loco.md`.

**Data.** `output/pinn_loco_2026-09-16/europe/`: `loco_europe_raw.csv` sha256
`86ae858f4eb9120c33c9b7b867038eb275fc652a50a4fced10e3650edc9d59b9`,
`loco_europe_national.csv` sha256
`fae6f99ef85176ece5b1e96687c01ad94cb85d799cc9f15e2cc42bedce9b43bf`,
`loco_europe_gates.json` sha256
`d8e1cf48e77f1c0bf04187e6542dd45e276ebed05bffc9913b61434f34098c32`. The gates
were recomputed independently of the script from the raw table and agree.

**One earlier launch was stopped before the run that counts, and produced no
fold result.** It was stopped at the maintainer's request 6 minutes in, during
the DK fold. Its manifest and log are kept in
`output/pinn_loco_2026-09-16/europe_interrupted_2026-09-16T1717/`. The run that
counts started at 22:05 BST on 2026-09-16 with the registered command and
finished with exit 0 about 7 hours later.

**Deviation: the features-off arm's seed spread is 6.2e-6 at most, not zero.**
Its heads see zeros, but the seed also orders the turbine regions' unit
minibatches, so floating-point sums run in a different order. It changes no
gate: G3's margin has a floor of 0.002.

**Not in any gate, recorded.**
- **UK transfer, seed 1:** national RMSE 0.101 against 0.012 to 0.030 for the
  other four seeds, mean speed-up 1.54 against 0.99 to 1.12, at a similar
  training loss. That seed alone sets the UK's seed spread of 0.036 and its G3
  margin of 0.051.
- **Common-month scoring** excluded no month in any fold.
- **Off-curve shares** are at most 0.002% of capacity-weighted unit-days.
- **Units with no wind:** DK drops 15 training units and 47 test units, and no
  other country drops any.

## Study B run record, 2026-09-18

Filled in after the run. The table and its interpretation are in
`method-physics-informed-loco.md`.

**Data.** `output/pinn_loco_2026-09-16/world/`: `loco_world_raw.csv` sha256
`ffc67e4ffcf106ee400edfef2959d263f6e904091ffc1ded56ba74e304e94974`,
`loco_world_gates.json` sha256
`495ec1772a395b01c2196e3834d2e38679c7c2ceeda0afb1341728e09d5a8134`. W1 was
computed from the two raw tables, independently of the script.

**The first launch died part-way and was repeated in full, as committed.** It
started at 12:35 BST on 2026-09-17 from `9d1767c`. It completed the DK, DE, UK and
FR folds and was killed during BE, some time after 14:53, with no traceback and
no exit line. The system log shows no memory kill and no sleep, so the cause is
not established. Its outputs are kept, unused, in
`output/pinn_loco_2026-09-16/world_interrupted_2026-09-17T1235/`. The run that
counts started at 23:55 BST from the same commit, with the registered command,
and finished with exit 0.

**Records.** G0's fields hold for Study B: its manifest records `9d1767c` and
`git_dirty: false`, the library hash matches, and there is no substituted unit.
Roughness is `derived` in every cache. The `uncorrected` and `in-country` arms
match Study A exactly, as they must, since neither uses the pool. No month was
excluded, and the off-curve share is zero.

**Study B's own G1 to G3, reported and not gated:** G1 9 of 9 below uncorrected
and none worse by more than 10%; G2 8 of 9; G3 7 of 9.

## Seen before registration

**One real figure was computed before this document was committed, and it is
disclosed here.** To exercise the country-level cache on real files, Belgium's
training and test caches were built in a session scratchpad, and its
`uncorrected` national series was scored: RMSE 0.1064 on 2015 to 2021, 0.1226
on 2023. No fitted arm was run on any real data, and no other country was
touched. The scorecard's BE uncorrected figure is 0.3399. It was simulated on a
100 kW distributed-wind curve substituted for every grid point, which is why
this study's uncorrected arm is not comparable to the scorecard's.

## Committed in advance

- Every gate and prediction is reported whichever way it goes, with the full
  national table for all twelve folds, flagged folds included, before any
  interpretation.
- A run that stops part-way is repeated in full from a clean tree, not resumed.
- A deviation from this design is recorded with its date, and with whether it
  came before or after a result was seen.
