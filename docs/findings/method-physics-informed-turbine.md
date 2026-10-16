# Does a terrain correction learned from turbine and farm data transfer between regions?

**Date:** 2026-10-16
**Scope:** the nine turbine-level regions, each held out in turn from a model
trained on the other eight (seven gated, DK the negative control, NZ reported
only), on the scorecard configs and the licensed library (`input/combined`,
`power_curves.csv` sha256 `689cfee7…`, no substituted unit in any of the 38
caches, roughness derived in all). Registered in
`method-physics-informed-turbine-prereg.md` (`4d671c7`, 2026-09-18), with
deviations of 2026-09-20 and 2026-10-16 recorded before any fit; run at
`285fd80` from a clean tree on 2026-10-16. Issue #12. Terms follow
`CONTEXT.md`.

**T0 passes; of the five skill gates, T1 and T4 pass and T2, T3 and T5 fail.
Because T3, the negative control, fails, the registration fixes that T1 and T2
are not read as evidence about terrain, whatever their counts.** In Denmark,
the flat control, the `full` arm lowers spatial RMSE against `no-terrain` by
0.021, five times the 0.0043 margin, while `relief-only` stays within its
margin. `full` beats `uncorrected` on per-unit RMSE in all seven gated
regions, and beats the atlas arm `gwa-ratio` beyond the margin in only two of
seven (BR, AU-NEM). The framing of these results is left to the maintainer.

Fold years: DK 2015-19, test 2020; DE and UK 2015-18, test 2019; US 2019-21,
test 2022; AU-NEM 2020-22, test 2023; BR, AR and CL 2021-23, test 2024; NZ
2019-23, test 2024. One test year per fold.

## T0

| Requirement | Recorded |
|---|---|
| Study manifest `git_dirty: false` at a commit with this registration and its code | `285fd80`, `git_dirty: false` |
| Cache manifest `5960efd`, `git_dirty: false` | `5960efd`, false; DE from `cache_mastr` at `14722be`, false (deviation of 2026-10-16) |
| Library hash `689cfee71dc9…` | `689cfee71dc9e1aa…` |
| No substituted unit; roughness `derived` in every cache | 38 caches: 0 substituted, `derived` in all |

**Pass.**

## The full tables

Test year, mean over five seeds (`uncorrected` is deterministic, one value).
Data: `output/pinn_turbine_2026-09-18/study/turbine_primary_raw.csv`, sha256
`fb88cf7f…`; whole-fleet rows only, as the gates read them.

**Spatial RMSE**

| Fold | uncorrected | no-terrain | relief-only | full | gwa-ratio | in-region |
|---|---|---|---|---|---|---|
| DE | 0.05832 | 0.05223 | 0.05104 | 0.05061 | 0.04709 | 0.04639 |
| UK | 0.13056 | 0.12265 | 0.11696 | 0.12214 | 0.11435 | 0.11455 |
| US | 0.08790 | 0.07669 | 0.07447 | 0.08385 | 0.07436 | 0.07373 |
| BR | 0.10876 | 0.09467 | 0.08878 | 0.08404 | 0.08824 | 0.08325 |
| AR | 0.14115 | 0.12733 | 0.11490 | 0.11266 | 0.11032 | 0.11301 |
| AU-NEM | 0.10571 | 0.08843 | 0.06666 | 0.05782 | 0.06379 | 0.05540 |
| CL | 0.10032 | 0.08706 | 0.09377 | 0.10242 | 0.10042 | 0.06410 |
| DK (control) | 0.07201 | 0.09915 | 0.09731 | 0.07772 | 0.08169 | 0.05724 |
| NZ (reported) | 0.13974 | 0.11250 | 0.08278 | 0.05429 | 0.05288 | 0.12139 |

**Level RMSE**

| Fold | uncorrected | no-terrain | relief-only | full | gwa-ratio | in-region |
|---|---|---|---|---|---|---|
| DE | 0.05040 | 0.03949 | 0.04990 | 0.03080 | 0.03667 | 0.00697 |
| UK | 0.04280 | 0.00943 | 0.05338 | 0.03267 | 0.04368 | 0.03067 |
| US | 0.03332 | 0.02468 | 0.05224 | 0.02069 | 0.04782 | 0.01178 |
| BR | 0.06919 | 0.07478 | 0.03922 | 0.04075 | 0.04840 | 0.02139 |
| AR | 0.02285 | 0.03939 | 0.05819 | 0.06616 | 0.07803 | 0.01041 |
| AU-NEM | 0.02349 | 0.01332 | 0.04398 | 0.03539 | 0.05587 | 0.01344 |
| CL | 0.03473 | 0.02173 | 0.03863 | 0.02435 | 0.06059 | 0.01651 |
| DK (control) | 0.11759 | 0.04591 | 0.03636 | 0.01866 | 0.02851 | 0.02991 |
| NZ (reported) | 0.06339 | 0.09634 | 0.06637 | 0.01990 | 0.01005 | 0.04345 |

**Per-unit RMSE**

| Fold | uncorrected | no-terrain | relief-only | full | gwa-ratio | in-region |
|---|---|---|---|---|---|---|
| DE | 0.08598 | 0.07513 | 0.08145 | 0.07042 | 0.07047 | 0.05960 |
| UK | 0.14560 | 0.13172 | 0.13741 | 0.13514 | 0.13144 | 0.12800 |
| US | 0.10979 | 0.09651 | 0.10593 | 0.10251 | 0.10289 | 0.09115 |
| BR | 0.13860 | 0.12997 | 0.10574 | 0.10264 | 0.10938 | 0.09496 |
| AR | 0.15117 | 0.14100 | 0.13733 | 0.13852 | 0.14308 | 0.12218 |
| AU-NEM | 0.11531 | 0.09741 | 0.08937 | 0.07832 | 0.09323 | 0.06854 |
| CL | 0.12249 | 0.10727 | 0.11520 | 0.12067 | 0.13111 | 0.08727 |
| DK (control) | 0.14750 | 0.12293 | 0.11812 | 0.09551 | 0.10213 | 0.08234 |
| NZ (reported) | 0.15674 | 0.15105 | 0.11053 | 0.06487 | 0.06066 | 0.13302 |

## Gates

Differences are in spatial RMSE, positive where the first arm is lower; the
margin is the registered max(0.002, 2 × pooled seed SD). Per-fold detail:
`output/pinn_turbine_2026-09-18/study/gate_detail.txt`.

| Fold | T1 relief-only vs no-terrain | T2 full vs relief-only | T5 full vs gwa-ratio | T4 full / uncorrected per-unit |
|---|---|---|---|---|
| DE | +0.0012 (0.0020) fail | +0.0004 (0.0020) fail | -0.0035 (0.0020) fail | 0.819 |
| UK | +0.0057 (0.0020) pass | -0.0052 (0.0058) fail | -0.0078 (0.0058) fail | 0.928 |
| US | +0.0022 (0.0020) pass | -0.0094 (0.0030) fail | -0.0095 (0.0030) fail | 0.934 |
| BR | +0.0059 (0.0020) pass | +0.0047 (0.0026) pass | +0.0042 (0.0026) pass | 0.741 |
| AR | +0.0124 (0.0020) pass | +0.0022 (0.0023) fail | -0.0023 (0.0023) fail | 0.916 |
| AU-NEM | +0.0218 (0.0020) pass | +0.0088 (0.0026) pass | +0.0060 (0.0026) pass | 0.679 |
| CL | -0.0067 (0.0020) fail | -0.0086 (0.0020) fail | -0.0020 (0.0020) fail | 0.985 |

| Gate | Requirement | Outcome |
|---|---|---|
| T0 | records | **Pass** |
| T1 | relief-only beats no-terrain beyond the margin in 5 of 7 | **Pass**, 5 of 7 (not DE, not CL) |
| T2 | full beats relief-only beyond the margin in 5 of 7 | **Fail**, 2 of 7 (BR, AU-NEM) |
| T3 | in DK, full and relief-only each within the margin of no-terrain | **Fail**: full -0.0214 against a margin of 0.0043; relief-only -0.0018 against 0.0020, within |
| T4 | full per-unit below uncorrected in 5 of 7, none above by more than 10% | **Pass**, 7 of 7 below, none above |
| T5 | full beats gwa-ratio beyond the margin in 5 of 7 | **Fail**, 2 of 7 (BR, AU-NEM) |

**The registered readings, applied as written:** T3 failing means "a flat
region gains from a terrain arm, so the arm is fitting something else", and T1
and T2 "are then not read as evidence about terrain, whatever their counts".
T4 passing means the full model does not harm per-unit accuracy in the regions
it has not seen. T5 failing means "a free atlas does at least as well, and the
learned terrain term is not worth its data where an atlas exists".

## Registered predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | T1 passes | **Held** (5 of 7), but see T3 |
| 2 | T2 fails | **Held** (2 of 7) |
| 3 | T3 passes: DK shows no terrain gain | **Failed**: `full` gains 0.021 in DK |
| 4 | BR shows the largest relief-only gain over no-terrain | **Failed**: AU-NEM's is largest (0.0218); BR's is third (0.0059), after AR (0.0124) |
| 5 | Secondary national scoring worse than Study A's transfer arm in the median over FR, BE, ES, IE, IT, NO | **Depends on the arm, which the prediction does not name**: median 0.0269 for `full` against Study A's 0.0353 (better); 0.0788 for `no-terrain` and 0.0537 for `relief-only` (worse) |
| 6 | T5 passes | **Failed** (2 of 7) |
| 7 | gwa-ratio beats no-terrain beyond the margin in fewer than 4 of 7 | **Failed**: 6 of 7 (all but CL) |

## Secondary scoring, reported and gated nowhere

The three arms fitted once per seed on all nine turbine-level regions, applied
to the country folds' test years (2023; training 2015-21, IE 2017-21), as
national monthly RMSE, mean over seeds. Data:
`output/pinn_turbine_2026-09-18/study/turbine_primary_secondary.csv`. Study
A's transfer arm from `method-physics-informed-loco.md`, on the same caches.

| Fold | uncorrected | no-terrain | relief-only | full | Study A transfer |
|---|---|---|---|---|---|
| FR | 0.02037 | 0.01896 | 0.01918 | 0.02033 | 0.0362 |
| BE | 0.12262 | 0.08772 | 0.06168 | 0.03354 | 0.0343 |
| ES | 0.07095 | 0.06995 | 0.03566 | 0.01906 | 0.0188 |
| IE | 0.04524 | 0.01671 | 0.04569 | 0.01769 | 0.0190 |
| IT | 0.15444 | 0.15000 | 0.11052 | 0.11350 | 0.0432 |
| NO | 0.12857 | 0.13828 | 0.10219 | 0.13553 | 0.0513 |
| SE, flagged | 0.09844 | 0.09955 | 0.06001 | 0.08159 | 0.1132 |
| PT, flagged | 0.18075 | 0.17637 | 0.13162 | 0.13970 | 0.1112 |
| NL, flagged | 0.21399 | 0.16119 | 0.08244 | 0.07520 | 0.0918 |

The country caches carry their 2023 grids as built on 2026-09-16, before the
fleet-register rebuild (#76) and the offshore grid points (#100), and Italy and
Portugal are suspended in the scorecard; these figures are as of those grids.

## Reported check, not a gate: did moving Germany's locations help?

Registered on 2026-09-20 before the run. The same DE fold, arms and seeds:

| | full | gwa-ratio | in-region | no-terrain | relief-only | uncorrected |
|---|---|---|---|---|---|---|
| 1. Mixed fleet, MaStR run, spatial | 0.05061 | 0.04709 | 0.04639 | 0.05223 | 0.05104 | 0.05832 |
| 2. Matched units, MaStR run, spatial | 0.04799 | 0.04257 | 0.04389 | 0.05038 | 0.04854 | 0.05625 |
| 3. Matched units, centroid run, spatial | 0.04796 | 0.04505 | 0.04406 | 0.05075 | 0.04923 | 0.05655 |
| Whole fleet, centroid run, spatial | 0.05057 | 0.04827 | 0.04655 | 0.05245 | 0.05146 | 0.05850 |
| 2. Matched units, MaStR run, per-unit | 0.06798 | 0.06827 | 0.05584 | 0.07288 | 0.07982 | 0.08417 |
| 3. Matched units, centroid run, per-unit | 0.06709 | 0.06718 | 0.05604 | 0.07305 | 0.08045 | 0.08429 |

The matched units are 60.0% of the DE test fleet's capacity. Data:
`turbine_de-centroid_raw.csv` (sha256 `79435002…`) beside the primary file.
The registered reading table ("better", "no change", "worse" for 2 against 3)
fixed no threshold, so no reading is assigned here.

## Caveats

- One test year per fold; screening-level, not an accredited yield
  assessment.
- The study ran four weeks after registration on code that had moved, with a
  defect in the German fold's gating fixed before the run; the deviation of
  2026-10-16 lists each change. The deviations, and the decision to run, were
  taken by Claude Code overnight and had not been reviewed by the maintainer
  when this was written.
- The German fold's fleet is of mixed location resolution: 70% of training
  capacity on MaStR coordinates, the rest on postcode centroids.
