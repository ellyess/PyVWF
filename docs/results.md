# Results

Every result in PyVWF's research record, one entry each, negative ones beside
positive ones. Each entry states its training years and its single test year,
or that none is involved, or that its record does not state them. It names any
gate that failed and links the full record, which is at tag
`archive/research-2026-10-27` (see {doc}`archive`). Per-region accuracy is in
the {doc}`scorecard`; this page covers the studies behind it.

The entries summarise their records as those records stand after every dated
correction notice. Where a record and this page differ, the record holds the
detail. Where two records differ from each other, the
[last section](#where-two-records-disagree) says so. Everything is
screening-level validation on one test year per region, not an accredited
yield assessment.

**Outcomes at a glance.** Of the 35 entries below, 5 are positive, 14
negative, 9 mixed, 1 diagnostic, and 6 void, stopped, never run or withdrawn.
A mixed entry has a passed gate or standing finding on one side and a failed
gate or withdrawn figure on the other.

| Area | Entry | Outcome |
|---|---|---|
| Affine correction by region | [United States and Brazil](#united-states-and-brazil) | Positive after a defect fix |
| | [New Zealand](#new-zealand) | Mixed: gain not resolved |
| | [Australia (NEM)](#australia-nem) | Withdrawn: corrected figures |
| | [Chile and Argentina](#chile-and-argentina) | Negative: limited skill |
| | [What generalises from Australia](#what-generalises-from-australia) | Mixed |
| | [Harness reproduces the legacy pipeline](#harness-reproduces-the-legacy-pipeline) | Positive |
| Fit health and cluster count | [Cluster count per region](#cluster-count-per-region) | Mixed |
| | [Denmark's cluster count](#denmarks-cluster-count) | Positive |
| | [Leakage-free cluster selection](#leakage-free-cluster-selection) | Mixed: C-G2 passed, C-G3 failed |
| | [Scalar bounds and minimum cluster size](#scalar-bounds-and-minimum-cluster-size) | Mixed: ceiling set, G1 failed |
| | [Scalar and offset are not identified](#scalar-and-offset-are-not-identified) | Negative |
| Inputs and method | [Power curve library](#power-curve-library) | Negative: G1 and G3 failed |
| | [European re-run](#european-re-run) | Mixed: two gates not re-measured |
| | [Roughness treatment](#roughness-treatment) | Mixed: adopted on method, not accuracy |
| | [Roughness back-fill within each year](#roughness-back-fill-within-each-year) | Negative: G4 breached, adopted |
| | [Hourly resolution](#hourly-resolution) | Negative: all gates failed |
| | [Terrain and the US wind deficit](#terrain-and-the-us-wind-deficit) | Positive |
| Country level | [The country-level estimator](#the-country-level-estimator) | Negative |
| | [Joint-fit reachability](#joint-fit-reachability) | Diagnostic |
| | [Per-year fleet weighting](#per-year-fleet-weighting) | Stopped |
| | [One cluster or many per country](#one-cluster-or-many-per-country) | Never run |
| Transfer and interpolation | [Machine-learning transfer](#machine-learning-transfer) | Negative |
| | [Why corrections do not transfer](#why-corrections-do-not-transfer) | Negative |
| | [Grid validation was in-sample](#grid-validation-was-in-sample) | Negative |
| | [The distance mask](#the-distance-mask) | Negative |
| | [Leave-one-country-out interpolation](#leave-one-country-out-interpolation) | Void |
| | [Offshore pool and domain split](#offshore-pool-and-domain-split) | Void (two studies) |
| | [Netherlands holdout](#netherlands-holdout) | Never run |
| Physics-informed correction | [Zero-shot transfer](#zero-shot-transfer) | Mixed: Q2 failed |
| | [Reproduction](#reproduction) | Positive |
| | [Leave one country out](#leave-one-country-out) | Negative: G1 and W1 failed |
| | [Terrain at turbine level](#terrain-at-turbine-level) | Negative: T2, T3, T5 failed |
| | [Per-unit wake coefficient](#per-unit-wake-coefficient) | Negative: K0 failed |
| Tooling | [Review agents](#review-agents) | Negative: bar not passed |
| | [Dataset survey](#dataset-survey) | Mixed: 2 of 4 confirmed |

## Affine correction by region

### United States and Brazil

**Years:** US training 2019-21, test 2022; BR training 2021-23, test 2024.
**Record:** [`region-us-br.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/region-us-br.md).

The first US run came out worse than no correction. It exposed a defect that
diluted observed means by the reporting fraction, giving a scalar of 0.47
where 1.09 was correct. After the fix the US improves from MAE 0.0796 to
0.0741 at ten clusters; the pre-fix US sweep must not be cited. Brazil was
unaffected and improves from 0.1102 to 0.0702. A single cluster is worse than
uncorrected in both. The registered prediction that geographic distance would
move the US most was wrong, and both clustering changes landed inside about 3%
noise. The near-zero US bias hides opposite regional biases (ERCOT and SPP
+0.0391, elsewhere -0.0241), and US curtailment remains unscreened.

### New Zealand

**Years:** training 2019-23, test 2024.
**Record:** [`region-nz.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/region-nz.md).

Over 12 farms, the correction at seven clusters lowers RMSE from 0.1568 to
0.1046, MAE from 0.1426 to 0.0761, and lifts r from 0.639 to 0.742, as
re-evaluated after the 2026-09-24 turbine-ordering fix. **The gain is not
resolved:** resampling the test farms gives an RMSE gain of 0.052 with a 95%
interval of -0.032 to 0.115. Before the fix, one farm carried 61% of the
corrected squared error, and that share was not recomputed. Only eight farms
reach training, so seven clusters is close to one farm per cluster.

### Australia (NEM)

**Years:** training 2020-22, test 2023.
**Record:** [`region-au-nem.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/region-au-nem.md).

Uncorrected ERA5 carries almost no level bias over the NEM fleet (MBE -0.024)
but over-amplifies South Australia's seasonal cycle: June is 1.617 times the
annual mean against 1.394 observed. **Every corrected figure was withdrawn on
2026-09-24**, including the pre-specified cycle gate first reported as passed
and the reported cost in farm-level skill: a turbine-ordering defect had given
units other units' curves and capacities. Whether the gate passes is unknown
until the validation is re-run. Curtailment in South Australia cannot be
separated from resource bias. The scorecard's AU-NEM row was re-evaluated
after the fix.

### Chile and Argentina

**Years:** test 2024 for both; the record does not state the training years.
**Record:** [`region-south-america.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/region-south-america.md).

The correction removes the mean bias but adds limited skill. ERA5 exaggerates
the north-south wind gradient while the real fleets run at a nearly flat
capacity factor. Chile's correction does not lower RMSE even on a training
year (0.1208 to 0.1209). On common plants its test-year gain is 0.006, with an
interval of -0.007 to 0.021, so an earlier "usable" claim was withdrawn on
2026-09-11. Matched curves made the northern under-prediction worse, which
rules out the wrong-turbine explanation. Argentina improves from 0.151 to
0.133 only after its capacity denominators were rebuilt. A loss-term
hypothesis did not hold. The record attributes the residual northern error to
the reanalysis's resolution, which a finer wind product would test.

### What generalises from Australia

**Years:** the original table's years are not stated; the re-measurement
trained DK 2015-19 and AU-NEM 2020-22, and tested DK to AU on 2023 and AU to
DK on 2020.
**Record:** [`method-generalisation.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-generalisation.md).

The synthesis argued that the correction helps where reanalysis bias is
level-dominated and that correction factors do not port between regions. Its
Denmark result stands: MBE +0.121, corrected RMSE 0.160 to 0.099. A 2026-09-24
notice withdrew every corrected Australia figure, its cycle-tracking gate and
the DK to AU transfer. A re-run found DK to AU lowers RMSE (0.1153 to 0.1128)
but worsens MBE and correlation, and **AU to DK worsens RMSE** (0.1482 to
0.1606). Correction parameters and cluster configuration do not generalise.

### Harness reproduces the legacy pipeline

**Years:** DK training 2015-19, test 2020; DE 2015-18, test 2019; NL and FR
2015-19, test 2023.
**Record:** [`method-harness-regression.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-harness-regression.md).

The refactored harness reproduced the legacy pipeline bit for bit on four
regions: correction factors and corrected capacity-factor frames match to a
worst difference of zero. The skill figures were informational, not gates;
they include France's corrected RMSE of 0.019 against 0.018 uncorrected, and
the Netherlands reproducing a known overshoot. The legacy path was removed on
2026-09-24, and `tests/test_harness_corrections.py` still pins the
equivalence.

## Fit health and cluster count

### Cluster count per region

**Years:** each region's scorecard test year; the record does not state the
training years.
**Record:** [`method-cluster-count.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-cluster-count.md).

Sweeping the cluster count in nine turbine-level regions gave a knee per
region (NZ 5, CL 8, AR 12, AU-NEM 15, BR 20, US 25, DK 100, UK 50, DE 50),
which became the shipped cluster lists. **The curves were read on the test
year,** and the differences between counts were never resampled. Notices of
2026-09-11 leave the UK and NZ readings untested and say Chile was scored on
different plants at each count. Chile doubles its RMSE at three and five
clusters, and no count is shown safe there: cluster count does not predict fit
health. The AU-NEM and NZ rows predate the 2026-09-24 fix.

### Denmark's cluster count

**Years:** training 2015-19, test 2020.
**Record:** [`method-cluster-count-dk.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-cluster-count-dk.md).

Cluster count is the dominant lever for Denmark's onshore fleet. RMSE falls
from about 0.096 with one cluster to a floor of 0.0857 at 200 clusters, and
3,300 clusters buy only 0.0006 more. Time resolution moves RMSE by about
0.003. The correction roughly halves RMSE, from 0.158 to 0.085. A 2026-09-12
notice says this is not a reproduction of the published paper: roughness,
fleet size and curve library all differ. Counts were read on the test year.

### Leakage-free cluster selection

**Years:** the records do not state them; folds chain forward inside the
training years, and the test year is read once.
**Record:** [`method-cluster-selection.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-cluster-selection.md),
registered in [`method-cluster-selection-prereg.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-cluster-selection-prereg.md).

Choosing each configuration's cluster count from forward-chained training
folds with a one-standard-error rule beat a fixed 100 clusters in exactly 3 of
5 turbine-level rows, meeting C-G2 at its threshold. **It failed C-G3 against
the chapter's own counts, winning 1 of 5.** The rule's caution explains
Denmark offshore entirely; in UK onshore and Denmark onshore the chapter's
count lay outside the registered grid, which was not widened afterwards. Two
registered predictions were refuted and a third half failed. The selections
were not adopted.

### Scalar bounds and minimum cluster size

**Years:** a survey of all runs; the Chile test is test year 2024, training
years not stated.
**Record:** [`method-scalar-bounds.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-scalar-bounds.md).

A survey of 373,179 fitted wind scalars supports a ceiling of 3.0, which
separates healthy regions from known pathologies; `fit_quality` uses it. A
floor of 0.2 was also set, but its Belgian rationale was withdrawn on
2026-10-25, and it flags 29 of 60 Netherlands scalars, not the 57 first
stated. For Chile at ten clusters **no minimum cluster size met all three
gates:** the setting that caps the scalar at 39.3 fails the severity gate
(G1), and the one that caps it at 5.50 doubles RMSE against no correction.
The Chile table was superseded on 2026-09-19 by a rerun whose values are only
in #17.

### Scalar and offset are not identified

**Years:** the chapter-era pool's rows, each on its own training years and
one test year, not named; Belgian and French points trained 2015-21, tested
2023.
**Record:** [`method-correction-identifiability.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-correction-identifiability.md).

The correction's scalar and offset are not separately identified: each fit
matches one mean capacity factor with two parameters, and a tie-break chooses
the pair. In the four dense turbine-level rows, 99% or more of the joint
variation lies on one line (r -0.979 to -0.998). Reparameterising cannot
change a geometry-weighted interpolator's predictions. **Leave-one-country-out
R² was negative in every standing fold but Italy's** (+0.010, three points).
Belgian and French points are withdrawn (2026-10-25).

## Inputs and method

### Power curve library

**Years:** country rows test 2023 (training 2015-21 stated for BE and FR);
turbine rows test 2019 (DE, UK), 2020 (DK) and 2022 (US), training years not
stated.
**Record:** [`method-curve-library.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-curve-library.md),
registered in [`method-curve-library-prereg.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-curve-library-prereg.md).

Giving each country grid its own Vestas curve instead of the 100 kW fallback
lowered uncorrected mean bias in all eight country rows, by 0.082 to 0.220,
but helped RMSE only in the three that started over-producing. **G1 failed**
(3 of 7 countries against 4 required); the record says the failure stands
whatever the Belgian and French refits give, though its multi-cluster
denominators were not re-measured after the 2026-09-25 solver fix. At turbine
level G2a and G2b passed, every corrected change under 0.0012, and **G3
failed in all four rows:** ERA5 bias outweighs curve assignment. Real curves
turned stable scalars degenerate (Italy 0 to 8). P1 held; P4, P5 and P6 were
refuted; P2 and P2b are withdrawn (2026-10-25), and P3 was not re-measured.
Belgian and French corrected figures are withdrawn.

### European re-run

**Years:** test 2019 (DE, UK), 2020 (DK), 2023 (country rows); training
years stated only for BE and FR (2015-21).
**Record:** [`method-eu-rerun.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-eu-rerun.md),
registered in [`method-eu-rerun-prereg.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-eu-rerun-prereg.md).

Re-running the eleven European rows on per-timestep roughness moved corrected
RMSE by under 0.0002 in every comparison; only Denmark resolved (-0.000187).
The wider ERA5 box returned Spain, Italy and Portugal on real instead of
extrapolated winds. G0, G1, G2 and G5 passed, and predictions P4 and P5 were
refuted, but the 2026-09-25 notice lists the headline, G2, G5 and P3 to P5 as
resting on multi-cluster country fits that used the wrong solver and were not
re-measured. Norway's correction still did not help on that run. An ordinary
affine fit silently drops calm days that `fit_quality` does not see (all
twelve Italian pairs have negative offsets). Belgian and French absolute
figures are withdrawn (2026-10-25).

### Roughness treatment

**Years:** DK training 2015-19, test 2020; FR training 2015-21, test 2023.
**Record:** [`method-roughness-treatment.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-roughness-treatment.md),
registered in [`method-roughness-treatment-prereg.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-roughness-treatment-prereg.md).

PyVWF adopted the per-timestep roughness derivation for method fidelity and
comparability, **not accuracy.** On Denmark, G1 resolved in its favour by only
0.0002 in corrected RMSE, while uncorrected RMSE got worse by 0.0015. France
was consistent with zero and cannot show the effect at its 90 m hubs, so one
row decides the method, with no replication. G3 was withdrawn before any
Danish result, and P1 was refuted. France's figures have been withdrawn since
2026-10-25; the adoption rests on Denmark.

### Roughness back-fill within each year

**Years:** per row as in the scorecard (for example DK 2015-19 and 2020, FR
2015-21 and 2023).
**Record:** [`method-roughness-fill-prereg.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-roughness-fill-prereg.md).

Confining the derived roughness back-fill to each calendar year removes a
small leak of test-year ERA5 input into training, up to 742 cell-hours at unit
cells (Germany). G1, G2 and G3 passed. **G4 was breached:** France's
degenerate N=10 season fit moved mean bias by 1.61e-4 against a 1e-4 limit,
so adoption stopped as registered. The maintainer then adopted the change, a
decision against the registered consequence rather than a pass. The
registration had wrongly classed that fit as not degenerate. Scorecard rows
not rerun since stand on the old fill.

### Hourly resolution

**Years:** test 2024 (Chile, ten clusters); the record does not state the
training years.
**Record:** [`method-hourly-resolution.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-hourly-resolution.md).

At hourly resolution the Chile correction made the simulation worse: corrected
RMSE 0.2613 against 0.2263 uncorrected. **All three pre-registered gates
failed,** and the registered prediction was wrong. Dropping three degenerate
clusters moves the change from -15.5% to -1.9%: roughly neutral, still not
helpful. A notice of 2026-09-11 corrects the monthly reading too: on the same
plants the monthly gain cannot be told from zero. One region, one year, not
replicated.

### Terrain and the US wind deficit

**Years:** training 2019-21 (G1 to G4), test 2022 (G5).
**Record:** [`method-terrain-wind-deficit.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-terrain-wind-deficit.md),
registered in [`method-terrain-wind-deficit-prereg.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-terrain-wind-deficit-prereg.md).

On the US fleet, sub-grid terrain relief predicts the uncorrected
simulation's capacity-factor shortfall: Spearman rho 0.488 on the training
years, confirmed at 0.424 on the test year. All five registered gates passed,
including two controls against a curve explanation. G4 was first read as not
assessable on a miscomputed precondition. Only 356 of 1,091 plants have an
outcome, on gentler ground than the fleet, rho leaves most of the variance
unexplained, and a curtailment confound remains. The recommended terrain flag
and separate regime are not implemented (#36).

## Country level

### The country-level estimator

**Years:** training 2015-21 (IE 2017-21), test 2023.
**Record:** [`method-country-level.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-country-level.md).

The country-level path fits a different estimator from the turbine path: one
national observation against N cluster offsets is under-determined, and the
scalar flattens the spatial field. Its observation audit found the Dutch
series unusable (peak capacity factor never above 0.57) and repaired
Ireland's frozen register. **Every country-level skill figure in it was later
qualified:** it ran on a 100 kW fallback curve, fits with more than one
cluster used the wrong solver, and Belgium's and France's rows were withdrawn
on defective observations (2026-10-25). On that curve Norway's correction was
worse than none. The reading that offsets repair the scalar survives only at
turbine level (DK, NZ).

### Joint-fit reachability

**Years:** training 2015-21 (IE 2017-21); no test year is read.
**Record:** [`method-joint-fit-reachability.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-joint-fit-reachability.md),
registered in [`method-joint-fit-reachability-prereg.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-joint-fit-reachability-prereg.md).

Of 540 joint national fits, 90 were refused, and 73 of those had observations
below anything the fit could produce. A registered follow-up, added after the
first table was seen and labelled so, found that 61 of the 73 become reachable
or near-miss once off-curve wind counts as zero output, so the "mostly
unreachable" reading is withdrawn. Most refusals behind the Italy and Portugal
suspensions come from the objective dropping off-curve values. Twelve stay
unreachable. No accepted fit was unreachable. Belgian and French classes are
withdrawn. Which off-curve treatment the method should use is open (#112).

### Per-year fleet weighting

**Years:** training 2015-21 (IE 2017-21); test 2023, not read.
**Record:** [`method-country-weighting.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-country-weighting.md),
registered in [`method-country-weighting-prereg.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-country-weighting-prereg.md).

The study asked whether simulating each training year with that year's fleet
lowers corrected national RMSE. **It stopped at its registered row rule
before any fit.** Only Spain's grid stayed within 10% of its observed capacity
denominator, against the three rows required. There is no finding on
weighting, and the test year was not read. Two registered conventions were
set by Claude Code overnight without maintainer review.

### One cluster or many per country

**Years:** the only fold would train 2015-18 and validate 2019.
**Record:** [`method-national-single-cluster-prereg.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-national-single-cluster-prereg.md).

Registered on 2026-09-15, this study would have asked whether a country
fitted against one national series gains from giving each cluster its own
correction. **It never ran:** the ENTSO-E files support only one
forward-chaining fold per country, which leaves its one-standard-error rule
inert, and the registration was not amended to fit the data. Its gates and
predictions are unscored.

## Transfer and interpolation

### Machine-learning transfer

**Years:** training DK 2015-19, DE and UK 2015-18, US 2019-21, BR 2021-23, NZ
2019-23, CL and AR 2021-23; no test year is read (the folds score fitted
factors).
**Record:** [`method-ml-transfer.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-ml-transfer.md).

Correction factors learned in some regions do not predict an unseen region's.
**Round one failed its gate:** leave-one-region-out random forests were
positive in 1 of 5 regions, Brazil only (US -1.059). **Round two failed G1:**
adding New Zealand, Chile and Argentina took the US to -94.6, or -1.12 with
scalars capped at 3.0. A 2026-10-27 notice (#155) found held-out coordinates
in the feature scaling; its check missed a 1e-9 gate by at most 1.2e-5 and
changed no count.

### Why corrections do not transfer

**Years:** one held-out year per fold; Belgium and France trained 2015-21,
tested 2023; the other folds' years are not stated.
**Record:** [`method-why-corrections-do-not-transfer.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-why-corrections-do-not-transfer.md).

Four explanations for the failure to transfer were tested and eliminated:
sample count, target conditioning, the identifiability defect and regime
coverage. Coverage failed in the unexpected direction: across twelve
leave-one-country-out folds, error correlates with median feature distance at
Pearson -0.016. The surviving hypothesis, that corrections are largely a
reanalysis-resolution artefact, has a falsification test (a refit on a finer
wind product) that has not been run. Eight folds hold five points or fewer,
and Belgian and French fold errors are withdrawn.

### Grid validation was in-sample

**Years:** test 2023 for the Netherlands; the others are not stated.
**Record:** [`method-grid-validation-in-sample.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-grid-validation-in-sample.md).

Thesis chapter 4's end-to-end grid validation is in-sample in all fourteen
configurations: each is scored on a surface containing its own control
points. Thirteen are majority self-determined. The one that is not, Denmark
offshore, is the one the chapter reports failing. The Netherlands' headline
includes its own five clusters, so it does not show prediction without local
data. The numbers are not wrong; what they were taken to show is.

### The distance mask

**Years:** country rows training 2015-21 (IE 2017-21), test 2023; DE and UK
2015-18, test 2019; DK 2015-19, test 2020.
**Record:** [`method-distance-mask.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-distance-mask.md).

The 5-degree distance mask on the chapter's kriged surface is a poor
instrument: it deletes 8,269 of 23,989 cells to remove 1,218 of the 2,659
implausible ones, catching 45.8% of the defect and discarding 7,051 good
cells. All 72 off-curve cells lie inside the horizon it keeps. A same-day
notice withdrew the claim that distance selects no risky cells: implausible
shares rise from 4.34% within 1 degree to 14.73% beyond 5.

### Leave-one-country-out interpolation

**Years:** not stated in the record.
**Record:** [`method-loco-interpolation-prereg.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-loco-interpolation-prereg.md).

**Void as a gated study** (2026-09-16). It ran four interpolators on twelve
country folds, but none of its gates could be read: L1 had no
machine-learning comparator on the same folds, N2 and N3 compared against
longitude bands with no per-country figure, and L2 never fixed its distance
statistic, which was refused once the scores existed. The scores stand as a
measurement and are reported in
[Why corrections do not transfer](#why-corrections-do-not-transfer).

### Offshore pool and domain split

**Years:** not stated in either record.
**Records:** [`method-offshore-pool-prereg.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-offshore-pool-prereg.md),
[`method-domain-split-prereg.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-domain-split-prereg.md).

**Both void.** The offshore-pool study, on whether Denmark offshore's
interpolation failure came from how offshore points were defined, was voided
before any gate was read: its premise was false, because the chapter's tables
use no domain split. Its replacement, the domain-split study, failed its
reproduction gate (Denmark offshore missed by 0.00175 against 0.001), which
was not restated after the miss; it is void a second time because it rests on
the retired uniform grids.

### Netherlands holdout

**Years:** test 2023; training years not stated.
**Record:** [`method-grid-nl-holdout-prereg.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-grid-nl-holdout-prereg.md).

**Never run.** Registered on 2026-09-13 to test whether the Netherlands'
gridded correction can be predicted from its neighbours alone. Until it runs,
the chapter's cross-border Netherlands claim is untested, and any figure would
carry the ENTSO-E defect capping Dutch capacity factor at 0.57.

## Physics-informed correction

### Zero-shot transfer

**Years:** test DK 2020, DE and UK 2019, US 2022, BR 2024, each on its
scorecard training years (DK 2015-19, DE and UK 2015-18, US 2019-21, BR
2021-23).
**Record:** [`method-physics-informed.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-physics-informed.md),
gates in [`method-physics-informed-prespecification.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-physics-informed-prespecification.md).

A physics-informed model trained on four of DK, DE, UK, US and BR, applied
zero-shot to the fifth, beat uncorrected ERA5 in all five. The US and UK
margins are about 0.0025 and were not resampled, and the UK's own affine gain
is unresolved. **Rescored like for like
on 2026-10-16, the in-region affine fit is lower than zero-shot in all five,**
and zero-shot wins only 2 of 4 fresh regions. P1 to P3 and Q1 passed; **Q2
failed,** so a later nine-region configuration did not replace the five-region
headline. Most registered predictions failed, including the wake term's
transfer and the joint constraint's confirmation (2 of 4).

### Reproduction

**Years:** as above.
**Record:** [`method-physics-informed-rerun-prereg.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-physics-informed-rerun-prereg.md).

Re-run on 2026-09-16 from a clean tree at a recorded commit, under a
registration fixed beforehand, all five gates passed: every seed's fitted-arm
RMSE matched the published figure exactly, and zero-shot still beat
uncorrected ERA5 in 5 of 5 holdouts. The ablation's large margins partly
reflect speeds driven past every power curve. The random-forest comparison
(P2) and the fresh-region gates were not re-run.

### Leave one country out

**Years:** country folds test 2023 (training 2015-21, IE 2017-21); DK test
2020, DE and UK 2019.
**Record:** [`method-physics-informed-loco.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-physics-informed-loco.md),
registered in [`method-physics-informed-loco-prereg.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-physics-informed-loco-prereg.md).

Trained on eight European countries and applied to the ninth, the model cut
national monthly RMSE in 7 of 9 countries after the 2026-10-25 re-score, to a
median 40% of the uncorrected error. **It failed the registered floor gate
G1:** France became 154% worse and Belgium 17% worse. **A world training pool
failed W1** and worsened four countries. Features beat a constants-only
control, but a country-identifying hub height confounds that. Belgian and
French figures await a rerun.

### Terrain at turbine level

**Years:** training DK 2015-19, DE and UK 2015-18, US 2019-21, AU-NEM
2020-22, BR, AR and CL 2021-23, NZ 2019-23; test DK 2020, DE and UK 2019, US
2022, AU-NEM 2023, BR, AR, CL and NZ 2024.
**Record:** [`method-physics-informed-turbine.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-physics-informed-turbine.md),
registered in [`method-physics-informed-turbine-prereg.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-physics-informed-turbine-prereg.md).

A terrain term learned from eight turbine-level regions did not establish that
terrain transfers. **The flat Danish negative control gained** 0.021 in
spatial RMSE against a 0.0043 margin, so T3 failed and T1's 5 of 7 pass is not
read as terrain evidence. **T2 and T5 failed:** the learned term beat a free
Global Wind Atlas ratio in only 2 of 7 regions. T4 passed, with per-unit error
below uncorrected in 7 of 7. The predictions were informed by a disclosed
training-year analysis, and the run and its deviations were made by Claude
Code overnight without maintainer review.

### Per-unit wake coefficient

**Years:** training DK 2015-19, DE and UK 2015-18, US 2019-21, BR 2021-23;
test 2020, 2019, 2019, 2022 and 2024.
**Record:** [`method-wake-unit-prereg.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/method-wake-unit-prereg.md).

The physics-informed write-up proposed fitting the wake coefficient separately
by observation unit. Its first, in-region stage was registered and run on
2026-09-23, and **failed (K0).** Only the US showed a positive density slope
beyond zero, and the unit contrast's interval spans zero. The one interval
that excluded zero had the opposite sign to the hypothesis. The idea stops at
stage 0; transfer was never tested.

## Tooling

### Review agents

**Years:** none involved.
**Record:** [`agent-critics.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/agent-critics.md),
registered in `docs/findings/agent-critics-prereg.md` (on main); summary in
`scripts/dev/agent_evals/results-2026-10-16.md`.

The three review agents were each run once on ten proposals, five seeded
defects and their five fixed controls, on 2026-10-16. **They failed the
pre-registered bar:** every critic caught all five defects, but the panel
raised a false alarm on 2 of 5 controls, where at most 1 was allowed, and no
review gave PROCEED. One draw per review, cases written by an agent that knew
the answers, a model as judge, and no baseline arm. A second run against a
no-agent baseline is registered in
`docs/findings/agent-critics-baseline-prereg.md` and has no result yet.

### Dataset survey

**Years:** none involved.
**Record:** [`dataset-survey.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/findings/dataset-survey.md).

A July 2026 desk survey ranked market-operator and registry sources for new
regions. Live tests split its first tier: Chile (per-plant hourly) and
Argentina (per-plant monthly) were confirmed; **Turkey, whose endpoint
silently ignored the plant id, and Peru, which publishes company totals, were
demoted,** both where the survey had marked them verified. The remaining
entries are candidates, not facts. Its premise that regime coverage limits
transfer was later ruled out.

## Open follow-ups

- #36: design a correction regime for high-relief clusters, from the terrain
  study.
- #112: whether an off-curve value should count as zero output in the fit and
  the simulation, from the reachability pass.
- #155: the random-forest baselines' feature scaling, checked in the
  machine-learning transfer record; open.
- Reruns on Belgium's and France's rebuilt observations, for every study
  whose Belgian and French figures are withdrawn.
- A refit on a finer wind product, the untested falsification of the
  reanalysis-resolution hypothesis.

## Where two records disagree

The records are frozen at the tag. These differences between them, or within
one, were found when this page was written. Where a record's own notice
settles a figure, the entry above follows the notice; these are the
differences no notice settles:

- **Pre-fix figures left unmarked.** `method-cluster-count.md` still shows NZ
  and AU-NEM corrected figures from before the 2026-09-24 turbine-ordering fix
  (NZ 0.106 against 0.1046 after it), and `region-south-america.md` quotes
  pre-fix AU-NEM figures.
- **Best cluster counts.** For DK, `method-cluster-count.md` gives 100 and
  `method-cluster-count-dk.md` a floor at 200. `region-us-br.md` gives UK 500,
  DE 1000 and US and BR still improving, against 50, 100, a plateau and 60 in
  `method-cluster-count.md`. They are different sweeps, unreconciled.
- **Physics-informed counts.** The prespecification says seven of thirteen
  predictions failed where its table marks eight. The country holdout's
  registration reads G3 as 7 of 9 where the re-scored result says 6 of 9.
- **The machine-learning table.** `method-ml-transfer.md`'s 2026-10-27 notice
  says its round-one table shows the scikit-learn 1.9.1 rerun; the table still
  prints the 1.7.2 values.
- **Three-point folds.** `method-loco-interpolation-prereg.md` says six of
  twelve folds hold three points; its own fold list shows four.
- **Regime coverage.** `method-correction-identifiability.md` and
  `dataset-survey.md` treat it as the standing constraint; the transfer
  record rules it out.
