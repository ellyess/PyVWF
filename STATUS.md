# Status

**Current as of 2026-09-16**, on branch `manuscript-chapters-45`, at the close
of a session. One page, kept short so that it stays true. It records what is
running, what is queued, what is open and what is settled. Detail lives in the
documents it names, not here.

## In flight

- **Study B of the physics-informed leave-one-country-out**, the world pool,
  launched 2026-09-17 (`method-physics-informed-loco-prereg.md`). Study A is
  done: transfer corrects national capacity factor in 8 of 9 gated countries
  and fails its floor gate on France (`method-physics-informed-loco.md`).

## Where the manuscript stands

Thesis chapters 4 (gridded interpolation) and 5 (machine-learning transfer) are
being merged into one manuscript with fresh results. The chapters are read-only
and outside this repository.

**The spine of the paper is
`docs/findings/method-why-corrections-do-not-transfer.md`.** A bias correction
fitted in some countries does not predict the correction in an unseen one, by
interpolation or by a learned model, and four candidate explanations have been
tested and eliminated: sample count, target conditioning, the identifiability
defect, and regime coverage, the last ruled out in the wrong direction. **One
explanation survives**: the correction is substantially a
reanalysis-resolution artefact rather than a transferable physical property,
supported by over 80% of its variance being within-region and by the four
eliminations.

- **Its falsification test** is named: refit on a finer wind product, CERRA at
  5.5 km or the New European Wind Atlas at 3 km, and see whether the extreme
  scalars disappear at source.
- **Its weakest link** is named: the coverage elimination used terrain features
  only, so a regime differing climatically but not topographically is invisible
  to it, and the Netherlands, best covered and among the worst predicted, is
  where that would bite.

The workstream's founding assumption is D0 in
`docs/design/manuscript-chapters-45.md`: country-level results are computed on
the maintained fleet-weighted grids, and the chapter's uniform-grid figures are
a historical baseline, not a target.

### Unrun and unblocked

- **The Netherlands holdout, on both grids**
  (`method-grid-nl-holdout-prereg.md`). Both modules it was waiting on are
  ported. H-G3 gates the uniform arm against the chapter's published figures;
  H-G4 reports the maintained arm beside it as a measured difference.

### Waiting on port phase 3, the machine-learning module

- **The ML module itself**, `extensions/ml/correction.py`, unported.
- **Leave-one-country-out's L1**, which needs a machine-learning score on the
  twelve country folds.
- **The candidate-pool study**, if still wanted: which pool gives the best
  leave-one-country-out prediction, varied on count and on label quality.
  Deliberately not registered, since the transfer synthesis makes the pool
  question a weaker one than it was.

### Void or blocked, one line each

- **Domain split**: void, S-G1 failed on the roughness treatment and the study
  is now on a construction the workstream replaced
  (`method-domain-split-prereg.md`).
- **Offshore pool**: void, replaced by the domain split
  (`method-offshore-pool-prereg.md`).
- **Leave-one-country-out**: void, three gates written against comparators
  that did not exist and a fourth ambiguous (`method-loco-interpolation-prereg.md`).
- **National single cluster**: blocked on observation files, one
  forward-chaining fold per country (`method-national-single-cluster-prereg.md`).

### Two decisions not yet made

1. **Whether to pursue a finer wind product.** It is the only test of the
   surviving explanation and the largest single piece of work the repository
   has left. Named, not costed in hours.
2. **Whether the country tier's pool question is worth reopening.** The
   selection study's counts are not adopted for the pool, and the transfer
   synthesis says count is not the binding constraint, so it may not be.

### Deferred to the next session

- **Three skills from `K-Dense-AI/scientific-agent-skills`**:
  `statistical-power`, `uncertainty-units`, `experimental-design`. Read each
  SKILL.md, report what it would change, and write nothing into the tree first.
  **Licence constraint:** `.claude/skills/` is checked in and the repository is
  public, so installing means vendoring third-party files under their licence;
  if a licence is awkward, take the idea and write our own. Also read
  `Galaxy-Dawn/claude-scholar`'s `research-contract.md` for ideas only, without
  installing it. `pubfig` and `pubtab` are logged as later candidates for the
  manuscript figures.

## Reading order for the findings

Start with **`method-why-corrections-do-not-transfer.md`**, which is the
argument and cites everything else. Then
**`method-correction-identifiability.md`** for why the fitted scalar and offset
are not separately identified, the pivot probe, and the invariance that makes
reparameterisation unable to change an interpolator's prediction. Then
**`method-cluster-selection.md`** for the one registered study that ran to
completion this week, and **`method-distance-mask.md`** for why the chapter's
5-degree mask was a poor instrument. For the country-level data that sits under
all of it, read **`method-country-level.md`**, including its correction notice
on Sweden's derived capacity register. The registrations named in the void and
blocked lines above explain why each stopped, and
**`docs/design/manuscript-chapters-45.md`** records the decisions, D0 first.

## Queued, in order

1. **Decide the two open decisions above.** Nothing further on the manuscript is
   well defined until the finer-wind question is settled.
2. **Port phase 3**, the machine-learning module, which unblocks L1 and any
   candidate-pool study.
3. **A skill for re-running an existing row on changed input.** See Open.

## Open

One line each. None is started.

- **The offset search cannot fail loudly, and throttles silently below about
  1 m/s.** `vwf.correction._find_offset_iterative` tests convergence on the
  step size, `abs(step) <= tolerance`, not on the residual, and the step
  shrinks whether or not the error did, so a search that never reaches the
  root still returns a number and reports success. It also clamps each
  proposed step to the previous magnitude, which binds only when the initial
  step is below the natural step size: a proposed step is the cube root of a
  capacity-factor error and so never exceeds about 0.7 m/s, so at the shipped
  `initial_step = 10.0` the clamp never binds, and at 0.25 it binds
  immediately and the offsets come back wrong by up to 7.2 m/s with zero
  reported non-convergences.
  **No result in this repository is affected.** The only other value the
  constant has ever had is 3.0, the default until 2026-02-14 alongside
  `max_iter = 30`, and probing three dense rows at 10, 4, 3 and 1 with both
  iteration caps gives pivots identical to three decimals
  (`method-correction-identifiability.md`).
  **Fixed 2026-09-16 in `724ab1b`**, as proposed: both
  `_find_offset_iterative` and the `_find_offset_scipy` fallback now test the
  residual rather than the step and return NaN instead of a non-root, with
  `MAX_OFFSET_RESIDUAL` at 1e-4 in capacity factor. Four tests on an analytic
  curve, and **the golden regression test did not move**, so no fit in this
  repository had been returning a non-root. Kept here as the record of what it
  was; nothing remains to do.

- **The national single cluster study is blocked on data, and the order that
  unblocks it is fixed.** `method-national-single-cluster-prereg.md` needs
  forward-chaining folds, each wanting a training window ending in Y-1 and a
  test file for Y, and exactly one such pair exists per country because only
  three training windows were ever generated. Generating more means
  regenerating the same series whose registers are under correction, so the
  sequence is: settle the capacity registers, build the `capacity_source`
  provenance field, generate the windows, then run as registered. The
  turbine-level rows were unaffected because `european-turbine` resolves
  observations by region and year while the ENTSO-E country path resolves by a
  pre-generated window file.

- **The cluster grid excluded two of the counts it was asked to beat, and
  widening it now would be selection on the outcome.**
  `method-cluster-selection.md` reports C-G3 failing in three of five rows, and
  in two of those the chapter's count, `k=300` for UK onshore and `k=884` for
  DK onshore, is not in the registered grid of 1, 10, 25, 50, 100, 200, 500,
  1000. Denmark onshore's `k=884` beats every count the grid holds. A future
  study may register a grid that contains the comparators, before seeing which
  ones matter; this one records the limitation instead.

- **Denmark onshore's flat cluster-count curve may not be real, and it was the
  motivation for an entire protocol.** `method-cluster-count-dk.md` reports the
  curve moving 0.0857 to 0.0851 across a sixteenfold increase in `k`, a spread
  of 0.7%, and that flatness is why `method-cluster-selection-prereg.md` adopted
  nested selection with a one-standard-error rule rather than minimum-picking.
  The same configuration swept on 2026-09-15 spreads **24.9%** across `k=1` to
  `k=1000`. The two differ in fleet, in grid, and in whether the score comes
  from forward-chained training folds or from a single scored test year, so
  nothing is yet shown to be wrong. But if the published curve is flat only
  because of how it was scored, the protocol was built on an artefact. Not
  reconciled inside the selection study, deliberately.

- **Sweden needs a real installed-capacity register, because its current one
  is derived from its own generation.** The four bidding zones each hold a
  capacity frozen across 2015 to 2019, each zonal series peaks at exactly
  0.900, and the four sum to the national 8354 MW exactly. 0.900 is the
  signature of the ENTSO-E fetcher's fallback for a country whose
  installed-capacity endpoint returns nothing,
  `estimated_cap = gen.max() / 0.9`. GWPT gives 4226 rising to 6270 over those
  years, so the two disagree by a factor of two and neither is trustworthy;
  the current file's plausible-looking capacity factors were constructed to
  peak near 0.9. Sweden and SE-BZ are excluded from
  `method-national-single-cluster-prereg.md` until Sweden has a register, and
  the exclusion is declared before any result. Portugal sits beside it for a
  different reason: same frozen register, and GWPT is the repair
  (`scripts/region_tools/repair_country_capacity.py PT`). Not searched for
  yet, deliberately.
- **Per-zone observation checks, and what the derived-capacity audit could not
  see.** One investigation, two halves. The exact-0.900 signature of
  `estimated_cap = gen.max() / 0.9` was searched across all 88 country-level
  observation files on 2026-09-15 and finds exactly sixteen, the Swedish zonal
  files `se_1` to `se_4` in every split. **The signature is not sufficient**:
  Sweden's national aggregated register is the sum of those four and is
  therefore derived too, while its own peak is 0.876, so a series built by
  aggregating derived ones passes the test. Norway aggregates the same way and
  its zones are not derived, so aggregation alone is not the defect. A
  provenance field on the capacity column would settle it; the signature
  cannot. **That field is the per-value provenance registry already open
  below**, which is blocked on the licence query, a constraint the costing of a
  `capacity_source` column on 2026-09-16 did not account for. The other half is that `check_country_cf` takes the median capacity
  across zones when handed one stacked frame, which is how SE-BZ came back as
  2158 MW unchanged for three years when the truth is four zones frozen for
  five: right finding, wrong figure.
- **Sweden has two national series that disagree.**
  `se_train_2015_2018.csv` carries 6247 MW and
  `se_train_2015_2018_aggregated.csv` carries 7034 MW for the same window. The
  aggregated one is the zone sum and is derived; the other's provenance has not
  been established.

- **Run chapter 5's models on all twelve country folds when port phase 3
  lands**, reporting MAE beside R-squared. The country-holdout study's gate L1
  is indeterminate because chapter 5 ran leave-one-country-out for Germany and
  the United Kingdom only, and reported R-squared. On those two folds the
  machine learning scores higher than every interpolator, +0.096 against -0.146
  and +0.091 against -0.046, which is the only direct comparison in the study
  and points against the chapter's conclusion. Settling it alone would not
  justify porting `extensions/ml` and rebuilding 37 features over 1,729
  centroids; phase 3 ports that module anyway, so this is a follow-on rather
  than a decision. Registered design in
  `docs/findings/method-loco-interpolation-prereg.md`.
- **Four PyPSA-Eur scenarios must not be run before the shipped kriging grid is
  settled.** `base-s100000-biaskriging` and its siblings in
  `pypsa-eur-wind/config/scenarios-validation.yaml` select a grid file that is
  neither the standard kriging the chapter adopts nor the split pipeline its
  tables evaluate: it is the hybrid configuration the chapter rejected, built
  from all 1,729 control points with no onshore and offshore split. No result
  has been produced from it, the thesis PyPSA-Eur results all used the IDW file,
  and that file is correctly described. The config is public on
  `github.com/ellyess/pypsa-eur-wind`; the grid file itself is untracked, so a
  clone fails on a missing path rather than using a mislabelled one. Details in
  `docs/design/manuscript-chapters-45.md`, section T3. Nothing in that
  repository has been edited from here.
- **Six scorecard rows predate common-row scoring and would move if re-run:**
  US, BR, AU-NEM, NZ, CL and AR, the six still standing on the 2026-08-24
  refresh. The eleven European rows were re-run in September and are under the
  convention. The spread in `n_samples` across a row's variants bounds what it
  would cost: BR, AU-NEM and NZ have none, so they would not move on this
  account; AR has 2 rows of 682; the US has 10 of 6,078 and was measured
  directly at 0.00003 in corrected RMSE and 0.00004 uncorrected; **CL has 46 of
  677, 6.8% of its rows, and is the one to look at.** Equal counts do not prove
  identical row sets, so this bounds rather than settles it. Scope of any
  correction is a scorecard decision, not a study one.

- **Bornholm.** DK's box stops at 13.5 E and Bornholm lies near 14.9 E, so 47
  units, 0.6% of capacity, sit outside the extent the row loads. The data is in
  the files, old and new: this is a box to widen, not a download. DK is now the
  only scorecard row carrying the section marker. It makes a new DK row.
- **Whether `fit_quality` should bound the share of training steps a fitted
  pair sends below zero.** Italy is the motivating case, and a better one than
  anything the archive offered: an ordinary fit, no implausible scalar, no
  failed offset, no dagger, on real winds, which still drops 1.0% of its
  capacity-weighted steps on calm days because every one of its twelve pairs
  has a negative offset. The loss also scales with configuration complexity, 1
  unit-month at `fixed_1` to 69 at `season_3`, so any bound has to say what it
  is a share of. Germany and the United Kingdom show the same mechanism.
  Whoever picks this up should start from Italy's `season_3` factors in
  `output/eu_rerun_2026-09-12/new/IT/train-new/`, not from a pathological fit.
- **`load_turbine_metadata` truncates the Danish manufacturer to its first
  word** ("NEG Micon" to "NEG"), and drops the `model` column entirely. The
  truncation mis-classified 19.9% of DK capacity in the curve-match audit,
  corrected in the scorecard on 2026-09-13 by reading the full string.
  Changing the loader alters an input to a published row, so it is separate
  work: it needs the DK row re-run, or a statement that the row's own figures
  do not depend on the manufacturer, which they may not.
- **`run_transfer` has no end-to-end test,** and returns frames without a run
  directory, so nothing records its provenance.
- **`scripts/era5/combine.py` back-fills undefined roughness within one month
  file,** while `prep_era5` back-fills across the whole record. Scope a fix to
  where undefined-shear cells sit near units.
- **The US and Brazil cannot answer the roughness question.** Their combined
  files carry no 10 m winds, so `roughness = "derived"` raises, and their
  stored daily roughness is recorded as `stored`, which a manifest cannot tell
  from an annual mean.
- **`docs/findings/TURBINE_GRID_EVALUATION_ANALYSIS.md` is git-ignored** yet
  reports figures from the ES row as published under extrapolated winds, now
  superseded. Track it with a correction, or retire it.
- **Reproducing the published Denmark figures** needs onshore-only mode, a
  sweep to 3,300 clusters and three metric scales. Candidate work, out of scope
  where it has come up.
- **A skill for re-running an existing row on changed input.** Four times now
  the structure has been invented fresh, the last of them with a written plan.
  Write it from what that plan actually needed, with it as the specification.
  The three `findings-doc` gaps go with it: superseding a row, a table-wide column
  change, and a check that old and new were compared rather than assumed
  equivalent.
- **The per-value provenance registry for capacity-factor denominators** is
  blocked on the licence query below. It is the same thing as the
  `capacity_source` column costed on 2026-09-16 against the derived-capacity
  audit above: per-row values of `entsoe`, `gwpt`, `derived`, `aggregated`,
  `curated` or `unknown`, roughly half backfillable from evidence and half
  honestly unknown.
- **The licence query itself:** whether values derived from the licensed
  turbine database may be named in the public repository. It reaches the
  Argentine denominators, which are estimates corrected by turbine research
  (`docs/runbooks/ar.md`), and the curated coordinate overrides.

## How this repository is checked

- **CI does not run on a branch push.** `.github/workflows/ci.yml` triggers on
  `pull_request`, on a push to `main`, and on `workflow_dispatch`. So pushing a
  feature branch is backup only, and the matrix has to be asked for by hand:

  ```bash
  gh workflow run CI --ref <branch>
  gh run list --branch <branch> --limit 1
  ```

  This is worth knowing before assuming a green branch means anything. The
  matrix has caught, in one afternoon, a segfault from writing netCDF in two
  threads at once, a test broken by a rename, and a test that was flaky by
  construction on one Python. None of the three failed locally.
- **Local green is not CI green,** and `CLAUDE.md` lists why: pandas-stubs
  absent here and present there, acquisition libraries present here and absent
  there, and a pandas major version split across the matrix.
- The full suite does not finish in one process on this machine. Run it one
  file per process; `CLAUDE.md` has the loop. Count the files before asserting
  how many there are.

## Settled, and not to be reopened

- **The curve library study is closed**, all four conditions run and reported
  (`docs/findings/method-curve-library.md`). Q1: giving a country grid its own
  curves lowers simulated output in all eight rows by 0.082 to 0.220 in mean
  bias, and whether that helps depends on where the row started; G1 fails at
  three of seven. Q2: curve assignment at turbine level changes corrected RMSE
  by at most 0.0012, G2a and G2b pass, G3 fails in all four rows, and ERA5 bias
  is 8 to 43 times the larger error source. **The strongest result is P6's
  refutation:** the wind scalar had been absorbing a systematic power-curve
  mismatch, so a scalar inside the plausible band was evidence of absorbed
  curve error rather than of a good fit. Seven predictions, one held.
- **Two defects in the study's own tooling were found, fixed and measured**,
  and both are recorded rather than tidied away. An override table built from
  the training fleet alone left the test fleet partly unreassigned; an override
  applied to the frame `train_set` returns arrived after the simulation the
  wind scalar is fitted from, so every affected fit was a hybrid of two
  conditions. The second was worth up to 0.018 in corrected RMSE at country
  level and at most 0.0006 at turbine level. C2's corrected figures carried a
  dated correction notice before the corrected values existed.

- **Every scorecard row now applies the per-timestep roughness**, and no row
  applies an annual mean. The three suspended rows returned on 2026-09-13,
  none daggered.
- **The per-timestep roughness derivation is the method**, adopted 2026-09-12.
  It was adopted on method fidelity and comparability, **not** on accuracy: the
  measured effect on Denmark is 0.0002 in corrected RMSE, resolved by the
  pre-registered gate and far too small to carry the change on its own. The
  reasons are that per timestep is what the published method describes, that
  most rows cannot show any difference at all, and that one treatment removes a
  split confounding every comparison between regions
  (`docs/findings/method-roughness-treatment.md`).
- **Releases are the maintainer's.** Merging to `main`, pushing, tagging,
  publishing and anything minting a DOI. Agents hand over the commands
  (`AGENTS.md`).
- **The scorecard markers** are fixed and mechanical: † a degenerate fit, ‡ a
  gain not distinguishable from zero under resampling, § a non-zero
  extrapolated share with its share stated. § holds even when a region opted in
  to extrapolation: opting in lets a run finish and does not make the number
  sound (`docs/README.md`).
- **A published row is superseded, never deleted,** and a wrong published claim
  gets a dated correction notice rather than a silent edit.
