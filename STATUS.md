# Status

**Current as of 2026-09-13**, on branch `agent-skills`. One page,
kept short so that it stays true. It records what is running, what is queued,
what is open and what is settled. Detail lives in the documents it names, not
here.

## In flight

Nothing. The curve library study closed on 2026-09-13, after fifteen conditions
were re-run to correct a defect in its own tooling; see Settled. Before it, the
extended European download completed on 2026-09-12 (108 months, 17.6 GB, no
failures, in `era5/EU_2026-09`) and the eleven European rows were re-run on it
(`docs/findings/method-eu-rerun.md`), returning Spain, Italy and Portugal from
suspension.

## Queued, in order

1. **Merging thesis chapters 4 and 5 into one manuscript**, co-authored, with
   fresh results from the repository as it stands. The chapters are read-only
   and outside this repository. Phase 0 is an inventory of what they did,
   phase 1 a survey of whether the code still runs, phase 2 a statement of what
   a re-run would change. No runs before phase 2 is read.
2. **A skill for re-running an existing row on changed input**, now that the
   re-run has happened twice and its procedure is known. See Open.

## Open

One line each. None is started.

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
  blocked on the licence query below.
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
