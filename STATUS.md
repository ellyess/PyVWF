# Status

**Current as of 2026-09-13**, on branch `agent-skills`. One page,
kept short so that it stays true. It records what is running, what is queued,
what is open and what is settled. Detail lives in the documents it names, not
here.

## In flight

Nothing. The extended European download completed on 2026-09-12: 108 months,
17.6 GB, no failures, in `era5/EU_2026-09`. The eleven European rows were
re-run on it on 2026-09-13 (`docs/findings/method-eu-rerun.md`), and Spain,
Italy and Portugal have returned from suspension.

## Queued, in order

1. **The curve library study.** Its pre-registration is an uncommitted draft
   **in `git stash@{0}`** ("curve-library prereg draft, stashed for the
   clean-tree roughness runs"), not in the working tree. Restore it before
   resuming. It still needs revising for the common-row scoring rule, the
   variant set as part of the registered design, the Chile marker and the
   roughness treatment.
2. **A skill for re-running an existing row on changed input**, now that the
   re-run has happened and its procedure is known. See Open.

## Open

One line each. None is started.

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
- The full suite is 52 files and does not finish in one process on this
  machine. Run it one file per process; `CLAUDE.md` has the loop.

## Settled, and not to be reopened

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
