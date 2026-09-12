# Status

**Current as of 2026-09-12**, on branch `agent-skills` at `0a989e2`. One page,
kept short so that it stays true. It records what is running, what is queued,
what is open and what is settled. Detail lives in the documents it names, not
here.

## In flight

**The extended European ERA5 download.** `era5/EU_2026-09`: hourly 10 m and
100 m winds, no roughness field, box 12 W to 31.5 E and 36 to 72 N, 2015 to
2023. 36 CDS requests, about 17 GB, four in flight at a time
(`scripts/fetch/era5.py --workers 4`). Resumable: a month is done when its file
exists, so an interrupted run picks up where it stopped.

It blocks everything below it. The eleven European scorecard rows cannot move
to the per-timestep roughness until it lands, and the three suspended rows
cannot return until their fleets are inside the data.

## Queued, in order

1. **The full G0 check.** Three sample years of overlapping cells, old files
   against new. It decides whether the re-run measures the treatment alone,
   the treatment plus a bounded data difference, or nothing about the
   treatment at all. The three branches are fixed in
   `docs/findings/method-eu-rerun-prereg.md`. A one-month probe was
   bit-identical, which is evidence and not the gate.
2. **The eleven-row re-run**, to the plan in that same document: eleven new
   configurations, eleven new rows, the three suspended rows returning, old
   rows superseded rather than deleted, and a paired comparison per row.
3. **The curve library study.** Its pre-registration is an uncommitted draft
   **in `git stash@{0}`** ("curve-library prereg draft, stashed for the
   clean-tree roughness runs"), not in the working tree. Restore it before
   resuming. It still needs revising for the common-row scoring rule, the
   variant set as part of the registered design, the Chile marker and the
   roughness treatment.

## Open

One line each. None is started.

- **Bornholm.** DK's box stops at 13.5 E and Bornholm lies near 14.9 E, so 47
  units, 0.6% of capacity, sit outside the extent the row loads. The data is in
  the files: this is a box to widen, not a download. It makes a new DK row.
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
  reports figures from the suspended ES row. Track it with a correction, or
  retire it.
- **Reproducing the published Denmark figures** needs onshore-only mode, a
  sweep to 3,300 clusters and three metric scales. Candidate work, out of scope
  where it has come up.
- **A skill for re-running an existing row on changed input.** Three times now
  the structure has been invented fresh. To be written after the re-run, from
  what it actually needed, with the re-run plan as its specification. The three
  `findings-doc` gaps go with it: superseding a row, a table-wide column
  change, and a check that old and new were compared rather than assumed
  equivalent.
- **The per-value provenance registry for capacity-factor denominators** is
  blocked on the licence query below.
- **The licence query itself:** whether values derived from the licensed
  turbine database may be named in the public repository. It reaches the
  Argentine denominators, which are estimates corrected by turbine research
  (`docs/runbooks/ar.md`), and the curated coordinate overrides.

## Settled, and not to be reopened

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
