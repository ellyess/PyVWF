# Re-running the eleven European rows: registered plan, conditions and gates

**Date:** 2026-09-12 (drafted; registered on commit, before the ERA5 download
completes and before any run)
**Scope:** how the eleven scorecard rows that read `era5/EU` move to the
per-timestep roughness and the wider ERA5 box, what is measured, and how the
old and new rows coexist in the scorecard. Terms follow `CONTEXT.md`. The
method decision is `method-roughness-treatment.md`; the extent defect is the
suspension notice in `scorecard.md`.

**Everything below is fixed before any re-run exists.** A condition or a gate
added later is labelled post hoc and cannot pass.

## What changes, and what must not

Two things change at once for some rows, and the plan is built around keeping
them apart.

- **The roughness treatment**, from the annual mean stored in `era5/EU` to the
  per-timestep derivation, because the new files carry no roughness field.
- **The loaded extent**, from the old box (12 W to 22 E, 42 to 72 N) to the new
  one (12 W to 31.5 E, 36 to 72 N), which is what returns ES, IT and PT.

Nothing else changes. Each new configuration differs from its scorecard
configuration in exactly three lines under `[era5]`: `path`, `file_tag` and
`roughness`. `file_tag` is not read at run time: it names the download, and is
changed so the configuration does not claim an input it no longer uses. Fleet,
observations, training years, test year, cluster count, time slice, correction
model, seasons and curve library are all held.

| Row | Scorecard config | New config | Extent changes for it |
|---|---|---|---|
| DE | `de_k100.toml` | `de_k100_ptz0.toml` | no |
| DK | `dk_k100.toml` | `dk_k100_ptz0.toml` | no |
| UK | `uk_k50.toml` | `uk_k50_ptz0.toml` | no |
| FR | `fr_country.toml` | `fr_country_ptz0.toml` | no |
| BE | `be_country.toml` | `be_country_ptz0.toml` | no |
| IE | `ie_country.toml` | `ie_country_ptz0.toml` | no |
| SE | `se_country.toml` | `se_country_ptz0.toml` | **yes**, 0.8% of capacity was outside |
| NO | `no_country.toml` | `no_country_ptz0.toml` | **yes**, 4.4% was outside |
| ES | `es_country.toml` | `es_country_ptz0.toml` | **yes**, 50.3% was outside |
| IT | `it_country.toml` | `it_country_ptz0.toml` | **yes**, 94.5% was outside |
| PT | `pt_country.toml` | `pt_country_ptz0.toml` | **yes**, 89.9% was outside |

The suffix is a placeholder for sign-off; `_ptz0` says per-timestep z0 and says
nothing about the box, which also changed. A name carrying both would be
unreadable, so the configuration comment carries the rest.

**DK keeps its bounding box and its opt-in.** Its box stops at 13.5 E while
Bornholm lies near 14.9 E, which the new download does not fix, because the box
is the limit and not the data. So the new DK row sets
`allow_extrapolation = true` exactly as its study runs did, carries § with its
share, and differs from the published row in the treatment alone. Widening the
box stays separate work, after this.

## Deviations

### D1, 2026-09-12: the new configurations set the roughness explicitly

**What changed.** Each new configuration was to change two lines. It changes
three: `roughness = "derived"` as well as `path` and `file_tag`.

**Why.** The first downloaded chunk was loaded through `prep_era5` to check the
file's structure before the other 35 requests ran. It carries the 10 m and
100 m winds and no roughness field, so the derived path runs and the treatment
applied is `derived`, which is what this plan wants. But with `roughness` left
at its default the run *requests* `stored` and gets `derived`, and every
manifest would read `requested: stored, applied: derived`. That record is
accurate and misleading at once, which is the failure this whole sequence has
been correcting. Setting it explicitly makes the request, the result and the
configuration's stated intent agree.

**When, relative to a result.** Before any re-run exists. No row has been
re-run and no comparison has been computed.

### D2, 2026-09-12: G0 was probed early, on one month

**What happened.** While the download was still running, the first month of the
new files (January 2015) was compared against `era5/EU` over Denmark's window,
all 744 hours, read-only. The coordinates matched exactly and `u10`, `v10`,
`u100` and `v100` were **bit-identical**, maximum absolute difference 0.000e+00
rather than merely within float32 tolerance.

**Why it was run early.** So that a reprocessed field would cost one request
rather than thirty-six.

**What it does not do.** It does not stand in for G0. The registered check
covers three sample years and runs before any row is re-run, and it is recorded
here, with its date, so that its result cannot later be read as having been
shaped by a probe that came first. A one-month probe is evidence; the gate is
the gate. The registered branches below are unchanged.

## G0: the inputs must agree where they overlap, before any row is re-run

The new files are a fresh CDS request. If their values differ from `era5/EU`
where the two boxes overlap, every comparison below measures a data change as
well as a treatment change, and the plan does not survive.

**Check, read-only, before any re-run:** for three sample years and a sample of
overlapping cells, compare `u10`, `v10`, `u100` and `v100` between
`era5/EU` and `era5/EU_2026-09`.

Three outcomes, fixed here because the likely cause of any difference is a CDS
reprocessing or an encoding change between the old download and now, which
cannot be undone. What happens in each case must not be decided once the
numbers are visible.

- **Agree.** The overlapping values are equal to within float32 round-trip.
  Comparisons measure the roughness treatment alone, exactly as planned below,
  and nothing else changes.
- **Differ, cause identified and bounded.** The difference is attributed (ERA5T
  against final ERA5 for particular months, a known reanalysis reprocessing, a
  grid registration change) and its size is measured per row. The re-runs
  proceed, every comparison is reported as measuring **the treatment plus that
  difference**, and the bound travels with each row's figure. No row claims a
  treatment effect on its own.
- **Differ, cause not established.** **The treatment claim is withdrawn for
  every row.** The re-run becomes a re-baselining rather than a comparison: the
  new rows stand as current, the published rows as superseded, and the
  scorecard makes no claim about which treatment is better. The method decision
  itself is unaffected, because it rests on Denmark's paired runs against a
  single unchanged input file and on the three reasons in
  `method-roughness-treatment.md`, none of which is this re-run.

**G0 is a stop, not a warning.** A quiet data change under a method change is
exactly the failure this whole sequence has been correcting.

## Conditions, by what changes for the row

| Class | Rows | Conditions run | What a difference measures |
|---|---|---|---|
| **Treatment only** | DE, DK, UK, FR, BE, IE | new files, derived | the roughness treatment, cleanly: no unit of these rows lay outside the old loaded extent, so the wider box adds no cell any of their units interpolates from |
| **Treatment and extent** | SE, NO | old files derived (**B**), and new files derived (**C**) | B minus published isolates the treatment; C minus B isolates the extent |
| **Returning from suspension** | ES, IT, PT | new files, derived | nothing about the treatment. Their published figures are not results, so there is no baseline to difference against |

The extra B runs exist because SE and NO are the only rows where both things
change and the published figures are still results. Six rows need no B run: the
audit of 2026-09-12 found no unit of theirs outside the extent their own
configuration loads (`output/extent_audit_2026-09-12/`), so the added cells
change no interpolation for them. DK's B run already exists: it is R1 of the
roughness study.

## What is measured

Per row, the published row against its new row, and for SE and NO the
decomposition above:

- Both runs scored on their common rows, as the harness does, with the excluded
  share reported. A row that loses units to the change is reported as a loss,
  not dropped quietly.
- The paired bootstrap of procedure B: 1,000 draws, seed 20260911, units for
  the turbine-level rows and months for the country-level ones, 95% percentile
  intervals for the corrected RMSE difference and the correction gain.
- Each run's `extrapolated_capacity_share`, off-curve counts, missing-value
  counts and fit quality, the same fields the scorecard already carries.
- For ES, IT and PT: the training-objective check that exposed them
  (`scripts/analysis/training_objective_check.py`), re-run on the new fits, so
  the fabricated-input mechanism is shown to be gone rather than assumed gone.

## Gates

| Gate | Requirement | Outcome |
|---|---|---|
| **G0** | Overlapping ERA5 values agree, as above. One of the three registered branches applies: agree, differ with the cause bounded, or differ with the cause unestablished, which withdraws the treatment claim for every row. | |
| **G1** | Every re-run row records `extrapolated_capacity_share` of zero, except DK, which keeps its 0.6% because its box is unchanged. A non-zero share anywhere else means the new box still does not cover a fleet, and that row does not return; it is reported with its share and stays marked. | |
| **G2** | Each re-run reproduces the harness's own checks: `git_dirty: false`, no failed offset, the substituted share unchanged from its published row, and the curve library identical by sha256. A change in any of these means the run differs from its published row in more than the input, and the row is not published until that is explained. | |
| **G3** | For the six treatment-only rows, the paired corrected-RMSE interval is reported per row with its width, and the direction recorded. **No threshold is set here and none is needed**: the method decision is already made on other grounds (`method-roughness-treatment.md`), so these intervals describe what the change did. They do not decide anything, and a row that moves the wrong way is published moving the wrong way. | |
| **G4** | For SE and NO, the decomposition is reported: treatment (B minus published) and extent (C minus B), each with its interval. If the extent term is the larger, the scorecard says so for those rows. | |
| **G5** | For ES, IT and PT, the training-objective check shows no cluster sending a majority of its capacity-weighted training days below 0 m/s. If one still does, the row carries a dagger and its diagnosis, and the suspension notice is resolved only as far as the extent defect, not as far as the fit. | |

## Registered predictions

| # | Prediction | Outcome |
|---|---|---|
| P1 | G0 takes its first branch: the overlapping values agree. | |
| P2 | Every row but DK reports a zero extrapolated share. | |
| P3 | The treatment-only rows move by less than 0.002 in corrected RMSE, since DK, the most exposed of them, moved by 0.0002. | |
| P4 | ES and IT move the most of any row, and their corrected RMSE gets worse, because their published figures were flattered by days that dropped out of the score. | |
| P5 | At least one of the three returning rows still carries a dagger, on fit quality rather than on extent. | |

## How the old and new rows coexist

**The new rows replace the old ones in the tables, and the old rows move to a
superseded section rather than being deleted.** A published number is not
withdrawn because a better one exists; it is superseded, and a reader must be
able to find what was published, on what input, and why it changed.

- The main tables carry one row per region, the new one, with `per timestep` in
  the Roughness column.
- A **Superseded rows** section below each table holds the published rows with
  their figures unchanged, their Roughness column reading `annual mean`, their
  input (`era5/EU`), their scorecard configuration, and the date they were
  superseded. Their markers travel with them.
- The three suspended rows leave the suspended table and enter the main table
  if G1 and G5 allow. Their suspension notice is not deleted: it gains a dated
  resolution stating what was wrong, what the new input is, and how the figures
  moved.
- The scorecard's roughness notice gains the outcome, and the row count in the
  treatment table changes from eleven annual mean and six per timestep to
  seventeen per timestep, by two routes.

## Cost

| Step | Cost |
|---|---|
| Download | 36 CDS requests, 13 to 18 GB, queue-dominated, running |
| G0 overlap check | minutes, read-only |
| 11 C runs | about 30 minutes of compute, one region per process |
| 2 B runs (SE, NO) | about 5 minutes |
| Extent audit on the new configurations | about 15 minutes |
| Paired comparisons and the training-objective check | minutes |
| Scorecard, notices and the findings document | the larger share, and not compute |

## What the skills cover, and where they fall short

This is the first use of `new-region` and `findings-doc` on work that matters,
and the gaps are part of the output.

**`findings-doc` covers** the shape of the findings document, the evidence
checks, the scorecard row rules including the § and dagger markers, the
committed scorecard configuration, the curve-library statement and the
CHANGELOG convention. Its fourth stop, added today, covers re-reading a
document when a decision changes, which this work will need repeatedly.

**`findings-doc` does not cover:**

- **Superseding a row.** It knows how to add a row and how to correct one. It
  has no rule for replacing a published row with a better one while keeping the
  old figures visible, which is the central act of this plan.
- **A table-wide column.** Adding the Roughness column changes every row at
  once. The skill's row-level checks say nothing about a change of that shape.
- **A re-run whose input changed.** Its checks assume a result is new, not that
  it replaces one. Nothing in it requires the old and new runs to be compared,
  which is the difference between measuring the change and asserting it.

**`new-region` does not apply here, and says so:** its scope is a turbine-level
region with a new adapter, and it declares country-level regions out of scope.
Eight of these eleven rows are country-level and none of the eleven is new. Its
phases 5 and 6, the runs and the write-up, are the only parts that transfer, and
they transfer as a checklist rather than as a procedure: fix the reported
configuration before running, check `git_dirty`, report the substituted share
and the curve shares, show the full metrics table before any prose.

**The gap this exposes:** there is no skill for re-running an existing row on
changed input, which is now a recurring act in this project (the 2026-08-24
refresh, the curve-library backfill, and this). Each time the structure was
invented fresh.

**Logged as candidate work, to be written after this re-run and not before:**

- a skill for re-running an existing row on changed input, drafted from what
  this plan actually needed rather than from what it predicted it would need,
  with this document as its specification;
- the three `findings-doc` gaps above: a rule for superseding a row, a rule for
  a table-wide column change, and a check that the old and new runs were
  compared rather than assumed equivalent.

Writing either ahead of the work would record the guess rather than the
procedure.

## Committed in advance

- Every row is reported whichever way it moves, and none is dropped.
- A row that fails G1, G2 or G5 is reported failing, and does not enter the
  main table.
- The old figures stay visible and dated.
- Bornholm, and the manifest's inability to tell a stored annual mean from a
  stored daily one, stay out of scope here and stay logged.
- A deviation from this design is recorded, with its date and whether it came
  before or after a result was seen.
