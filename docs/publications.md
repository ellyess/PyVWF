# Publications

Results published from PyVWF, with the commit that produced each. They are
legacy. The repository does not reproduce them: its purpose is the most
accurate results it can produce now, and its current outputs supersede them
(see the policy on real-data pins in
[CONTRIBUTING.md](https://github.com/ellyess/PyVWF/blob/main/CONTRIBUTING.md#running-the-tests-and-linter)).
The commit is recorded so that the code behind a published number can still be
checked out and read.

Each commit carries an annotated tag, named below, once the maintainer has
created it. Until then, check out the commit hash.

## The method paper (Energy, 2024)

> Benmoufok, E. F., Warder, S. C., Zhu, E., Bhaskaran, B., Staffell, I., and
> Piggott, M. D. (2024). *Improving wind power modelling through granular
> spatial and temporal bias correction of reanalysis data.* Energy.

| | |
|---|---|
| DOI | [10.1016/j.energy.2024.133759](https://doi.org/10.1016/j.energy.2024.133759) |
| Commit | `75f2678`, 2024-04-08, "cleaned up for publishing" |
| Tag | `publication/energy-2024` |

## The PhD thesis (2026)

> Benmoufok, E. F. (2026). *Data Science-Enhanced Wind Power Modelling: From
> Reanalysis Correction to Energy System Representation.* PhD thesis, Imperial
> College London. Submitted 2026-04-05.

| | |
|---|---|
| DOI | to be added when the thesis is deposited |
| Commit | `d989d52`, 2026-04-03, "Enhance plotting utilities; add research notebook": the last commit on the `development` branch before submission |
| Tag | `publication/thesis-2026` |

The code of chapters 4 and 5 is on the `development` branch, not on `main`:
it was moved there on 2026-07-06 (`e8208e1`).

## Archived submissions

Work submitted from PyVWF that was not published. It is listed here so that a
reader who finds the file in the tree knows its status, and does not read it
as a current description of the software.

### The JOSS paper (2026)

> Benmoufok, E. F., Warder, S. C., and Piggott, M. D. *PyVWF: An open Python
> framework for bias-corrected wind power simulation from reanalysis data.*
> Submitted to the Journal of Open Source Software.

| | |
|---|---|
| File | [`paper/paper.md`](https://github.com/ellyess/PyVWF/blob/main/paper/paper.md), with `paper/paper.bib` |
| Outcome | Not accepted. The review is closed and the paper is not under submission. |
| Commit | `db882d2`, 2026-07-17, "Finalize v0.3.0 release metadata: changelog, citation, and paper dates": the last change to the paper's text before it was marked archived |
| Tag | `archive/joss-ready` |

The paper carries a dated status line of its own, added in `44e7cf9` on
2026-09-11. It describes PyVWF as it stood at submission and is not updated.
It states no results, so nothing in it is affected by a later correction to a
number. For the current state of the software, read the
[project README](https://github.com/ellyess/PyVWF#readme).

## Removed code paths

Code removed from the tree that produced results above, with the last commit
that has it, so a legacy run can still be checked out and re-run.

| Code | Last commit with it | Removed |
|---|---|---|
| The legacy `PyVWF` class (`src/vwf/vwf.py`), `pyvwf-train`, the batch scripts `train_all_bias_corrections.py` and `evaluate_all_pyvwf_runs.py`, the year-specific grid loader, the config writer, and the harness-versus-legacy runners `regression_run_legacy.py` and `regression_run_harness.py` | `d039608`, 2026-09-24, merged through pull request #49 | 2026-09-24, so that the harness is the one path ([training guide](guides/training.md#the-legacy-path-removed)) |

The thesis-era runs under `output/runs/` and the harness-regression check in
`docs/findings/method-harness-regression.md` were made with that code.

## How the commits were chosen

Neither publication's outputs carry a run manifest, so no output records the
commit that produced it. Each commit above is the maintainer's attribution:
for the paper, the commit that prepared the repository for publication; for
the thesis, the last commit on `development` before the submission date; and
for the archived JOSS submission, the last commit that changed the paper's
text before the archive note was added.
