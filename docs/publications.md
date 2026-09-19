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

## How the commits were chosen

Neither publication's outputs carry a run manifest, so no output records the
commit that produced it. Each commit above is the maintainer's attribution:
for the paper, the commit that prepared the repository for publication; for
the thesis, the last commit on `development` before the submission date.
