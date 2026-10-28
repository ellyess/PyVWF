# PyVWF documentation

New to the project? Read the [project README](../README.md) first, for what
PyVWF is, what it is for, and the quickstart. The documentation site is built
from this folder and hosted at
[pyvwf.readthedocs.io](https://pyvwf.readthedocs.io/); [`index.md`](index.md)
is its front page and the canonical contents list.

## How this folder is organised

Filenames are lower-case with hyphens, and the folder supplies the category,
so a name never repeats it (`guides/training.md`, not
`guides/TRAINING_GUIDE.md`).

| Folder or file | Holds | Kind | Naming |
|---|---|---|---|
| [`guides/`](guides) | How to use PyVWF: installing it, data, training, outputs, and extending it with a region or a study (`adding-a-*.md`). Published. | procedural | `<topic>.md` |
| [`runbooks/`](runbooks) | Per-region acquisition and processing steps. Published. | procedural | `<iso-code>.md`; `au_nem.md` and `entsoe.md` for the NEM and the country-level regions |
| [`design/`](design) | Why the code is shaped as it is. Published. | argumentative | `<component>.md` |
| [`findings/`](findings) | The records of active studies only. A finished study's records leave at the next archive tag. **Not** published. | argumentative | `<type>-<subject>.md` |
| [`results.md`](results.md), [`scorecard.md`](scorecard.md), [`archive.md`](archive.md) | One entry per finished result, the per-region scorecard, and where the full records are. Published. | reference | fixed names |
| [`api.md`](api.md), [`CONTEXT.md`](CONTEXT.md), [`publications.md`](publications.md) | The API reference, the controlled vocabulary, and the commit behind each published result. Published. | reference | fixed names |

The folders follow the Diataxis split. **Learning** is the project README's
quickstart and `examples/run_minimal.py`. **Doing** is `guides/` and
`runbooks/`. **Looking something up** is `api.md`, `CONTEXT.md`, and two
guides that are reference in kind, `data-sources.md` and
`output-structure.md`. **Understanding** is `design/`. `findings/` sits
outside the split: it records what was found, dated, and is never revised into
guidance. A document serves one purpose; when a page starts to serve two, the
second part moves to the folder that fits it, and the first page links to it.

Private working notes, which record decisions not yet made or work outside
this repository, are the maintainer's: untracked, and excluded locally. A
document here cites one by a permalink to its last tracked version, never by a
path the repository no longer has.

## One home per fact

Each fact has one home, and every other document links to it rather than
restating it. A restated fact drifts, and nothing says which copy is right.

| Fact | Home |
|---|---|
| Installing PyVWF, what each extra adds, and Docker | `guides/installation.md` |
| What PyVWF does not do, and what its numbers do not support | `design/limitations.md` |
| Running the tests and the other CI checks | `CONTRIBUTING.md` |
| Which module may import which | `.importlinter`, at the repository root |
| Choosing the input root (`PYVWF_INPUT`), and running a region | `guides/training.md` |
| The removed legacy path | the last section of `guides/training.md` |
| What a run directory contains | `guides/output-structure.md` |
| Adding a region, an adapter or a study | `guides/adding-a-region.md`, `adding-a-study.md` |
| Each data source, its licence and its processing | `guides/data-sources.md` |
| Fetching and combining ERA5 | `guides/data-sources.md`, section 3 |
| The approved term for a concept | `CONTEXT.md` |
| The rules for agents | `AGENTS.md` |
| How agents are used in development, and the counts that describe it | `design/ai-native-development.md` |
| The commit behind a published result | `publications.md` |
| Every finished study's result, with its years and its record | `results.md` |
| Each region's uncorrected and corrected error, and its markers | `scorecard.md`; the project README's table is a copy, checked by `tests/test_readme_results.py` |

## Findings documents

A findings document answers one question, and its name carries its type:
`region-*.md` for one region's validation, `method-*.md` for one method
question (a `-<code>` suffix for a single-region deep dive, `-prereg` for the
pre-registered gates), and `agent-*.md` for the development tooling, whose
results have no training or test year and say so. It is revised in place as
the answer changes, and git holds its history. Every result states its
training years and its single test year. A correction to a published claim is
a dated correction notice at the top, never a quiet edit. When the study
finishes, its result gets an entry in `results.md`, and its records leave at
the next archive tag (`archive.md`).

## Scorecard markers

A scorecard row carries a marker when its number does not mean what it would
mean alone. The markers are set by rule, from the run's records:

| Marker | Set when | Read from |
|---|---|---|
| † | The fit is degenerate: a scalar outside 0.2 to 3.0, or an offset that did not converge. | `fit_quality`, in the row's `metrics.csv` |
| ‡ | The gain cannot be distinguished from zero when the test year's units are resampled. | the correction notice that set it |
| § | `extrapolated_capacity_share` is above zero: part of the fleet lies outside the loaded ERA5 extent, and its winds were extrapolated. The share is stated beside the marker. | `extrapolated_capacity_share`, in the row's `metrics.csv` |
| ¶ | `observations_clipped_share` is above zero: part of the observed series sits on the fetcher's 1.5 capacity-factor ceiling, so those values were **discarded rather than wrong** and the metric is computed over fewer observations than the row claims. The share is stated beside the marker. | `observations_clipped_share`, in the row's `metrics.csv` |

A row with a non-zero extrapolated share carries § whether or not its region
opted in to extrapolation: opting in lets the run finish, and does not make the
number sound. For a row run before the field existed, the share comes from the
region's extent check, and the row cites it. The worst shares of training steps
a fitted pair sends below 0 m/s or above the curve (`max_below_zero_share`,
`max_above_curve_share`, `max_period_dropped_share`) are reported beside the
dagger and do not set it: a bound on them needs its own pre-registered
calibration, as the scalar bounds had.

## Two kinds of document

Terms in every document follow [`CONTEXT.md`](CONTEXT.md).

**Procedural documents** (`guides/`, `runbooks/`) are for readers who do not
know the project. They follow three rules, adapted from ASD-STE100 Simplified
Technical English:

1. One instruction per sentence, and one imperative per runbook step. A
   condition goes in the sentence before the instruction, or in a nested
   bullet, never in a subordinate clause.
2. At most 20 words in a procedural sentence, and 25 in a descriptive one.
3. One approved term per concept, each in one sense only, as listed in
   `CONTEXT.md`.

The rules simplify sentence structure only. They never remove a proper noun, a
document reference or an identifier; a kept reference may push a sentence past
the limit, and is then marked with an HTML comment naming what it keeps. They
do not simplify the field's standard technical terms: rule 3 governs the
project's own vocabulary. They apply to new procedural documents, and to
existing ones when next revised.

**Argumentative documents** (`findings/`, `design/`) make a case, and the
procedural rules do not apply. `findings/` is for readers who know the project and its history. `design/` is published,
for readers who know the domain but not the project: no references to
conversations, no "as we decided", no assumed context. Both define each term
once by citing `CONTEXT.md` and use it consistently, state the claim before the caveat, and keep
reasoning about why the code is shaped as it is in `design/`.
