# PyVWF documentation

New to the project? Read the [project README](../README.md) first, for what
PyVWF is, how to install it, and a Denmark quickstart. The full documentation
site is built from this folder and hosted at
[pyvwf.readthedocs.io](https://pyvwf.readthedocs.io/); [`index.md`](index.md)
is its front page and the canonical contents list.

## How this folder is organised

Every filename is lower-case with hyphens, and the folder supplies the
category, so a name never repeats it: the training guide is
`guides/training.md`, not `guides/TRAINING_GUIDE.md`.

| Folder | Holds | Kind | Naming |
|---|---|---|---|
| [`guides/`](guides) | How to use PyVWF: data, training, outputs, extending it. Maintained, and published to the site. | procedural | `<topic>.md` |
| [`runbooks/`](runbooks) | Per-region acquisition and processing steps, one file per region. Published. | procedural | `<iso-code>.md` |
| [`design/`](design) | Why the code is shaped as it is. Published. | argumentative | `<component>.md` |
| [`findings/`](findings) | Research records, one question each, including the negative results. Kept in the repository, readable on GitHub, deliberately **not** published to the site. | argumentative | `<type>-<subject>.md` |

`findings/` names carry their document type as a prefix, so two files sharing a
prefix share a shape:

- `scorecard.md` is the index of per-region results, and the entry point.
- `region-*.md` reports one region's validation, e.g. `region-nz.md`.
- `method-*.md` studies one method question across regions, e.g.
  `method-cluster-count.md`. A `-<code>` suffix marks a single-region deep dive
  of the same question, as in `method-cluster-count-dk.md`. A `-prereg`
  suffix holds a method's pre-registered gates and predictions. The one older
  record, `method-physics-informed-prespecification.md`, keeps its name as
  history; the different suffix is not drift.
- `dataset-survey.md` surveys candidate observation datasets.

A findings document answers one question. It is revised in place as the
answer changes, and git holds its history. Every result states its training
years and its single test year. A correction to a published claim is a dated
correction notice at the top of the document, never a quiet edit.

### Scorecard markers

A scorecard row carries a marker when its number does not mean what it would
mean alone. The markers are set by rule, from the run's records, not by
anyone remembering:

| Marker | Set when | Read from |
|---|---|---|
| † | The fit is degenerate: a scalar outside 0.2 to 3.0, or an offset that did not converge. | `fit_quality`, in the row's `metrics.csv` |
| ‡ | The gain cannot be distinguished from zero when the test year's units are resampled. | the correction notice that set it |
| § | `extrapolated_capacity_share` is above zero: part of the fleet lies outside the loaded ERA5 extent, and its winds were extrapolated. The share is stated beside the marker. | `extrapolated_capacity_share`, in the row's `metrics.csv` |
| ¶ | `observations_clipped_share` is above zero: part of the observed series sits on the fetcher's 1.5 capacity-factor ceiling, so those values were **discarded rather than wrong** and the metric is computed over fewer observations than the row claims. The share is stated beside the marker. | `observations_clipped_share`, in the row's `metrics.csv` |

`fit_quality` also reports, beside the dagger, the worst shares of training
steps a fitted pair sends below 0 m/s or above the power curve
(`max_below_zero_share`, `max_above_curve_share`, `max_period_dropped_share`).
They do not set the dagger. A bound on them needs its own pre-registered
calibration against the archive, as the scalar bounds had.

A row with a non-zero extrapolated share carries § whether or not its region
opted in to extrapolation. Opting in lets the run finish; it does not make the
number sound. For a row run before the field existed, the share comes from the
region's extent check, and the row cites it.

The findings tree is excluded from the built site on purpose. Several documents
record negative results whose value is the reasoning rather than the number.
Publishing them as site pages would present run-specific figures as guidance.

## Two kinds of document

Terms in every document follow [`CONTEXT.md`](CONTEXT.md), the project's
controlled vocabulary.

**Procedural documents** (`guides/`, `runbooks/`) are for readers who do not
know the project. They follow three rules, adapted from the principles of
ASD-STE100 Simplified Technical English:

1. One instruction per sentence, and one imperative per runbook step. A
   condition goes in the sentence before the instruction, or in a nested
   bullet, never in a subordinate clause.
2. At most 20 words in a procedural sentence, and 25 in a descriptive one.
3. One approved term per concept, each in one sense only, as listed in
   `CONTEXT.md`.

The rules simplify sentence structure only:

- They never remove a proper noun, a document reference or an identifier. A
  searchable name is worth the words it costs.
- A kept reference may push a sentence past the length limit. In that case,
  keep the reference, and mark the sentence with an HTML comment naming what it
  keeps.
- They do not simplify the field's standard technical terms, such as
  "dimensionless". Rule 3 governs the project's own vocabulary.

The rules apply to new procedural documents and to existing ones when they are
next revised.

**Argumentative documents** (`findings/`, `design/`) make a case rather than give
instructions. The procedural rules do not apply. Their readers differ:

- `findings/` is for readers who know the project and its history: the
  maintainer and reviewers.
- `design/` is published, and is for readers who understand the domain but
  have no history with the project. It carries no references to conversations,
  no "as we decided", and no assumed context from earlier work.

Both follow three structural rules instead:

- Define each term once, by citing `CONTEXT.md`, and use it consistently.
- State the claim before the caveat.
- Put reasoning about why the code is shaped as it is in `design/`, not in a
  findings document.
