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

The findings tree is excluded from the built site on purpose. Several documents
record negative results whose value is the reasoning rather than the number.
Publishing them as site pages would present run-specific figures as guidance.

## Two kinds of document

Terms in every document follow [`CONTEXT.md`](../CONTEXT.md), the project's
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
