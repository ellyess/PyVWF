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

| Folder | Holds | Naming |
|---|---|---|
| [`guides/`](guides) | How to use PyVWF: data, training, outputs, extending it. Maintained, and published to the site. | `<topic>.md` |
| [`runbooks/`](runbooks) | Per-region acquisition and processing steps, one file per region. Published. | `<iso-code>.md` |
| [`design/`](design) | Why the code is shaped as it is. Published. | `<component>.md` |
| [`findings/`](findings) | Dated research records: one experiment each, including the negative results. Kept in the repository, readable on GitHub, deliberately **not** published to the site. | `<type>-<subject>.md` |

`findings/` names carry their document type as a prefix, so two files sharing a
prefix share a shape:

- `scorecard.md` is the index of per-region results, and the entry point.
- `region-*.md` reports one region's validation, e.g. `region-nz.md`.
- `method-*.md` studies one method question across regions, e.g.
  `method-cluster-count.md`. A `-<code>` suffix marks a single-region deep dive
  of the same question, as in `method-cluster-count-dk.md`.
- `dataset-survey.md` surveys candidate observation datasets.

The findings tree is excluded from the built site on purpose. Each document
reports one dated experiment against one held-out test year, and several record
negative results whose value is the reasoning rather than the number, so
publishing them as site pages would present run-specific figures as guidance.
