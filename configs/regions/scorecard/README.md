# Scorecard configurations

The region configuration behind each row of the
[scorecard](../../../docs/scorecard.md), one file per row. Each is the file its
run was made from, with three later edits that keep each run's meaning:
`br_k60.toml` and `us_k250.toml` gained `roughness = "stored"` (`a023b4d`,
2026-09-20), the treatment they had run under, when the default changed to
derived; and a comment in `uk_k50.toml` and `us_k250.toml` names the renamed
package (`4d06163`). *[Corrected 2026-10-27: this file said every
configuration was byte-identical to its run's.]*

**The canonical name always points at the row standing today.** When a row is
superseded, the new configuration takes the canonical name, so a reader
reaching for `fr_country.toml` gets the current row; the old file stays in the
history. The eleven European rows superseded on 2026-09-13 are kept under
`superseded/2026-09-13/` at tag `archive/research-2026-10-27`. Those gained
`roughness = "stored"` on 2026-10-27, after their runs (`8105277`), the
treatment they had run under, and the superseded `uk_k50.toml` the same
comment edit (`4d06163`).

These are records, not the maintained region configs. `configs/regions/<code>.toml`
carries the cluster sweep a region is developed with; the files here fix the
cluster count a scorecard row reports. For the nine turbine-level regions the
two differ, and for seven of them (all but DK and UK) the maintained sweep does
not contain the reported cluster count, so re-running the maintained config does
not reproduce the row.

**Corrected 2026-09-13.** This file said the configurations here fix the single
cluster count and time slice a row reports. Eleven of the seventeen carry more
than one variant: all eight country-level rows carry four, and AR, BR, CL and
US carry two. The row reports one of them, named in the scorecard's Best cfg
column, and `tests/test_scorecard_configs.py` checks that the configuration can
produce it. The others are not incidental: every variant of a run is scored on
the rows common to all of them, so a variant a row does not report still moves
the one it does (`../../../docs/guides/output-structure.md`). The variant set is
part of a row's design.

Re-run a row from its file here:

```bash
pyvwf-validate train --region configs/regions/scorecard/ie_country.toml
```

A configuration does not fix the curve library. Each run's `run_manifest.json`
records the library it resolved, by sha256, and the scorecard states it per
region; the rows that used the licensed library are not reproducible without it.

The comments inside each file are as they stood when the row was run, and some
now name documents that have since been renamed or describe processing that has
since changed. The settings, not the comments, are the record. Do not edit these
files: a changed configuration is a new row, with a new file.
