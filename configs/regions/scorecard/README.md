# Scorecard configurations

The exact region configuration behind each row of
[`docs/findings/scorecard.md`](../../../docs/findings/scorecard.md), one file
per row, byte-identical to the file each run was made from.

**The canonical name always points at the row standing today.** When a row is
superseded, its configuration moves unchanged to `superseded/<date>/` under the
same name and the new one takes the canonical name, so a reader reaching for
`fr_country.toml` gets the current row and the scorecard's Superseded section
cites the dated path. The six rows never superseded still date from 2026-08-24
(commit `41462e9`, clean tree); the eleven European rows were superseded on
2026-09-13 (`../../../docs/findings/method-eu-rerun.md`).

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
python scripts/analysis/validate_region.py train --region configs/regions/scorecard/ie_country.toml
```

A configuration does not fix the curve library. Each run's `run_manifest.json`
records the library it resolved, by sha256, and the scorecard states it per
region; the rows that used the licensed library are not reproducible without it.

The comments inside each file are as they stood when the row was run, and some
now name documents that have since been renamed or describe processing that has
since changed. The settings, not the comments, are the record. Do not edit these
files: a changed configuration is a new row, with a new file.
