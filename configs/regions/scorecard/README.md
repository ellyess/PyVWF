# Scorecard configurations

The exact region configuration behind each row of
[`docs/findings/scorecard.md`](../../../docs/findings/scorecard.md), one file
per row, byte-identical to the file each run was made from on 2026-08-24
(commit `41462e9`, clean tree).

These are records, not the maintained region configs. `configs/regions/<code>.toml`
carries the cluster sweep a region is developed with; the files here fix the
single cluster count and time slice a scorecard row reports. For the eight
country-level regions the two are identical. For the nine turbine-level regions
they are not, and for seven of them (all but DK and UK) the maintained sweep does
not contain the reported cluster count, so re-running the maintained config does
not reproduce the row. Re-run a row from its file here:

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
