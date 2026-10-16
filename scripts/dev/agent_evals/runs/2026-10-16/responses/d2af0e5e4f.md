**Verdict:** REVISE. A grid audit is worth having, but this one picks its file in a way the code does not, so it would label figures with grids they were never computed on.

**Blockers:**
1. **The audit picks the wrong file.** The proposal sorts the per-year files and takes the newest. The code does something else. `EntsoeFileSource._grid_points_candidates` (`/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221/src/vwf/sources/entsoe_files.py:66-87`) chooses the file by the split's fleet year:
   - training uses the file for `train_years[1]`;
   - testing uses the file for `test_year`;
   - if that file is missing, it falls back to `<c>_grid_points.csv`.

   For BE (`/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221/configs/regions/be.toml`: training 2015-2021, test year 2023), training reads the 2021 file and testing reads the 2023 file. If a 2024 file exists, the audit reports a file neither split read. The `_20*` pattern also skips the fallback file, so a region running on its static grid is reported "missing". AGENTS.md forbids this ("Never sort a glob and take the last entry"). Fix: write one row per region config and split, and get the path from the source the driver builds (`resolve_source(spec, split)` in `/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221/src/vwf/harness/driver.py:66`). That needs a public `grid_points_path()` on `EntsoeFileSource`, since the current method is private.
2. **A file's name does not say whether it is uniform or weighted.** Without `--per-year`, `/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221/scripts/region_tools/weight_country_grid_points.py:322-337` overwrites `<c>_grid_points.csv` with weighted points and backs up the uniform grid. Fix: decide uniform or weighted from the file's contents (the `capacity` and `weight` columns), not its name.
3. **The two Belgian figures cannot be labelled from today's files.** The manifest records no grid path or hash (`/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221/src/vwf/harness/provenance.py:270-288`), so nothing shows which grid an earlier figure was computed on. Fix: record the grid path and its sha256 for each split in `build_manifest`. Leave figures that came before that change labelled "grid unrecorded". The Belgian figures must also state training 2015-2021 and test year 2023.

**Plan:**
1. Add a public `grid_points_path()` in `src/vwf/sources/entsoe_files.py`, with fast tests in `/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221/tests/test_harness_entsoe_files.py` using fake files under `tmp_path`:
   - files for 2015 to 2024 present: training resolves to 2021, testing to 2023 (the case the proposal gets wrong);
   - only the static file present: resolves to the static file (a negative).
2. Record the grid path and hash in the manifest (`provenance.py` and `driver.py`), tested in `/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221/tests/test_harness_provenance.py`.
3. Add `scripts/analysis/grid_audit.py`, importing only public `vwf` names. It writes under `output/` and changes nothing in the tree. Its output stays local, because it describes input data.

One concern per commit. Step 2 changes harness behaviour, so run `pytest -m "not slow and not realdata"` plus mypy. It adds manifest keys only and should not move any metrics.

**Risks:** Labels taken from today's files would quietly be wrong for any figure computed before the grids were rebuilt. Zonal SE inherits grid loading from the national source (`/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221/src/vwf/sources/entsoe_zonal.py:55`), so a second SE row is needed.

**This checkout differs from what the review checklist assumes:** the package is `vwf`, not `pyvwf`. There is no `.importlinter`, no `src/*/AGENTS.md` and no `tests/test_public_analysis_api.py`, so I could not check the layer contracts or the public-API test.

Files read:
/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221/AGENTS.md
/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221/CONTRIBUTING.md
/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221/src/vwf/sources/entsoe_files.py
/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221/src/vwf/loaders/country_level_loaders.py
/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221/src/vwf/harness/driver.py
/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221/src/vwf/sources/entsoe_zonal.py
/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221/src/vwf/harness/provenance.py
