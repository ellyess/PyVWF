**Verdict: REJECT** (in its current form; replace it with a record kept in each run)

**Steelman the objection.** An audit of the grid files on disk today says nothing about which grid an earlier figure was computed on. `weight_country_grid_points.py` overwrites `<c>_grid_points.csv` in place (lines 332-337) and rewrites every `_<year>.csv` (line 285), and the grids have been rebuilt before. Labelling the two earlier Belgian figures from today's table would only look like provenance. The resolver is also the wrong one. The harness does not call `load_year_specific_grid_points`. `driver.resolve_source` builds `EntsoeFileSource`, which reads one file per split: `_grid_points_<fleet_year>.csv`, where fleet_year is the last training year for train and the test year for test, and it falls back to the base file. So a single figure uses two grids. A table of grids per training year would record grids that training never read.

**Cheaper test.** Each training run already writes `train_turb_info_<n>.csv` into its run directory (driver.py lines 238-240). That file holds the points and capacities the run actually used. Reading it from the existing BE run directories labels the BE figures at no cost. For future runs, add the grid path, its hash, its point count and its capacity to `write_manifest_safe`'s `extra`, the way `curve_resolution` is recorded. The record then stays with the number.

**Simpler explanation.** The expected finding (which regions are fleet-weighted and which uniform) is already in `docs/findings/method-country-level.md` §3. The table would mostly restate it.

**Kill criterion.** None is named. Drop the idea if the existing `train_turb_info` files already give the grid for every reported country figure. Drop it as well if the BE figures predate the current files, because then the audit cannot label them.

**Opportunity cost.** Small in effort. But a separate table that goes stale delays the manifest field, the version of this that would keep working.

Files read:
/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221/src/vwf/loaders/country_level_loaders.py
/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221/src/vwf/sources/entsoe_files.py
/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221/src/vwf/sources/entsoe_zonal.py
/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221/src/vwf/vwf.py
/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221/src/vwf/harness/driver.py
/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221/docs/findings/method-country-level.md
/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/a75ab221/scripts/region_tools/weight_country_grid_points.py
