# The multi-region validation harness

Why the seams in `vwf/harness/` are where they are. The seams themselves are
readable from the code; the reasoning behind them is not, which is what this
document is for. Region-by-region results are in
[`../findings/scorecard.md`](../findings/scorecard.md).

The harness is **additive**. The legacy `PyVWF` training loop, `vwf/metrics.py`,
`vwf/data.py` and the legacy scripts all keep working unchanged, and the
affine baseline is wrapped rather than modified, so an alternative correction
compares against the validated core in `vwf/correction.py` and `vwf/wind.py`
without touching it.

## What it is built to make cheap

1. Adding a region is a config change plus, for a new observation family, one
   adapter. Never a rewrite.
2. Four concerns separated and independently testable: observed-CF ingestion,
   ERA5 extraction, correction training, skill computation.
3. Every run self-describes which curve library produced it. Without that, no
   claim about bundled-versus-licensed curves is checkable.
4. Two silent-corruption hazards are closed by construction rather than by
   convention: Northern-Hemisphere seasons, and the ERA5 longitude convention.

```
src/vwf/harness/regions.py       # RegionSpec dataclass + TOML loader/validator
src/vwf/harness/corrections.py   # CorrectionModel ABC, registry, AffineWindCorrection
src/vwf/harness/driver.py        # run_train / run_evaluate / run_transfer
src/vwf/harness/skill.py         # skill metrics on tidy frames
src/vwf/harness/provenance.py    # run_manifest.json + curve-library identity
src/vwf/harness/export.py        # gridded correction fields
src/vwf/harness/hindcast.py      # applying fitted factors to other years
src/vwf/sources/                 # one ObservationSource adapter per data family
configs/regions/*.toml           # one file per region
scripts/analysis/validate_region.py   # driver CLI: train / evaluate / transfer
```

`configs/` lives at the repository root and is versioned: these are experiment
definitions, not data, so they do not belong under `input/`, which is
user-supplied and partly gitignored.

## Region configs

TOML, because the project already ships `tomli` for Python 3.10 and stdlib
`tomllib` covers 3.11+, so it costs no new dependency.

```toml
[region]
code = "AU-NEM"            # free-form id, NOT restricted to ISO-2
name = "Australia (National Electricity Market)"

[observations]
source = "aemo-nem"        # ObservationSource registry name
obs_level = "turbine"      # which training branch runs: "turbine" | "country"
obs_unit = "farm"          # the true independent measurement unit
train_years = [2018, 2022] # inclusive
test_years  = [2023]

[era5]
path = "era5/AU"           # relative to PYVWF_INPUT
bbox = [129.0, 154.0, -44.0, -10.0]   # lon_min, lon_max, lat_min, lat_max
file_tag = "AU"            # era5_combined_{year}_{tag}.nc

[correction]
model = "affine-wind"      # CorrectionModel registry name
cluster_list = [5]
time_slices = ["fixed", "season"]

[seasons]                  # explicit month groups, no hemisphere heuristics
summer = [12, 1, 2]
autumn = [3, 4, 5]
winter = [6, 7, 8]
spring = [9, 10, 11]
```

**Seasons are explicit month lists per region, not a hemisphere flag.** A flag
would fix Australia while leaving the mapping implicit and unauditable, and
explicit months also serve regions where meteorological seasons are not the
right split at all, such as monsoon and trade-wind regimes. European configs
carry the Northern-Hemisphere mapping verbatim, so legacy results reproduce
bit-for-bit.

`[seasons]` is **mandatory even when `"season"` is not among the region's
`time_slices`**, because season definitions are used beyond seasonal training:
transfer matches slices by season name against the target's definitions, and
the bias-structure and temporal-resolution diagnostics group residuals by
season regardless of what was trained. A config without them would make those
operations fall back to something, and that something would be the
Northern-Hemisphere mapping this design exists to eliminate. Configs fail
validation if `[seasons]` is missing or its month lists do not partition 1 to 12
exactly.

Precedence with the legacy config machinery is mechanical: the harness reads
TOML only, legacy entry points read legacy config only.

## Observation granularity

`obs_level` is mechanical, naming which training branch runs. `obs_unit` records
the true independent measurement unit, in the config and in every manifest, so
outputs self-describe at their real granularity.

| Region | `obs_level` | `obs_unit` | Basis |
|---|---|---|---|
| DK | turbine | `turbine` | per-GSRN monthly metering, distinct values per turbine |
| DE | turbine | `turbine` | 11,433 unit IDs, all distinct within a station prefix. **Locations are postcode centroids**, recorded as `location_resolution = "postcode"` |
| UK | turbine | `farm` | 360 ROC stations expanded to 6,604 rows with identical monthly values, so farm generation is pre-split equally. Recorded as `pseudo_replicated_rows = true` |
| AU-NEM | turbine | `farm` | one row per DUID |
| ENTSO-E countries | country | `country` | national CF |

The UK case is data at farm level in turbine-shaped rows: every multi-turbine
station has identical monthly values across its rows, and per-row CF is
plausible (median 0.275) only under the equal-split reading. Germany is
genuinely unit-metered but geolocated to postcode, so it is `obs_unit =
"turbine"` with the resolution caveat rather than forced into "farm".

The consequences are load-bearing. The UK has 360 independent observations per
month, not 6,604, so `skill.py` reports counts of independent units and UK
distribution comparisons deduplicate to stations first. Any per-turbine skill
claim for the UK is really a per-farm claim, and Germany's postcode caveat
bounds how much spatial precision a German correction can claim.

Regions with `pseudo_replicated_rows = true` declare a `station_id_regex` with
one capture group yielding the independent-station ID. It is consulted only for
those regions, IDs that do not match map to themselves, and
`collapse_pseudo_replicates` **refuses to run if any (station, time) group
carries divergent `cf_obs` values**: divergent rows are not replicates of one
measurement, and averaging them would fabricate an observation. A wrong regex
fails loudly at evaluation time instead of silently mis-pooling.

## ERA5 extraction

Two additive changes to `vwf/datasets/era5.py`. Longitude is normalised to
[-180, 180] on load if any value exceeds 180, and re-sorted, which closes the
silent empty-subset hazard. The loader accepts a directory and file tag from
`RegionSpec` rather than only the module-level constant, which remains as the
legacy default.

## Correction models

```python
class CorrectionModel(ABC):
    name: ClassVar[str]

    @abstractmethod
    def fit(self, gen_cf, clus_info, reanalysis, power_curves, time_res) -> pd.DataFrame:
        """Return a factors table, one row per cluster x slice x year."""

    @abstractmethod
    def apply(self, unc_ws, factors, time_res) -> "xr.Dataset":
        """Return corrected wind speeds for simulation."""
```

`AffineWindCorrection` is a thin delegate to the existing `vwf.correction` and
`vwf.wind` paths, pinned by a golden regression test that requires bit-for-bit
agreement with the legacy pipeline on the synthetic Denmark fixture. Variants
register alongside it and none modify the baseline.

## Skill metrics

From one tidy frame (`time, ID, cf_sim, cf_obs, capacity, region`): MBE, MAE,
RMSE and Pearson r, capacity-weighted where applicable; a capacity-factor
distribution comparison by 1-D Earth mover's distance with exported Q-Q
quantiles; and seasonal-cycle RMSE against the mean monthly climatology. All
reported before and after correction, in sample and held out. Legacy
`vwf/metrics.py` is untouched and the harness does not call it.

## Run provenance

Every harness run writes `run_manifest.json`: package version, git commit and
dirty flag, creation time, the region config and its hash, the observation
source with `obs_level`, `obs_unit`, years and time convention, the correction
model and its settings, and the curve library.

The curve-library block carries paths, SHA-256 hashes, a count, and a `library`
field of `"synthetic-bundled"` or `"external"`, decided by hash equality with
the bundled files in `vwf/resources/`. A locally *edited* bundled file therefore
also labels as `"external"`, which is deliberate and fail-safe in the
underclaiming direction: nothing unverified can masquerade as the bundled
library. Contents of external curve files are never copied into the manifest,
only hashes and counts, so the manual `cp models.real.csv models.csv` workflow
is unchanged while every output becomes attributable. The rule is then
mechanical: a result that turns on licensed rather than bundled curves is only
reportable from a manifest with `library == "external"`.

The legacy `PyVWF.train` and `simulate_cf` write the manifest too. A
manifest-write failure never aborts a run, in the harness or the legacy path:
provenance is diagnostic, not load-bearing.

## Driver

```bash
validate_region.py --region configs/regions/au_nem.toml train
validate_region.py --region configs/regions/au_nem.toml evaluate
validate_region.py --factors-from output/validation/AU-NEM/<run>/ \
                   --region configs/regions/uk.toml transfer
```

Outputs land under `output/validation/<region-code>/<run-id>/`, with `run-id` a
UTC timestamp plus a short config hash.

**Transfer semantics.** Cluster labels are region-specific, so a source
region's factors have no cluster correspondence in the target. Transfer is
therefore defined as exactly this, and the driver implements nothing else:

1. Collapse the source factors to one scalar and one offset per time slice, as
   the **capacity-weighted** mean over source clusters, never an unweighted one.
   Multi-year factors are averaged over training years within (cluster, slice)
   first.
2. Apply that single pair uniformly to every site in the target region for the
   matching slice.
3. **Seasonal slices match by season name, not by month.** Australian
   `winter = [6,7,8]` and British `winter = [12,1,2]` both mean winter;
   matching by month would silently reintroduce the hemisphere inversion.

No spatial matching scheme is implemented: no nearest-cluster, no regridding,
no wind-climate analogues. The driver enforces the validated pair set,
Australia against Europe, and exits with an explanatory error on anything else
rather than silently running it.

## Tests worth knowing about

Two are written to **distinguish** rather than confirm, which is why they are
worth more than their line count. The mirrored-hemisphere transfer fixture uses
two synthetic regions identical except for season definitions, with
season-dependent planted biases; it asserts that name matching and month
matching give *different* answers and that the name-matched one is correct. A
fixture on which both modes agree would pass vacuously and is explicitly not
acceptable. The transfer-collapse test likewise asserts that the unweighted mean
would differ from the capacity-weighted one on its fixture.

The rest are conventional: config round-trip and validation, a
Southern-Hemisphere end-to-end mirror of the synthetic Denmark pipeline test,
ERA5 longitude normalisation, provenance hashing and schema, the golden affine
regression, per-adapter ingest arithmetic, and pair-set enforcement. All
synthetic, no network.

## Known limits

- **Monthly observation contract** (`obs_1..obs_12`), so regions are compared
  like for like. AEMO's 5-minute SCADA is aggregated to monthly. Finer
  resolutions are possible but not part of the contract.
- **Monthly bins are UTC, pipeline-wide.** AEMO SCADA is market time (AEST,
  UTC+10, no DST) and is converted on ingest. Australian monthly CFs therefore
  differ slightly from AEMO's market-time publications, by about 10 hours of a
  730-hour bin at each edge. Recorded in every manifest as
  `"time_convention": "utc-monthly-bins"`.
- **NEM semi-scheduled farms are curtailed by dispatch** and SCADA records
  actual output, so Australian corrections absorb curtailment in a way European
  monthly generation data does not. This caveat travels with every Australian
  finding.
- **Two config systems** coexist, legacy dicts and TOML, with the precedence
  rule above.
- **Antimeridian-crossing ERA5 subsets are not supported.** No current region
  needs one.
