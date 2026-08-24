# Multi-Region Validation Harness: Design

Status: **implemented**. This document is the design record for the harness
in `vwf/harness/`, kept because the reasoning behind the seams is harder to
recover from the code than the seams themselves. Where the text describes a
future step, read it as the plan that was carried out; the results are in
`docs/findings/`, and the region-by-region numbers in
`docs/findings/scorecard.md`.

The original scope was the European re-runs (DK, DE, UK and nine ENTSO-E
countries) plus Australia (NEM), with transfer restricted to Australia
against Europe. The United States and Brazil were deferred at design time
and added afterwards.

Rev 2 amendments (per design review): transfer semantics specified (§7);
season-name matching for seasonal transfer (§7, tested in §8); observation
granularity `obs_unit` field with verified per-region classification (§2);
AEMO timezone convention decided and recorded in the manifest (§2, §6);
conditions on legacy manifest writing recorded (§6).

## Goals

1. Adding a region or a correction variant is a config change plus (for a new
   observation family) one adapter, never a rewrite.
2. Four concerns separated and independently testable: observed-CF ingestion,
   ERA5 extraction, correction training, skill computation.
3. The affine baseline is wrapped, not modified: alternative correction
   formulations compare against it through a common interface, and the
   validated core in `vwf/correction.py` / `vwf/wind.py` is untouched.
4. Every run's output is self-describing about which curve library produced
   it, without which no synthetic-versus-real curve claim is checkable.
5. The two silent-corruption hazards found in Phase 0 (Northern-Hemisphere
   seasons and the ERA5 longitude convention) are closed by construction,
   not by convention.

## Non-goals (unchanged code)

- `vwf/vwf.py` training loop, `vwf/metrics.py`, `vwf/data.py`, legacy scripts:
  all keep working exactly as today. The harness is additive.
- The country-level joint-offset optimiser: the identifiability work
  *studies* it; nothing changes it.
- Sub-monthly validation: the observation contract stays monthly
  (`obs_1..obs_12`) so Europe and AU are compared like-for-like. AEMO's 5-min
  SCADA is aggregated to monthly; finer resolutions are a possible follow-up,
  out of scope here.
- Antimeridian-crossing ERA5 subsets (AU NEM is 129–154°E; not needed).

## Module layout

```
src/vwf/harness/__init__.py
src/vwf/harness/regions.py       # RegionSpec dataclass + TOML loader/validator
src/vwf/harness/corrections.py   # CorrectionModel ABC, registry, AffineWindCorrection
src/vwf/harness/skill.py         # standard skill metrics on tidy frames
src/vwf/harness/provenance.py    # run_manifest.json writer + curve-library identity
src/vwf/sources/aemo.py          # AEMONemSource (skeleton in Phase 1, real ingest Phase 2)
configs/regions/*.toml           # dk, de, uk, nl, fr, be, no, se, es, it, pt, ie, au_nem
scripts/analysis/validate_region.py       # driver: train / evaluate / transfer
```

`configs/` lives at the repo root and is versioned: these are experiment
definitions, not data, so they do not belong under `input/` (which is
user-supplied and partly gitignored).

## 1. Region configs (`configs/regions/*.toml`)

TOML because the project already ships `tomli` for Python 3.10 and stdlib
`tomllib` on 3.11+: no new dependency. One file per region:

```toml
[region]
code = "AU-NEM"            # free-form id; NOT restricted to ISO-2
name = "Australia (National Electricity Market)"

[observations]
source = "aemo-nem"        # ObservationSource registry name
obs_level = "turbine"      # mechanical pipeline contract: "turbine" | "country"
obs_unit = "farm"          # true independent measurement unit (see §2)
train_years = [2018, 2022] # inclusive
test_years  = [2023]

[era5]
path = "era5/AU"           # relative to PYVWF_INPUT
bbox = [129.0, 154.0, -44.0, -10.0]   # lon_min, lon_max, lat_min, lat_max
file_tag = "AU"            # era5_combined_{year}_{tag}.nc

[correction]
model = "affine-wind"      # CorrectionModel registry name (baseline)
cluster_list = [5]
time_slices = ["fixed", "season"]

[seasons]                  # explicit month groups, no hemisphere heuristics
summer = [12, 1, 2]
autumn = [3, 4, 5]
winter = [6, 7, 8]
spring = [9, 10, 11]
```

Design decision: **seasons are explicit month lists per region**, not a
hemisphere flag. Rationale: (a) the Phase 0 finding was that NH months are
hardcoded in two places; a flag would fix AU but leave the mapping implicit
and unauditable; (b) explicit months also serve regions where meteorological
seasons are not the right split (monsoon and trade-wind regimes), which the
temporal-resolution work needs.
European configs carry the current NH mapping verbatim, so legacy results
reproduce bit-for-bit.

**`[seasons]` is mandatory in every region config**, even when `"season"` is
not among the region's `time_slices`. Rationale: season definitions are used
beyond seasonal correction training: transfer matches slices by season name
against the *target's* definitions (§7.3), and the bias-structure and
temporal-resolution diagnostics group residuals by season regardless of
which slices were trained. A config without
season definitions would make those operations silently fall back to
*something*, and that something would inevitably be the NH mapping this
design exists to eliminate. Configs fail validation if `[seasons]` is missing
or its month lists do not partition 1–12 exactly.

`vwf/harness/regions.py` provides `RegionSpec` (frozen dataclass) and
`load_region(path)` with schema validation and actionable error messages.
The legacy `BoundingBoxes` / `DEFAULT_TRAIN_YEARS` / `pyvwf_config.py`
machinery is not read by the harness and not removed: precedence is simply
"harness reads TOML only; legacy entry points read legacy config only."

## 2. Observation ingestion

Existing seam, unchanged contract (`ObservationSource.load_metadata` /
`load_observations`). New in Phase 1:

### Observation granularity: `obs_unit`

`obs_level` stays exactly what the pipeline needs mechanically ("turbine" |
"country", which training branch runs). A new declarative field, `obs_unit`,
records the *true independent measurement unit* in the config and in every
manifest, so outputs self-describe at their real granularity:

| Region | `obs_level` (pipeline) | `obs_unit` (truth) | Verified basis |
|---|---|---|---|
| DK | turbine | `turbine` | per-GSRN monthly metering, distinct values per turbine |
| DE | turbine | `turbine` | 11,433 unit IDs; outputs within a station prefix are all distinct (per-unit metering). **Caveat: locations are postcode-centroid only**, recorded as `location_resolution = "postcode"` |
| UK | turbine | `farm` | 360 ROC stations expanded to 6,604 turbine rows; monthly values are identical across all rows of a station (farm generation equally pre-split). Recorded as `pseudo_replicated_rows = true` |
| AU-NEM | turbine | `farm` | one row per DUID (wind farm) |
| NL/FR/BE/NO/SE/ES/IT/PT/IE | country | `country` | ENTSO-E national CF |

Verification note (2026-07-15): the Phase 0 label "DE/UK are turbine-level"
was checked against the raw files. UK is farm-level data in turbine-shaped
rows: every multi-turbine station in 2015 has identical monthly values across
its rows, and per-row CF is plausible (median 0.275) only under the
equal-split interpretation. DE is genuinely unit-metered (not farm
aggregates), but with postcode-level geolocation, so DE is classified
`obs_unit = "turbine"` with `location_resolution = "postcode"` rather than
forced into "farm".

#### `station_id_regex`

Regions with `pseudo_replicated_rows = true` must also declare
`station_id_regex`: a regex with exactly one capture group whose match on a
row ID yields the independent-station ID. Semantics:

- Consulted **only** when `pseudo_replicated_rows = true`; other regions
  never have IDs rewritten (DK/DE are untouched by construction; verified:
  zero DK or DE IDs even match the UK pattern).
- IDs that do not match map to themselves (single-unit stations without a
  suffix survive unchanged).
- UK uses `^(.+)-\d+$`. Verified against the data: all 6,618 metadata IDs
  match, producing exactly the 360 stations found independently on the
  observation side. Greedy `(.+)` strips only the *final* `-N`, so
  multi-hyphen IDs like `E_ABRTW-1-1` collapse to the phase-level station
  `E_ABRTW-1`, which is the level at which observations are actually
  replicated (phase counts agree with the 360).
- **Identity guard** (implemented in `collapse_pseudo_replicates`): the
  collapse refuses to run if any (station, time) group carries divergent
  `cf_obs` values. Divergent rows are not replicates of one measurement,
  and averaging them would fabricate an observation. A wrong regex therefore
  fails loudly at evaluation time instead of silently mis-pooling.

Consequences the harness must respect:
- **Effective sample size**: UK has 360 independent observations per month,
  not 6,604. Any per-"turbine" skill claim for the UK is really a per-farm
  claim; `skill.py` reports counts of *independent* units (by `obs_unit`),
  and UK Q-Q/EMD comparisons deduplicate to stations first.
- The DE postcode caveat bounds how much spatial precision DE corrections
  can claim, and any pooling result for DE inherits that bound.

### AEMO timezone convention (decided now, applies to Phase 2 ingest)

**Monthly bins are UTC, pipeline-wide.** AEMO SCADA timestamps (market time,
AEST = UTC+10, no DST) are converted to UTC during ingest, then binned to
calendar months in UTC, the same convention the ERA5/simulation side already
uses everywhere. Rationale: sim-vs-obs bins must share edges within a region,
and one global convention beats per-region bin logic; the cost is that AU
monthly CFs will differ slightly from AEMO's market-time monthly publications
(~10 h of a ~730 h bin at each edge), which is documented and acceptable
because we ingest 5-min SCADA, not published monthly totals. The convention
is recorded in every manifest as `"time_convention": "utc-monthly-bins"`, and
the Phase 1 synthetic AEMO fixtures embed AEST-labelled input timestamps so
the UTC conversion is exercised from the start.

- `AEMONemSource` (`vwf/sources/aemo.py`), `obs_level="turbine"` with one row
  per wind farm (DUID). Phase 1 delivers the adapter skeleton + synthetic-
  fixture tests; the real ingest (Phase 2, after data-download approval) maps:
  - AEMO MMS `DISPATCH_UNIT_SCADA` 5-min MW per DUID → monthly energy →
    monthly CF via registered capacity and hours-in-month;
  - AEMO registration lists + the user's turbine-metadata compilation →
    `ID, lat, lon, height, capacity, model, type`;
  - months before commissioning (or with registered-capacity changes mid-month)
    are masked to NaN rather than diluted.
- **Known limitation, stated up front:** NEM semi-scheduled farms are
  curtailed by dispatch; SCADA records actual (curtailed) output. AU
  corrections will therefore absorb curtailment to a degree European monthly
  generation data does not. This is documented as a caveat on every AU
  finding; correcting for it (semi-dispatch cap flags) is out of scope.

## 3. ERA5 extraction

Two additive changes to `vwf/datasets/era5.py`:

1. **Longitude normalisation**: on load, if any `lon > 180`, convert to
   [-180, 180] and re-sort. Unit-tested with a 0–360 fixture. Closes the
   silent empty-subset hazard.
2. **Config-driven location**: the loader accepts an ERA5 directory + file tag
   from `RegionSpec` instead of only the module-level `ERA5_DATA` constant.
   The constant remains as the legacy default so existing callers are
   untouched.

Acquiring the AU ERA5 subset itself (CDS API download, user credentials,
multi-GB) is a Phase 2 action for the user to run or approve.

## 4. Correction-model interface

`vwf/harness/corrections.py`:

```python
class CorrectionModel(ABC):
    name: ClassVar[str]

    @abstractmethod
    def fit(self, gen_cf, clus_info, reanalysis, power_curves, time_res) -> pd.DataFrame:
        """Return a factors table (long format, one row per cluster × slice × year)."""

    @abstractmethod
    def apply(self, unc_ws, factors, time_res) -> "xr.Dataset":
        """Return corrected wind speeds for simulation."""
```

plus a `register` decorator mirroring `vwf/sources/registry.py`. Phase 1 ships
exactly one implementation: `AffineWindCorrection`, a thin delegate to the
existing `vwf.correction` + `vwf.wind` code paths. A golden regression test
pins that it reproduces the legacy pipeline's outputs bit-for-bit on the
synthetic DK fixture from `tests/test_pipeline.py`. Phase 3 variants register
here as opt-in alternatives; none of them modify the baseline.

## 5. Skill metrics

`vwf/harness/skill.py` computes, from one tidy frame
(`time, ID, cf_sim, cf_obs, capacity, region`):

- MBE, MAE, RMSE, Pearson r (capacity-weighted where applicable);
- CF-distribution comparison: 1-D Earth mover's distance
  (`scipy.stats.wasserstein_distance`, already a scipy dependency) plus
  exported Q-Q quantiles for plotting;
- seasonal-cycle RMSE (error of the mean monthly climatology).

All reported before/after correction, in-sample and held-out. Legacy
`vwf/metrics.py` is untouched; the harness does not call it.

## 6. Run provenance

`vwf/harness/provenance.py` writes `run_manifest.json` into every harness run
directory. This is a shape sketch rather than a literal document: `...`
marks an elided value and `|` an alternative, so it is fenced as text
rather than json.

```text
{
  "pyvwf_version": "...", "git_commit": "...", "git_dirty": false,
  "created_utc": "...",
  "region": {"code": "AU-NEM", "config_sha256": "...", "config": { ... }},
  "curve_library": {
    "power_curves_path": ".../power_curves.csv",
    "power_curves_sha256": "...",
    "models_path": ".../models.csv",
    "models_sha256": "...",
    "library": "synthetic-bundled" | "external",
    "n_curves": 5
  },
  "observations": {"source": "aemo-nem", "obs_level": "turbine", "obs_unit": "farm",
                    "train_years": [2018, 2022], "test_years": [2023],
                    "time_convention": "utc-monthly-bins"},
  "correction": {"model": "affine-wind", "cluster_list": [5], "time_slices": ["fixed", "season"]}
}
```

`library` is determined by SHA-256 equality with the bundled synthetic files in
`vwf/resources/`; anything else is `"external"`. A locally *edited* synthetic
file therefore also labels as `"external"`; this is intentional and fail-safe
in the underclaiming direction (nothing unverified can masquerade as the
bundled synthetic library), and the code will carry a comment saying so.
Contents of external curve files are never copied into the manifest, only
hashes and counts, so the manual `cp models.real.csv models.csv` workflow
stays exactly as it is while every output becomes attributable. The rule is
then mechanical: a result that turns on real rather than bundled curves is
only reportable from a manifest with `library == "external"`.

Legacy `PyVWF.train`/`simulate_cf` also write the manifest (approved), under
two binding conditions:
1. It lands as **its own commit**, independently revertable, touching nothing
   else.
2. A manifest-write failure must **never abort a run**: the writer is wrapped
   so any exception logs a warning and the run continues. This holds for the
   harness driver too: provenance is diagnostic, not load-bearing.

## 7. Driver

`scripts/analysis/validate_region.py`:

```
validate_region.py --region configs/regions/au_nem.toml train
validate_region.py --region configs/regions/au_nem.toml evaluate
validate_region.py --factors-from output/validation/AU-NEM/<run>/ \
                   --region configs/regions/uk.toml transfer
```

- Outputs under `output/validation/<region-code>/<run-id>/` with
  `run_manifest.json`, `factors.csv`, `metrics.csv`, and `figures/`.
  `run-id` = UTC timestamp + short config hash.
### Transfer semantics (normative)

Cluster labels are region-specific: a source region's factors table has no
cluster correspondence in the target region. Transfer is therefore defined as
exactly the following, and the driver implements this and nothing else:

1. Collapse the source region's factors to **one scalar and one offset per
   time-slice**, as the **capacity-weighted** mean over the source clusters
   (weights = installed capacity per cluster from the source run's
   `turb_info`; never an unweighted mean). Multi-year factors are first
   averaged over training years within (cluster × slice), then collapsed.
2. Apply that single (scalar, offset) pair uniformly to every site in the
   target region's simulation for the matching time-slice.
3. **Seasonal slices match by season NAME, not by month.** With explicit
   month lists, AU `winter = [6,7,8]` and UK `winter = [12,1,2]`;
   winter-trained factors apply to the target's winter *as defined in the
   target's config*. Matching by month would silently reintroduce the
   hemisphere inversion this design exists to eliminate. (`fixed` and
   bimonth slices match by label as well; bimonth transfer AU↔EU is
   physically questionable and excluded from the approved pair runs.)

No spatial matching schemes (nearest-cluster, regridding, wind-climate
analogues) are implemented.

- `transfer` applies a foreign factors table to the target region's simulation
  and evaluates. The driver enforces the approved pair set (AU↔Europe);
  anything else exits with an explanatory error rather than silently running.

## 8. Phase 1 test plan (all synthetic, no network)

1. Region-config loader: round-trip, missing-field and bad-month validation.
2. Southern-hemisphere end-to-end: mirror of the synthetic-DK pipeline test
   with an AU-style config; asserts season grouping follows the config months,
   not the legacy NH map.
3. ERA5 longitude normalisation on a 0–360 fixture; no-op on [-180, 180].
4. Provenance: synthetic-vs-external detection, hash stability, manifest schema.
5. Golden regression: `AffineWindCorrection` output identical to the legacy
   path on the existing synthetic fixture.
6. `AEMONemSource` unit tests on synthetic SCADA fixtures (commissioning mask,
   kWh/CF arithmetic, capacity-change month masking, and AEST→UTC monthly
   binning per the §2 convention; fixtures carry AEST-labelled timestamps).
7. Transfer driver: pair-set enforcement.
8. **Mirrored-hemisphere transfer fixture** (normative for §7.3): two
   synthetic regions identical except for season definitions (NH months vs
   SH months), with season-dependent planted biases. The test computes the
   transferred output under (a) season-name matching and (b) month matching,
   asserts the two *differ*, and asserts the name-matched result is the
   correct one. It must distinguish the two modes: a fixture on which both
   modes agree (e.g. season-independent factors) would pass vacuously and is
   explicitly not acceptable.
9. Transfer collapse arithmetic: capacity-weighted collapse of a synthetic
   factors table with unequal cluster capacities; asserts the unweighted mean
   would differ (same must-distinguish principle as test 8).

## 9. Risks

- **AEMO ingest complexity** (capacity evolution, curtailment, DUID↔farm
  mapping) is the schedule risk. Mitigation: adapter contract fixed in
  Phase 1 against synthetic fixtures, so ingest problems stay contained.
- **Dual config systems** (legacy dicts + TOML) until/unless legacy paths are
  migrated. Accepted for now; precedence is documented above.
- **ERA5 AU acquisition** needs CDS credentials and multi-GB downloads; user
  executes or approves in Phase 2.
- Monthly observation contract limits distribution comparison to monthly CFs
  for training parity; the EMD/Q-Q comparison for AU can additionally be run
  at native resolution *descriptively*, but the headline metrics stay monthly.

## Decisions (resolved at design review, 2026-07-15)

1. Package name: `vwf.harness`.
2. `configs/regions/` at repo root.
3. Seasons as explicit month lists per region config.
4. Legacy `PyVWF.train`/`simulate_cf` write the manifest: own commit,
   never-abort semantics (§6).
5. Driver is a script; a console entry point may be promoted later if the
   harness stabilises.

Amendments A–D from the same review are folded into §7 (transfer semantics,
season-name matching), §2 (`obs_unit` granularity with verified
classification; AEMO timezone convention), §6 (manifest fields and legacy
conditions), and §8 (must-distinguish transfer tests).
