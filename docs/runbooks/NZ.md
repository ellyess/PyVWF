# New Zealand (EMI)

**Source:** EA EMI `Generation_MD` half-hourly metered injection.
**Adapter:** `emi-nz` · turbine-level · unit = farm · 🟢 open.
**Fleet:** 13 dispatched farms, ~1.5 GW, with per-farm hub heights.

All inputs are open Electricity Authority data (no registration) or your own
CDS/ERA5 credentials; none of the raw or derived data is committed (`input/` is
git-ignored), so this runbook is how you build it. NZ was the #1 pick of the
July-2026 dataset survey
(`docs/findings/dataset_survey_2026-07.md`): per-plant half-hourly metered
generation, openly downloadable, in a Southern-Hemisphere temperate-westerly
complex-terrain climate the validation set does not yet cover, and with
per-farm hub heights, which no other non-European region except Canada has.

## 0. What is already committed

- `configs/curation/nz_wind_farms.csv`: the curated farm table (13 dispatched farms):
  Gen_Code/POC keys into EMI files, coordinates, final-build capacity,
  turbine model, hub height with per-farm provenance (`height_source` marks
  the unverified ones: tararua_3, mill_creek, kaiwera_downs_2). Compiled
  July 2026 from NZWEA/operator/Wikipedia/EMI-register sources.
- `configs/curation/nz_capacity_stages.csv`: stable capacity plateaus for staged
  builds (currently Turitea North-only 118.8 MW → full 221.4 MW).
- `configs/curation/nz_mask_windows.csv`: commissioning-ramp month windows masked at
  load (Waipipi, Turitea x2, Harapaki, Kaiwera Downs 1 and 2).
- The adapter (`vwf/sources/emi_nz.py`), transforms
  (`vwf/datasets/emi_nz.py`), and their tests (DST trading-period mapping is
  pinned by must-distinguish tests).

Known exclusions (documented, not silent): **Mahinerangi** is metered inside
the Waipori hydro scheme and never appears as wind in Generation_MD; the
seven small **embedded** farms (Brooklyn, Hau Nui, Mt Stuart, Flat Hill,
Horseshoe Bend, Weld Cone, Lulworth, ~28 MW total) are distribution-connected
and outside the dispatched dataset. **Te Rere Hau** is included but degraded
late-window (5 turbines stopped, 2 derated); its observed CF understates
the resource; a standing caveat and an exclusion candidate if it distorts.

## 1. Observations (user-executed, no credentials)

```bash
python scripts/fetch/emi_nz.py            # 72 monthly CSVs 2019-2024 + register
python scripts/process/emi_nz.py          # -> input/observations/turbine/NZ/
```

The fetch is plain HTTP (each URL 302-redirects to an open Azure blob),
~0.4-0.9 MB per monthly file. The processing step selects wind rows
(`Fuel_Code` in {Wind, WIN}), keys on case-normalised `Gen_Code` (Site_Code
is not stable across years), maps trading periods to UTC through
`Pacific/Auckland` (46/48/50-period DST days handled and pinned by tests),
sums multi-POC farms (West Wind, Tararua I/II), computes monthly CF against
the stable-plateau capacity history, and writes the build mask. An unmapped
wind Gen_Code is a hard error: it means a new farm needs a curated row
(Kaiwaikawe, Northland 77 MW, is expected to appear ~mid-2026).

Then read `input/observations/turbine/NZ/join_report.md` before trusting
anything: farm count (13), capacity (~1.5 GW), matched-curve count, masked
months.

## 2. ERA5 (user-executed, your CDS key)

```bash
python scripts/fetch/era5.py --region nz           # 72 months, 2019-2024, NZ box
```

Small box (53 x 49 cells), so no daily pre-combine is needed, unlike BR/US.

## 3. Train and evaluate

```bash
PYVWF_INPUT=<your input root with the real/combined curve library> \
python scripts/analysis/validate_region.py train --region configs/regions/nz.toml
python scripts/analysis/validate_region.py evaluate --region configs/regions/nz.toml \
    --train-run output/validation/NZ/train-<stamp>
```

Notes for reading the result:

- **Bias-structure diagnosis first** (the D2 lesson): check uncorrected MBE
  before judging the correction. NZ CFs are among the world's highest
  (~40%); ERA5 in complex terrain (Manawatu Gorge, Cook Strait funnelling)
  plausibly under-resolves the resource, the Tehachapi-like regime the ML
  transfer re-test identified as globally under-represented.
- **k ceiling**: 13 farms, 12 unique coordinates (Tararua III shares the
  I/II ridge coordinates; KD2 shares KD1's). `cluster_list = [5]` shipped;
  sweep [1, 5, 10] and remember k above ~12 is a fake plateau
  (`docs/findings/us_br_first_run.md`).
- **Train/test**: 2019-2023 → 2024. Turitea contributes 2022 (North plateau)
  and 2024; its 2021/2023 ramps are masked. Harapaki effectively enters at
  test time (masked to Jul 2024); watch its months in evaluation.
- Metered injection is net of availability; NZ sees little economic wind
  curtailment over this window (hydro-dominated system), but no
  curtailment screen exists: a standing caveat, unlike BR.

## 4. Refresh path

- New months: re-run both fetch scripts (they skip existing files).
- New farms (Kaiwaikawe, Mt Munro, Te Rere Hau repowering): add a row to
  `configs/curation/nz_wind_farms.csv` (+ stages/mask windows if staged); the
  processing step will fail loudly until you do.
- The register filename is publication-dated (~6-monthly); the fetch script
  scrapes the directory for the newest. EMI has signalled Generation_MD will
  eventually be superseded by a richer dataset; if fetches 404, check the
  EMI dataset page.
