# Denmark (Danish Energy Agency) region: acquisition and processing runbook

Denmark is the programme's **reference region**: the level-dominated case
where the affine correction wins on every metric (D1/D2), and the lineage of
the Staffell & Pfenninger (2016) validation. Unlike the other regions it keeps
the shared `european-turbine` adapter (DK/DE/UK); this runbook only scripts the
data acquisition that used to be a manual ens.dk download.

## Source

The **Master data register of wind turbines** ("Stamdataregister for
vindkraftanlæg"), published by the Danish Energy Agency (Energistyrelsen) from
its energy-sector data page,
<https://ens.dk/en/analyses-and-statistics/overview-energy-sector>. It is the
national register of every grid-connected Danish turbine > 6 kW, with location,
technical specifications, and monthly output, the most complete open
per-turbine wind dataset anywhere, which is why Denmark anchors the validation.

Two workbooks:

| File | ens.dk media | Content |
| --- | --- | --- |
| `anlaeg.xlsx` | `/media/4945/download` | Master data register: one row per turbine (GSRN id, connection date, capacity kW, rotor diameter, hub height, UTM X/Y, manufacturer/model), split into "existing" and "decommissioned" sheets. Header on row 9. |
| `maanedsdata_2002_2020.xlsx` | `/media/4948/download` | Monthly production to grid: one sheet per year (`Månedsprod_<year>`, 2002-2020), kWh per turbine per month. Header on row 7. |

**Vintage.** The pinned media ids are the register snapshot taken ultimo
January 2022, the exact files the committed Denmark results were built on, so
the fetch reproduces them bit-for-bit (verified: byte-identical). ens.dk serves
the monthly file under the stale name `maanedsdata_2002_2017.xlsx`, but the
workbook carries all 19 years through 2020; the fetch saves it under the
`_2020` name the processor expects. The 2002-2020 production history is fixed;
the *master data register* refreshes monthly under a new media id; pass
`--anlaeg-url <current link>` to `fetch/dk.py` for a newer snapshot (only adds
post-2022 turbines; does not change the 2002-2020 production).

## 1. Acquire + process (user-executed, no credentials)

```bash
python scripts/fetch/dk.py       # anlaeg.xlsx + maanedsdata -> input/observations/turbine/DK/
python scripts/process/dk.py     # -> dk_md.csv, dk_obs_2002_2020.csv
```

`process` (pure transforms in `vwf.datasets.process_dk_raw_data`) reads the
header-offset sheets, converts the UTM X/Y coordinates to lon/lat, maps
`Land`/`Hav` to onshore/offshore, and reshapes the per-year production sheets
into the long `ID, year, month, generation_kwh, connection_date,
decommission_date` frame. On the pinned files it yields **6,297 turbines** and
**1,229,634 turbine-months** (2002-2020).

The `european-turbine` adapter reads `dk_md.csv` + `dk_obs_2002_2020.csv`
directly (converting monthly kWh to capacity factor at load), so no further
step is needed.

## 2. Train and evaluate

```bash
PYVWF_INPUT=<combined-library root> \
python scripts/analysis/validate_region.py train    --region configs/regions/dk.toml
python scripts/analysis/validate_region.py evaluate --region configs/regions/dk.toml \
    --train-run output/validation/DK/train-<stamp>
```

Denmark trains on 2015-2019 and tests on 2020 (`dk.toml`). It is
level-dominated (uncorrected MBE +0.121), and the affine correction wins on
every metric, the anchor the other regions are read against
(`docs/findings/d1_regression.md`, `d2_synthesis.md`). ERA5 for DK comes from
the shared European box (`era5/EU`), already on disk; there is no DK-specific
ERA5 fetch.

## Notes

- **UTM → lon/lat.** The register gives ETRS89 / UTM 32N X/Y; the processor
  converts them. Turbines with missing coordinates or capacity are dropped.
- **Coordinates are per-turbine and exact**: Denmark needs no GWPT join, the
  reason it is the cleanest region in the set.
- **Licence.** Danish Energy Agency open public data; free to use with
  attribution.
