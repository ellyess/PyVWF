# Spain, historical (WindStats)

**Data source:** WindStats per-turbine monthly output, 1998-2000, with GWPT coordinates. **Adapter:** `windstats` · turbine-level · unit = turbine · mixed: confidential generation, open coordinates.

ES-WS, a legacy check rather than a modern benchmark, is coded so it does not
collide with the country-level `ES` ([ENTSO-E runbook](entsoe.md)). It has no
[scorecard](../scorecard.md) row; its config is `configs/regions/es_ws.toml`.

## Source and licence

**Mixed licence.** Generation is confidential, commercial WindStats data;
coordinates are the open Global Wind Power Tracker (CC-BY-4.0). Neither the
data nor the observation and metadata tables derived from them may be
committed or redistributed. `input/` is git-ignored, and the script below only
reshapes files you already hold.

## 1. Observations

```bash
python scripts/process/windstats.py --country ES --src "<WindStats folder>"
```

It reads `ES_md.csv`, `ES_data.csv` and `geolocate.spain.csv` from `--src`,
and the GWPT from `<input root>/reference/gwpt/` (`--gwpt` names another
file), and writes `es_md.csv`, `es_obs.csv` and `es_join_report.md` to
`<input root>/observations/turbine/ES/`. About 80% of turbines match a GWPT
project; the rest are dropped and listed in the join report.

**Sweden and Finland** share this pipeline, but the GWPT matches only 7-9% of
their farms. To add one, supply the thewindpower.net coordinate table the
`twp` names point at, add `SE-WS` or `FI-WS` to `countries` in
`src/pyvwf/sources/windstats.py`, and run
`scripts/process/windstats.py --country SE`. The extracts cover SE 1998-2013 and FI 2005-2012; the script
defaults to 1998-2013 outside Spain, so pass `--years 2005 2012` for FI.

## 2. ERA5

ES-WS needs its own 1998-2000 ERA5 over the mainland box in `es_ws.toml`, not
`era5/EU_2026-09`; it goes to `<input root>/era5/ES/` ([general fetch](../guides/data-sources.md#3-shared-inputs-all-regions)).

```bash
python scripts/fetch/era5.py --region es-ws
```

## 3. Train and evaluate

ES-WS runs on the default input root, `input/`, training on 1998-99 and
testing on 2000 with `cluster_list = [10]`:

```bash
PYVWF_INPUT=input pyvwf-validate train    --region configs/regions/es_ws.toml
PYVWF_INPUT=input pyvwf-validate evaluate --region configs/regions/es_ws.toml \
    --train-run output/validation/ES-WS/train-<stamp>
```

## Caveats

- **One curve for the whole fleet:** the open library's
  `2019COE_Market_Average_2.6MW_121` stands in for machines of about 600 kW.
- **Canary Islands turbines** (latitude below 30) fall outside the box and are excluded.
- **One test year,** 2000; the fleet predates the 2015-onward regions.
