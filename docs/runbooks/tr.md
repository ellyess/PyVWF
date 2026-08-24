# Turkey (EPİAŞ/EXIST): evaluated, not shipped

**Status (2026-07-22): NOT confirmed tier-1. The survey's "per-plant hourly
via free API" claim does not hold on a Data Consultation subscription.**
Authentication works and the 2,584-plant list downloads, but no reachable
endpoint returns per-plant generation:

| Endpoint | Result |
| --- | --- |
| `/generation/data/injection-quantity-powerplant-list` | **200 (GET)**: 2,584 plants with `id`, `eic`, `name`, `shortName` |
| `/generation/data/injection-quantity` (UEVM) | 200, but returns the **national fuel-mix series** and **silently ignores `powerPlantId`** |
| `/generation/data/realtime-generation` | 200 without a plant filter (national mix); **400** when `powerPlantId` is in the body |
| `/renewables/data/licensed-realtime-generation` | **403: not covered by this subscription.** This is the endpoint most likely to hold per-plant licensed wind output |
| `/generation/data/realtime-generation-bulk` | 400 (body shape unresolved) |

**How the false positive was caught, and why it matters.** The UEVM endpoint
answers `200` with a plausible 24 rows for any `powerPlantId`, including
invented ones; it reads exactly like per-plant hourly data. The
must-distinguish test in `scripts/fetch/epias_tr.py --probe` requests two
different wind plants and compares the numbers: they are byte-identical, so
the filter is ignored. Without that comparison this would have been recorded
as "Turkey verified, 2016→ history", which is false. The probe now prints the
verdict every run so the mistake cannot be re-made.

**What would unblock it:** subscribe to the business unit covering
`renewables-service` (licensed real-time generation) on the EPİAŞ platform (a
user action), then re-run `--probe`. If that 403 becomes a 200 with
per-plant-varying numbers, Turkey returns to tier-1 candidacy with its TÜREB
hub-height advantage intact.

---

**Original acquisition guide below (adapter not written).** Turkey is the #3
candidate region in the July-2026 dataset survey
(`docs/findings/dataset-survey.md`) and the only candidate with BOTH
per-plant hourly generation and a national turbine register (TÜREB) carrying
turbine models, i.e. hub heights become derivable per plant, which no other
non-European candidate offers except Canada. Climate: Mediterranean Aegean/
Marmara (Csa/Csb) + cold semi-arid Anatolia (BSk). ~380 wind plants, ~16 GW.

One open question decides whether Turkey is a full tier-1 region: **per-plant
history depth**. The transparency platform launched 2016 but per-plant series
start dates were not verified in the survey. The first API call answers it.

## 1. The step only you can do: register

EPİAŞ Transparency Platform requires free self-service registration:

1. Register at https://seffaflik.epias.com.tr (kayıt/register), email-based.
2. Store the credentials outside the repo (e.g. environment variables
   `EPIAS_USERNAME` / `EPIAS_PASSWORD`); never commit them.

## 2. What to get

| Data | Source | Notes |
| --- | --- | --- |
| Per-plant hourly injection ("Injection Quantity", UEVM) | Transparency 2.0 API, https://seffaflik.epias.com.tr/electricity-service/technical/tr/index.html | Ex-post settlement-grade metered injection per UEVCB (settlement unit) |
| Real-time generation per plant | Same API ("Real-Time Generation", santral bazlı) | Operational, not settlement-grade; prefer UEVM for CF |
| Plant list / IDs | API powerplant-list endpoints | Maps plant IDs → names for the TÜREB join |
| Turbine models + capacity per plant | TÜREB Turkish Wind Energy Statistics Reports (tureb.com.tr, ~6-monthly PDFs) | Manufacturer/model per plant → hub height via model lookup (partially); coordinates via GWPT |
| API wrapper | `eptr2` (PyPI, https://github.com/Tideseed/eptr2) | Mature wrapper over >110 transparency services incl. per-plant hourly generation |

## 3. The decisive first check

With credentials in place (`pip install eptr2`):

```python
from eptr2 import EPTR2
eptr = EPTR2()  # reads credentials from env
# pick one long-lived plant (e.g. an Aegean plant commissioned pre-2016),
# request its UEVM hourly series for 2017-01; if data comes back, Turkey
# has 8+ years of per-plant history and is a full tier-1 region.
```

Record the answer (earliest year with per-plant UEVM data, and for how many
plants) in the dataset survey doc before any adapter work.

## 4. Verification checklist before writing the adapter

1. UEVM units (MWh vs kWh) and timezone. Turkey abolished DST in 2016
   (permanent UTC+3), which makes the UTC conversion a fixed shift, but
   confirm the API labels.
2. Plant-vs-UEVCB granularity: one plant can have several settlement units;
   decide the aggregation unit (parallel to the BR conjunto decision).
3. TÜREB report → plant list match rate, and hub-height coverage after the
   model lookup (record `height_source` per plant, as US/BR do).
4. Redistribution terms: transparency data is public-by-regulation, but check
   the EPİAŞ terms of use for republication of derived tables before any
   committed artefact carries their numbers.
5. Curtailment: check whether the transparency platform exposes wind
   curtailment/redispatch series for a BR-style mask.

## 5. Expected shape of the region

- Adapter: `epias-tr`, `obs_level = "turbine"`, `obs_unit = "plant"`.
- ERA5 box: roughly lon [25, 45], lat [35, 43].
- Seasons: Northern Hemisphere explicit month lists (copy `dk.toml`).
- The fleet is heavily Aegean/Marmara; expect strong within-region climate
  contrast with Anatolian BSk plants, which is exactly the regime-coverage
  property the ML transfer re-test asks for
  (`docs/findings/method-ml-transfer.md`).
