# United Kingdom (Ofgem ROC + REPD) region — acquisition runbook

**Read this before assuming the UK data is reproducible like Denmark's — it is
only partly so.** The committed UK inputs (`uk_md.csv`, `ukobs.csv`) came from
sources that are gated (Ofgem per-station certificates) or not openly
redistributable (turbine-spec metadata). This runbook documents what CAN be
scripted from open data, exactly where the gaps are, and how the pieces map to
the committed files. The UK keeps its shared `european-turbine` adapter
(DK/DE/UK); these scripts produce its inputs.

## The two source components (per the co-authored UK methodology)

| Component | Committed file | Open + scriptable? |
| --- | --- | --- |
| **Observations** — Ofgem Renewables Obligation certificates → energy | `ukobs.csv` | **No, not end-to-end.** Per-station monthly ROC issuance is a gated Ofgem RER export; the login-free bulk file is technology-aggregate only. The banding conversion and pseudo-replication ARE scripted; the input export is manual. |
| **Metadata** — location, capacity, turbine specs | `uk_md.csv` | **Partly.** REPD (open) gives location, capacity, turbine count. It has NO turbine model, NO rotor diameter, and hub height for only ~13% of sites. So the open reconstruction is a location/capacity table with uniform-default specs — materially thinner than the curated file. |

The paper's *second* observed source, **Elexon settlement data** (half-hourly
metered volumes via the Insights Solution / BMRS), is not what the committed
`ukobs.csv` holds (those are ROC-derived, keyed by accreditation number); it is
not scripted here.

## 1. Metadata — REPD (fully scriptable, open)

```bash
python scripts/fetch/uk.py             # scrapes gov.uk, downloads the current REPD CSV
python scripts/process/uk.py metadata  # -> uk_md_open.csv + uk_md_open_divergence.md
```

`fetch/uk.py` scrapes the [REPD publication page](https://www.gov.uk/government/publications/renewable-energy-planning-database-quarterly-extract)
for the current quarterly CSV (the direct URL rotates each quarter; OGL
licence). `process uk.py metadata` reconstructs a per-turbine table:
OSGB X/Y **reprojected** to lon/lat, installed capacity, turbine count, REPD
tip height where present else a uniform default, and a **uniform model/rotor**
default (REPD carries neither) — the same approach the AU/BR/CL/AR adapters use
for metadata-poor regions.

**It does NOT overwrite the curated `uk_md.csv`.** The open table ships as
`uk_md_open.csv` with a divergence report. On the current data the open table
is the *whole current operational fleet* (826 stations, ~30 GW, median tip
height 100 m from defaults) versus the curated *2015-2019 RO-accredited set*
(360 stations, ~16 GW, real median hub height 70 m). REPD has no ROC
accreditation key and no historical snapshot, so the fleets are not the same
population — treat `uk_md_open.csv` as an open reference, and keep the curated
`uk_md.csv` for reproducing published DK/DE/UK results.

## 2. Observations — Ofgem ROC (manual export, then scripted)

Per-station monthly ROC issuance is only in Ofgem's Renewable Electricity
Register, behind a gated public-report export (`fetch/uk.py` prints the steps):

1. Go to <https://rer.ofgem.gov.uk/> (View Public Reports — no login), or email
   `renewable.enquiry@ofgem.gov.uk` for the current access link.
2. Export the ROCs-issued report for the Renewables Obligation, filtered to
   wind and your output-period window. Save as
   `input/ofgem_raw/roc_issuance.xlsx`.
3. Process it:

```bash
python scripts/process/uk.py observations --roc input/ofgem_raw/roc_issuance.xlsx
```

This converts **ROCs → MWh** with the grandfathered banding (Ofgem RO Guidance,
Appendix 4: onshore wind 0.9 ROC/MWh for 2013/14+ vintages, 1.0 before; offshore
2.0 / 1.9 / 1.8 by vintage — decoded from each station's accreditation-number
technology code when a technology column is absent), then **equal-splits** each
station's monthly energy across its turbines (from the curated `uk_md.csv`
station→turbine counts) into the `ukobs.csv` schema. Output is `ukobs_open.csv`
(not overwriting the committed file). Verified to reproduce the committed
arithmetic exactly: a 3-turbine station's 1709 MWh January becomes three
identical 569,666.7 kWh turbine rows.

Caveats: the RER export's exact column names are unverified (the register is
gated), so the processor matches columns tolerantly and accepts
`--station-col/--period-col/--certs-col`; and RER retains only ~7 years of
issuance, so the full 2015-2019 window may need archived reports.

## Bottom line

- `uk_md_open.csv` + divergence report: **reproducible today** from open REPD,
  but a location/capacity table, not the curated spec-complete one.
- `ukobs_open.csv`: **reproducible once you run the one manual Ofgem RER
  export**; the banding + pseudo-replication are scripted and verified.
- The committed `uk_md.csv` / `ukobs.csv` remain the inputs for published
  results; these scripts make the open half reproducible and document the rest.
