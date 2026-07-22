#!/usr/bin/env python
"""Fetch and reshape the CAMMESA renewables database (Argentina).

USER-EXECUTED, but unusually cheap: the whole history is one ~2 MB ZIP with
no credentials, no API, no rate limit.

    python scripts/fetch_cammesa_ar.py            # download + reshape
    python scripts/fetch_cammesa_ar.py --probe    # report coverage, write nothing

Verified against the live file on 2026-07-23 (the 2026-06 edition):

- Source: https://cammesaweb.cammesa.com/erenovables/?wpdmdl=37500
  returns `Energía Renovables - Base de Datos <YYYY-MM>.xlsx` inside a ZIP.
- The useful sheet is **"Tabla Resumen x Central"**, whose header sits on
  **row 4** (0-indexed), not row 0: `FUENTE DE ENERGÍA`, `REGIÓN`,
  `PROVINCIA`, `CENTRAL DESCRIPCIÓN`, then one column per month.
- Coverage: **186 monthly columns, 2011-01 through 2026-06**, values in GWh.
- Wind (`FUENTE DE ENERGÍA == "EOLICO"`): **75 distinct centrales**, 4
  reporting in 2011 rising to 70 in 2025 (18,628 GWh).
- Region/province cells are **merged in Excel**, so they arrive blank on
  continuation rows and must be forward-filled or plants inherit the wrong
  province.

Monthly GWh per plant is *exactly* PyVWF's native training resolution — no
half-hourly reshaping, no timezone conversion, no DST edge cases. What is
missing is coordinates, capacity, and hub heights: CAMMESA carries none, so
those come from the Global Wind Power Tracker (CC-BY-4.0, already on disk).

Argentina's regime value is Patagonia: Chubut (23 plants) and Santa Cruz (5)
sit in the cold-steppe westerlies at capacity factors near 50%, which nothing
else in the validation set covers.

Output: <input-root>/cammesa_raw/ (the ZIP and the extracted workbook) and a
tidy long CSV `ar_wind_monthly.csv` with columns ID, year, month, gwh.
"""
import argparse
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://cammesaweb.cammesa.com/erenovables/?wpdmdl=37500"
SHEET = "Tabla Resumen x Central"
HEADER_ROW = 4          # verified; the sheet has a title block above it
WIND = "EOLICO"


def download(dest_dir: Path) -> bytes:
    dest_dir.mkdir(parents=True, exist_ok=True)
    cached = dest_dir / "cammesa_erenovables.zip"
    if cached.is_file():
        print(f"using cached {cached}")
        return cached.read_bytes()
    print(f"downloading {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        blob = resp.read()
    cached.write_bytes(blob)
    print(f"  {len(blob) / 1e6:.1f} MB -> {cached}")
    return blob


def load_wind(blob: bytes) -> tuple[pd.DataFrame, list[str]]:
    """Wind rows of the per-central sheet, with merged cells forward-filled."""
    z = zipfile.ZipFile(io.BytesIO(blob))
    inner = z.namelist()[0]
    with z.open(inner) as fh:
        df = pd.read_excel(fh, sheet_name=SHEET, header=HEADER_ROW)
    df.columns = [str(c) for c in df.columns]
    keys = df.columns[:4].tolist()
    # REGIÓN/PROVINCIA are merged cells: blank on continuation rows.
    for col in keys[:3]:
        df[col] = df[col].ffill()
    df = df.rename(columns=dict(zip(keys, ["fuente", "region", "provincia", "central"])))
    df["fuente"] = df["fuente"].astype(str).str.strip().str.upper()
    wind = df[df["fuente"].str.startswith(WIND)].copy()
    months = [c for c in df.columns if c[:4].isdigit()]
    return wind, months


def tidy(wind: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    long = wind.melt(id_vars=["central", "region", "provincia"],
                     value_vars=months, var_name="month_ts", value_name="gwh")
    long["gwh"] = pd.to_numeric(long["gwh"], errors="coerce")
    ts = pd.to_datetime(long["month_ts"])
    long["year"], long["month"] = ts.dt.year, ts.dt.month
    long = long.rename(columns={"central": "ID"})
    long["ID"] = long["ID"].astype(str).str.strip()
    return long[["ID", "region", "provincia", "year", "month", "gwh"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="store_true",
                    help="Report coverage and exit without writing the CSV")
    ap.add_argument("--out", default=None, help="Output directory")
    args = ap.parse_args()

    root = Path(os.environ.get("PYVWF_INPUT", "input"))
    out = Path(args.out) if args.out else root / "cammesa_raw"

    wind, months = load_wind(download(out))
    print(f"\nwind centrales: {wind['central'].nunique()}")
    print(f"month columns:  {len(months)}  ({months[0][:7]} .. {months[-1][:7]})")

    long = tidy(wind, months)
    reporting = long[long["gwh"] > 0]
    by_year = reporting.groupby("year")["ID"].nunique()
    print("\nplants reporting >0 GWh by year:")
    for year, n in by_year.items():
        if year % 2 == 1 or year >= 2023:
            print(f"  {year}: {n}")
    print("\ntop provinces by plant count:")
    for prov, n in wind["provincia"].value_counts().head(5).items():
        print(f"  {prov}: {n}")

    if args.probe:
        print("\n--probe: nothing written.")
        return
    dest = out / "ar_wind_monthly.csv"
    long.to_csv(dest, index=False)
    print(f"\n{len(long)} plant-months -> {dest}")
    print("Coordinates/capacity/hub heights are NOT in this file: join the "
          "Global Wind Power Tracker before building the region.")


if __name__ == "__main__":
    main()
