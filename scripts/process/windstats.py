#!/usr/bin/env python
"""Build a WindStats region's inputs from the CONFIDENTIAL extract + GWPT coords.

⚠ CONFIDENTIAL / COMMERCIAL (WindStats) generation, with open GWPT coordinates
for Spain (mixed licence). Nothing it writes may be committed or redistributed;
`input/` is git-ignored, and this only reshapes files the user already holds.

    python scripts/process/windstats.py --country ES --src "<WindStats folder>"

Reads ``<CC>_md.csv``, ``<CC>_data.csv``, ``geolocate.<country>.csv`` from
``--src`` and the Global Wind Power Tracker, and writes to
``input/observations/turbine/<CC>/``:
    <cc>_md.csv     per-turbine metadata with coordinates
    <cc>_obs.csv    monthly CF, wide
    <cc>_join_report.md   coordinate match rate + excluded farms

Coordinates come from matching each WindStats id -> its thewindpower name
(``geolocate``) -> a GWPT project name (fuzzy). Turbines whose farm does not
match GWPT are dropped and listed. Spain's WindStats extract is 1998-2000, so
the default window is historical.
"""
import argparse
from pathlib import Path

import pandas as pd

from vwf.datasets.windstats import (
    build_windstats_metadata,
    match_twp_to_coords,
    windstats_metadata,
    windstats_monthly_cf,
)

GEO_NAME = {"ES": "spain", "SE": "sweden", "FI": "finland"}
GWPT_COUNTRY = {"ES": "Spain", "SE": "Sweden", "FI": "Finland"}
DEFAULT_WINDOW = {"ES": (1998, 2000)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--country", required=True, choices=["ES", "SE", "FI"])
    ap.add_argument("--src", required=True, help="CONFIDENTIAL WindStats folder")
    ap.add_argument("--gwpt", default="input/reference/gwpt/Global-Wind-Power-Tracker-February-2026.xlsx")
    ap.add_argument("--years", type=int, nargs=2, default=None, metavar=("START", "END"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--height", type=float, default=80.0)
    args = ap.parse_args()

    cc = args.country
    src = Path(args.src)
    print("=" * 70)
    print(f"⚠  CONFIDENTIAL WindStats ({cc}) generation + open GWPT coords "
          "(mixed licence).")
    print("   Output is NOT redistributable; input/ is git-ignored.")
    print("=" * 70)

    md_raw = pd.read_csv(src / f"{cc}_md.csv", encoding="latin-1")
    data = pd.read_csv(src / f"{cc}_data.csv")
    geo = pd.read_csv(src / f"geolocate.{GEO_NAME[cc]}.csv", encoding="latin-1")
    g = pd.read_excel(args.gwpt, sheet_name="Data")
    gsub = g[(g["Country/Area"].astype(str).str.strip() == GWPT_COUNTRY[cc])
             & (g["Status"].astype(str).str.lower() == "operating")]

    smd = windstats_metadata(md_raw, cc)
    coords = match_twp_to_coords(geo, gsub)
    matched = smd["link"].isin(set(coords["ws"]))
    unmatched_links = tuple(smd.loc[~matched, "link"].unique())
    print(f"turbines: {len(smd)} | with a GWPT coordinate: {matched.sum()} "
          f"({matched.mean():.0%}) | farms unmatched: {len(unmatched_links)}")

    fmd = build_windstats_metadata(smd, coords, height_default=args.height,
                                   exclude=unmatched_links)
    y0, y1 = args.years or DEFAULT_WINDOW.get(cc, (1998, 2013))
    obs = windstats_monthly_cf(data, smd[["ID", "capacity"]], y0, y1)
    obs = obs[obs["ID"].isin(set(fmd["ID"]))].reset_index(drop=True)

    out = Path(args.out_dir) if args.out_dir else Path("input/observations/turbine") / cc
    out.mkdir(parents=True, exist_ok=True)
    fmd.to_csv(out / f"{cc.lower()}_md.csv", index=False)
    obs.to_csv(out / f"{cc.lower()}_obs.csv", index=False)
    lines = [
        f"# {GWPT_COUNTRY[cc]} (WindStats) coordinate-join report", "",
        "⚠ Generation is CONFIDENTIAL WindStats; coordinates are open GWPT.",
        f"- turbines with a coordinate: {len(fmd)} of {len(smd)} "
        f"({len(fmd)/len(smd):.0%})",
        f"- capacity: {fmd['capacity'].sum()/1e6:.2f} GW; "
        f"lat {fmd['lat'].min():.1f}..{fmd['lat'].max():.1f}",
        f"- coordinate source: GWPT (open); real WindStats hub heights on "
        f"{(fmd['height_source']=='windstats').sum()} turbines",
        f"- data window: {y0}-{y1}; obs plant-years: {len(obs)}",
        f"- farms with NO GWPT match (excluded): {len(unmatched_links)}",
    ]
    (out / f"{cc.lower()}_join_report.md").write_text("\n".join(lines))
    print(f"metadata: {len(fmd)} turbines -> {out / f'{cc.lower()}_md.csv'}")
    print(f"observations: {len(obs)} plant-years -> {out / f'{cc.lower()}_obs.csv'}")
    print(f"join report -> {out / f'{cc.lower()}_join_report.md'}")
    if cc in ("SE", "FI"):
        print("\nNOTE: GWPT under-covers SE/FI (7-9% match). Supply a "
              "thewindpower coordinate table for a usable region.")


if __name__ == "__main__":
    main()
