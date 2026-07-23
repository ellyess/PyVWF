#!/usr/bin/env python
"""Build the CAMMESAArgentinaSource inputs from the monthly GWh + a GWPT join.

Reads (all local): the per-plant monthly wind energy CSV written by
`scripts/fetch/cammesa_ar.py` (input/raw/cammesa/ar_wind_monthly.csv:
ID, region, provincia, year, month, gwh), and the Global Wind Power Tracker
workbook (coordinates AND capacity — CAMMESA carries neither).

Writes (under <out>, default input/observations/turbine/AR/):
    ar_obs.csv        monthly CF wide frame (commissioning prefix stripped)
    ar_md.csv         CAMMESAArgentinaSource metadata (coords+capacity joined)
    join_report.md    match evidence: every plant, GWPT match, capacity, CF
    ar_join_residual.csv   plants NOT matched OR capacity-suspect, to curate

Coordinate + CAPACITY join. CAMMESA has no capacity, so the CF denominator is
external: each central is matched to a GWPT operating Argentine farm by
normalised name for both lon/lat and capacity_mw. Because capacity is the
*output* of the join it cannot confirm the match, so a second guard runs on
the result: a plant whose median monthly CF exceeds CAP_SUSPECT_CF has almost
certainly matched a too-small capacity (real AR wind tops out near 0.5-0.6),
and is written to the residual for re-curation. Overrides
(configs/curation/ar_coord_overrides.csv: ID,lon,lat,capacity_mw) win over the auto
join; build_ar_metadata then fails loudly on any plant lacking a coordinate or
a capacity.

    python scripts/process/cammesa_ar.py
    python scripts/process/cammesa_ar.py --years 2021 2024
"""
import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

from vwf.datasets.cammesa_ar import (
    build_ar_metadata,
    capacity_suspect_ids,
    monthly_cf_from_gwh,
    strip_commissioning_prefix,
)

#: Plants dropped from the fleet, with the reason. Mostly self-generation
#: autoproducers (behind-the-meter wind at cement works, oil fields, an
#: aluminium smelter) that are absent from a confident GWPT match, so no
#: coordinate or capacity can be sourced without fabricating one; plus
#: "Fin del Mundo" which is in Tierra del Fuego, south of the -49 ERA5 box.
#: The sizeable ones (Casa YPF Luz / Cañadón León ~123 MW Santa Cruz, ALUAR
#: ~50 MW, La Elbita ~50 MW) are worth reinstating via the override table if
#: their coordinates and capacity can be verified.
EXCLUDE: tuple[str, ...] = (
    "AG Cementos Avellaneda-Olav.",           # cement-works self-gen, Olavarría
    "EL TORDILLO",                            # YPF oil-field self-gen, Chubut
    "EOLICO EL JUME",                         # small, Santiago del Estero
    "L.BLANC 4 ENARS",                        # Loma Blanca IV / ENARSA, Trelew — no confident GWPT cap
    "P.E. LA ELBITA",                         # Buenos Aires, absent from GWPT match
    "P.EOLICO CASA YPF LUZ",                  # Cañadón León (YPF Luz), Santa Cruz — reinstate if verifiable
    "P.EOLICO VIENTOS LA RINCONADA",          # Buenos Aires, no confident match
    "P.EOLICO VIENTOS OLAVARRIA",             # Olavarría self-gen, ambiguous vs Ternium
    "Parque eólico autogeneración ALUAR",     # ALUAR smelter self-gen, Puerto Madryn
    "Parques Eólicos del Fin del Mundo SA",   # Tierra del Fuego — SOUTH of the ERA5 box
)

_DROP = {"PARQUE", "EOLICO", "EOLICA", "PE", "WIND", "FARM", "DEL", "DE",
         "LA", "LOS", "LAS", "EL", "GENNEIA", "SA", "S", "P", "AG"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9 ]", " ", s.upper())
    toks = [t for t in s.split() if t not in _DROP and not re.fullmatch(r"I{1,3}V?|IV", t)]
    return " ".join(toks).strip()


def gwpt_argentina(xlsx: Path) -> pd.DataFrame:
    g = pd.read_excel(xlsx, sheet_name="Data")
    g = g[(g["Country/Area"].astype(str).str.strip() == "Argentina")
          & (g["Status"].astype(str).str.lower() == "operating")].copy()
    g["norm"] = g["Project Name"].map(norm)
    g["cap"] = pd.to_numeric(g["Capacity (MW)"], errors="coerce")
    return g[["Project Name", "norm", "cap", "Latitude", "Longitude"]]


def join_coords_caps(fleet: pd.DataFrame, g: pd.DataFrame, overrides: pd.DataFrame):
    """Return join_df[ID,lon,lat,capacity_mw,gwpt_name] and unmatched IDs."""
    ov = ({str(i): (lo, la, c) for i, lo, la, c in zip(
        overrides["ID"].astype(str), overrides["lon"].astype(float),
        overrides["lat"].astype(float), overrides["capacity_mw"].astype(float))}
        if len(overrides) else {})
    rows, unmatched = [], []
    for f in fleet.itertuples():
        fid, nm = str(f.ID), norm(f.site_name)
        if fid in ov:
            lo, la, c = ov[fid]
            rows.append((fid, lo, la, c, "override"))
            continue
        cand = g[g["norm"] == nm]
        if not len(cand):
            cand = g[g["norm"].apply(lambda x: bool(x) and (x in nm or nm in x))]
        if len(cand):
            best = cand.nlargest(1, "cap").iloc[0]  # phase-split: take full-farm cap
            rows.append((fid, best["Longitude"], best["Latitude"], best["cap"],
                         best["Project Name"]))
        else:
            unmatched.append(fid)
    join = pd.DataFrame(rows, columns=["ID", "lon", "lat", "capacity_mw", "gwpt_name"])
    return join, unmatched


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--monthly", default="input/raw/cammesa/ar_wind_monthly.csv")
    ap.add_argument("--gwpt", default="input/reference/gwpt/Global-Wind-Power-Tracker-February-2026.xlsx")
    ap.add_argument("--overrides", default="configs/curation/ar_coord_overrides.csv")
    ap.add_argument("--years", type=int, nargs=2, default=[2021, 2024],
                    metavar=("START", "END"))
    ap.add_argument("--out", default="input/observations/turbine/AR")
    ap.add_argument("--height", type=float, default=100.0)
    ap.add_argument("--model", default="2019COE_Market_Average_2.6MW_121")
    args = ap.parse_args()

    mp = Path(args.monthly)
    if not mp.is_file():
        sys.exit(f"{mp} not found — run scripts/fetch/cammesa_ar.py first.")
    gwh = pd.read_csv(mp)
    gwh["ID"] = gwh["ID"].astype(str)
    fleet = (gwh.groupby("ID").agg(site_name=("ID", "first"),
             region=("region", "first"), provincia=("provincia", "first"))
             .reset_index())

    g = gwpt_argentina(Path(args.gwpt))
    ov_path = Path(args.overrides)
    overrides = pd.read_csv(ov_path) if ov_path.is_file() else \
        pd.DataFrame(columns=["ID", "lon", "lat", "capacity_mw"])
    join, unmatched = join_coords_caps(fleet, g, overrides)

    y0, y1 = args.years
    obs = strip_commissioning_prefix(monthly_cf_from_gwh(gwh, join, y0, y1))
    suspect = set(capacity_suspect_ids(obs)) - set(overrides["ID"].astype(str))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    obs.to_csv(out / "ar_obs.csv", index=False)

    residual = pd.concat([
        fleet[fleet["ID"].isin(unmatched)].assign(reason="no GWPT match"),
        fleet[fleet["ID"].isin(suspect)].assign(reason="capacity suspect (median CF > 0.65)"),
    ], ignore_index=True)
    residual.to_csv(out / "ar_join_residual.csv", index=False)

    print(f"CAMMESA wind plants: {len(fleet)} | joined: {len(join)} | "
          f"unmatched: {len(unmatched)} | capacity-suspect: {len(suspect)}")
    if unmatched or suspect:
        print(f"  -> curate configs/curation/ar_coord_overrides.csv (ID,lon,lat,capacity_mw) "
              f"from {out/'ar_join_residual.csv'} then re-run.")

    # Drop capacity-suspect from the join so they fail loudly (not trusted).
    join_ok = join[~join["ID"].isin(suspect)]
    try:
        md = build_ar_metadata(fleet, join_ok, height=args.height,
                               model=args.model, exclude=EXCLUDE)
    except ValueError as exc:
        (out / "ar_md.csv").unlink(missing_ok=True)
        print(f"\nmetadata NOT written: {exc}", file=sys.stderr)
        sys.exit(2)

    md.to_csv(out / "ar_md.csv", index=False)
    lines = [
        "# Argentina (CAMMESA) join report", "",
        f"- wind plants: {len(fleet)}; metadata rows: {len(md)}; "
        f"capacity: {md['capacity'].sum()/1e6:.2f} GW",
        f"- lat {md['lat'].min():.1f}..{md['lat'].max():.1f}; "
        f"Patagonia (Chubut+Santa Cruz): "
        f"{int(md['provincia'].isin(['CHUBUT','SANTA CRUZ']).sum())} plants",
        "- height/model uniform defaults; coords AND capacity from GWPT.", "",
        "| ID | region | cap MW | GWPT match |", "|---|---|---|---|",
    ]
    cap_kw = dict(zip(md["ID"], md["capacity"]))
    for r in join_ok.itertuples():
        if str(r.ID) in set(md["ID"]):
            reg = fleet.loc[fleet["ID"] == r.ID, "provincia"].iloc[0]
            lines.append(f"| {r.ID} | {reg} | {cap_kw.get(r.ID,0)/1000:.0f} | {r.gwpt_name} |")
    (out / "join_report.md").write_text("\n".join(lines))
    print(f"metadata: {len(md)} plants -> {out/'ar_md.csv'}")
    print(f"observations -> {out/'ar_obs.csv'} | join report -> {out/'join_report.md'}")


if __name__ == "__main__":
    main()
