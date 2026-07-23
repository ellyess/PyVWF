#!/usr/bin/env python
"""Build the CENChileSource inputs from the CEN downloads + a GWPT coord join.

Reads (all local): the per-month CEN wind generation JSON written by
`scripts/fetch/cen_cl.py --years` under input/cen_raw/, and the Global Wind
Power Tracker workbook (coordinates; CEN exposes none).

Writes (under <out>, default input/turbine_level_data/CL/):
    cl_obs.csv        monthly CF wide frame (UTC bins, coverage-screened)
    cl_md.csv         CENChileSource metadata contract (needs coords joined)
    join_report.md    match evidence: every plant, its GWPT match, both caps
    cl_coord_residual.csv   plants NOT auto-matched, with candidates to curate

Coordinate join. CEN carries no lon/lat, so plants are matched to GWPT
operating Chilean farms by normalised name, confirmed by capacity. CEN splits
a few GWPT farms into co-located phases (Malleco Norte/Sur, Renaico I/II,
Cabo Leones II/III, Horizonte Norte/Sur) — a GWPT name that is a substring of
the CEN name, with capacities within tolerance, is accepted and both phases
inherit the parent coordinate (correct at 0.25 deg ERA5 resolution). Anything
not confidently matched is written to cl_coord_residual.csv for hand curation
into configs/curation/cl_coord_overrides.csv (ID,lon,lat); build_cl_metadata then fails
loudly on any plant still lacking a coordinate.

    python scripts/process/cen_cl.py
    python scripts/process/cen_cl.py --years 2021 2024
"""
import argparse
import glob
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

from vwf.datasets.cen_cl import (
    build_cl_metadata,
    monthly_cf_from_generation,
    strip_commissioning_prefix,
    wind_fleet_from_generation,
)

CAP_TOL = 0.35  # fractional capacity agreement required to confirm a name match

#: Plants dropped from the fleet, with the reason. The four tiny PMGD plants
#: (5-9 MW each, ~30 MW total = <1% of the ~3.4 GW fleet) are real and report
#: generation, but are absent from GWPT and the GWPT Below-Threshold sheet, so
#: no coordinate can be sourced without fabricating one. Excluded rather than
#: mis-located; add a row to configs/curation/cl_coord_overrides.csv to reinstate any
#: whose location you can verify. (Magallanes / isolated systems south of -44
#: would also live here; none are in the SEN wind fleet today.)
EXCLUDE: tuple[str, ...] = ("334", "350", "414", "436")  # Raki, Huajache, Las Peñas, Lebu III

_DROP = {"PARQUE", "EOLICO", "EOLICA", "PMGD", "PE", "WIND", "FARM",
         "CHILE", "ENEL", "DEL", "DE", "LA", "LOS", "LAS", "EL"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9 ]", " ", s.upper())
    toks = [t for t in s.split() if t not in _DROP]
    return " ".join(toks).strip()


def load_generation(raw_dir: Path, y0: int, y1: int) -> pd.DataFrame:
    paths = sorted(glob.glob(str(raw_dir / "cen_gen_*.json")))
    paths = [p for p in paths if y0 <= int(Path(p).stem.split("_")[-2]) <= y1]
    if not paths:
        sys.exit(f"no cen_gen_*.json under {raw_dir} for {y0}-{y1} — run "
                 "scripts/fetch/cen_cl.py --years first.")
    return pd.concat([pd.DataFrame(json.load(open(p))) for p in paths],
                     ignore_index=True)


def gwpt_chile(xlsx: Path) -> pd.DataFrame:
    g = pd.read_excel(xlsx, sheet_name="Data")
    g = g[(g["Country/Area"].astype(str).str.strip() == "Chile")
          & (g["Status"].astype(str).str.lower() == "operating")].copy()
    g["norm"] = g["Project Name"].map(norm)
    g["cap"] = pd.to_numeric(g["Capacity (MW)"], errors="coerce")
    return g[["Project Name", "norm", "cap", "Latitude", "Longitude"]]


def match(fleet: pd.DataFrame, g: pd.DataFrame, overrides: pd.DataFrame):
    """Return (coords_df[ID,lon,lat,gwpt_name], residual_df)."""
    # Explicit column access, NOT itertuples: with extra provenance columns
    # present, itertuples attribute access shifted r.ID onto the lon value and
    # silently produced longitude-keyed entries that matched no plant. Only
    # ID/lon/lat are used from `overrides`.
    ov = (dict(zip(overrides["ID"].astype(str),
                   zip(overrides["lon"].astype(float), overrides["lat"].astype(float))))
          if len(overrides) else {})
    rows, residual = [], []
    for f in fleet.itertuples():
        fid, cap, nm = str(f.ID), f.capacity_mw, norm(f.site_name)
        if fid in ov:
            rows.append((fid, ov[fid][0], ov[fid][1], "override"))
            continue
        cand = g[g["norm"] == nm]
        if not len(cand):  # substring either direction (phase splits)
            cand = g[g["norm"].apply(lambda x: bool(x) and (x in nm or nm in x))]
        if len(cand) and pd.notna(cap):
            cand = cand.assign(dcap=(cand["cap"] - cap).abs() / cap)
            best = cand.nsmallest(1, "dcap").iloc[0]
            if best["dcap"] <= CAP_TOL:
                rows.append((fid, best["Longitude"], best["Latitude"],
                             best["Project Name"]))
                continue
        top = g.assign(dcap=(g["cap"] - cap).abs()).nsmallest(3, "dcap") \
            if pd.notna(cap) else g.head(3)
        residual.append({
            "ID": fid, "site_name": f.site_name, "capacity_mw": cap,
            "candidates": "; ".join(
                f"{r['Project Name']} ({r['cap']}MW @{r['Latitude']:.3f},{r['Longitude']:.3f})"
                for _, r in top.iterrows()),
        })
    coords = pd.DataFrame(rows, columns=["ID", "lon", "lat", "gwpt_name"])
    return coords, pd.DataFrame(residual)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="input/cen_raw")
    ap.add_argument("--gwpt", default="input/Global-Wind-Power-Tracker-February-2026.xlsx")
    ap.add_argument("--overrides", default="configs/curation/cl_coord_overrides.csv")
    ap.add_argument("--years", type=int, nargs=2, default=[2021, 2024],
                    metavar=("START", "END"))
    ap.add_argument("--out", default="input/turbine_level_data/CL")
    ap.add_argument("--height", type=float, default=100.0,
                    help="Uniform hub-height default, m (CEN has no hub height)")
    ap.add_argument("--model", default="2019COE_Market_Average_2.6MW_121",
                    help="Uniform power-curve key (a column of power_curves.csv)")
    args = ap.parse_args()

    y0, y1 = args.years
    gen = load_generation(Path(args.raw), y0, y1)
    fleet = wind_fleet_from_generation(gen)
    obs = strip_commissioning_prefix(monthly_cf_from_generation(gen, y0, y1))

    g = gwpt_chile(Path(args.gwpt))
    ov_path = Path(args.overrides)
    overrides = pd.read_csv(ov_path) if ov_path.is_file() else pd.DataFrame(columns=["ID", "lon", "lat"])
    coords, residual = match(fleet, g, overrides)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    obs.to_csv(out / "cl_obs.csv", index=False)
    residual.to_csv(out / "cl_coord_residual.csv", index=False)

    print(f"CEN wind plants: {len(fleet)} | auto/override-matched: {len(coords)} "
          f"| residual (need curation): {len(residual)}")
    if len(residual):
        print(f"  -> fill configs/curation/cl_coord_overrides.csv (ID,lon,lat) from "
              f"{out/'cl_coord_residual.csv'} then re-run.")

    try:
        md = build_cl_metadata(fleet, coords.rename(columns={}), height=args.height,
                               model=args.model, exclude=EXCLUDE)
    except ValueError as exc:
        (out / "cl_md.csv").unlink(missing_ok=True)
        print(f"\nmetadata NOT written: {exc}", file=sys.stderr)
        sys.exit(2)

    md.to_csv(out / "cl_md.csv", index=False)
    lines = [
        "# Chile (CEN) join report", "",
        f"- wind plants: {len(fleet)}; matched to GWPT: {len(coords)}; "
        f"metadata rows: {len(md)}",
        f"- capacity: {md['capacity'].sum()/1e6:.2f} GW; "
        f"lat {md['lat'].min():.1f}..{md['lat'].max():.1f}",
        "- height/model are uniform defaults (CEN has neither); coords from GWPT.",
        "", "## matches", "",
        "| ID | CEN plant | cap MW | GWPT match |", "|---|---|---|---|",
    ]
    cap_by_id = dict(zip(fleet["ID"].astype(str), fleet["capacity_mw"]))
    for r in coords.itertuples():
        lines.append(f"| {r.ID} | {fleet.loc[fleet['ID'].astype(str)==r.ID,'site_name'].iloc[0]} "
                     f"| {cap_by_id.get(r.ID,'')} | {r.gwpt_name} |")
    (out / "join_report.md").write_text("\n".join(lines))
    print(f"metadata: {len(md)} plants -> {out/'cl_md.csv'}")
    print(f"observations -> {out/'cl_obs.csv'} | join report -> {out/'join_report.md'}")


if __name__ == "__main__":
    main()
