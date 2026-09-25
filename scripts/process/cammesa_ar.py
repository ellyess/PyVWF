#!/usr/bin/env python
"""Build the CAMMESAArgentinaSource inputs from the monthly GWh + a GWPT join.

Reads (all local): the per-plant monthly wind energy CSV written by
`scripts/fetch/cammesa_ar.py` (input/raw/cammesa/ar_wind_monthly.csv:
ID, region, provincia, year, month, gwh), and the Global Wind Power Tracker
workbook (coordinates AND capacity; CAMMESA carries neither).

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
import sys
from pathlib import Path

import pandas as pd

from pyvwf.cli.common import add_input_path
from pyvwf.datasets.gwpt import load_gwpt, projects_with_keys
from pyvwf.datasets.cammesa_ar import (
    ar_plant_key,
    build_ar_metadata,
    capacity_suspect_ids,
    join_coords_caps,
    monthly_cf_from_gwh,
    strip_commissioning_prefix,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_input_path(ap, "--monthly", "raw", "cammesa", "ar_wind_monthly.csv")
    add_input_path(
        ap, "--gwpt", "reference", "gwpt", "Global-Wind-Power-Tracker-February-2026.xlsx"
    )
    ap.add_argument("--overrides", default="configs/curation/ar_coord_overrides.csv")
    ap.add_argument(
        "--exclusions",
        default="configs/curation/ar_fleet_exclusions.csv",
        help="Centrals dropped from the fleet, one row each with its reason: mostly "
        "self-generation with no confident GWPT match, and one south of the ERA5 box",
    )
    ap.add_argument("--years", type=int, nargs=2, default=[2021, 2024], metavar=("START", "END"))
    add_input_path(ap, "--out", "observations", "turbine", "AR")
    ap.add_argument("--height", type=float, default=100.0)
    ap.add_argument("--model", default="2019COE_Market_Average_2.6MW_121")
    args = ap.parse_args()

    mp = Path(args.monthly)
    if not mp.is_file():
        sys.exit(f"{mp} not found; run scripts/fetch/cammesa_ar.py first.")
    gwh = pd.read_csv(mp)
    gwh["ID"] = gwh["ID"].astype(str)
    fleet = (
        gwh.groupby("ID")
        .agg(
            site_name=("ID", "first"), region=("region", "first"), provincia=("provincia", "first")
        )
        .reset_index()
    )

    # The exclusions are a hard drop from the fleet. The self-generation autoproducers
    # were previously dropped only by failing the GWPT join, but that does not
    # remove a plant that DOES join (the Arauco / La Castellana codes join fine
    # yet are unrepresentative), so the drop is applied here to the fleet itself.
    exclude = tuple(pd.read_csv(args.exclusions, dtype=str)["ID"])
    fleet = fleet[~fleet["ID"].isin(exclude)].reset_index(drop=True)

    g = projects_with_keys(load_gwpt(Path(args.gwpt)), "Argentina", ar_plant_key)
    ov_path = Path(args.overrides)
    overrides = (
        pd.read_csv(ov_path)
        if ov_path.is_file()
        else pd.DataFrame(columns=["ID", "lon", "lat", "capacity_mw"])
    )
    join, unmatched = join_coords_caps(fleet, g, overrides)

    y0, y1 = args.years
    obs = strip_commissioning_prefix(monthly_cf_from_gwh(gwh, join, y0, y1))
    suspect = set(capacity_suspect_ids(obs)) - set(overrides["ID"].astype(str))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    obs.to_csv(out / "ar_obs.csv", index=False)

    residual = pd.concat(
        [
            fleet[fleet["ID"].isin(unmatched)].assign(reason="no GWPT match"),
            fleet[fleet["ID"].isin(suspect)].assign(reason="capacity suspect (median CF > 0.65)"),
        ],
        ignore_index=True,
    )
    residual.to_csv(out / "ar_join_residual.csv", index=False)

    print(
        f"CAMMESA wind plants: {len(fleet)} | joined: {len(join)} | "
        f"unmatched: {len(unmatched)} | capacity-suspect: {len(suspect)}"
    )
    if unmatched or suspect:
        print(
            f"  -> curate configs/curation/ar_coord_overrides.csv (ID,lon,lat,capacity_mw) "
            f"from {out / 'ar_join_residual.csv'} then re-run."
        )

    # Drop capacity-suspect from the join so they fail loudly (not trusted).
    join_ok = join[~join["ID"].isin(suspect)]
    try:
        md = build_ar_metadata(
            fleet, join_ok, height=args.height, model=args.model, exclude=exclude
        )
    except ValueError as exc:
        (out / "ar_md.csv").unlink(missing_ok=True)
        print(f"\nmetadata NOT written: {exc}", file=sys.stderr)
        sys.exit(2)

    md.to_csv(out / "ar_md.csv", index=False)
    lines = [
        "# Argentina (CAMMESA) join report",
        "",
        f"- wind plants: {len(fleet)}; metadata rows: {len(md)}; "
        f"capacity: {md['capacity'].sum() / 1e6:.2f} GW",
        f"- lat {md['lat'].min():.1f}..{md['lat'].max():.1f}; "
        f"Patagonia (Chubut+Santa Cruz): "
        f"{int(md['provincia'].isin(['CHUBUT', 'SANTA CRUZ']).sum())} plants",
        "- height/model uniform defaults; coords AND capacity from GWPT.",
        "",
        "| ID | region | cap MW | GWPT match |",
        "|---|---|---|---|",
    ]
    cap_kw = dict(zip(md["ID"], md["capacity"]))
    for r in join_ok.itertuples():
        if str(r.ID) in set(md["ID"]):
            reg = fleet.loc[fleet["ID"] == r.ID, "provincia"].iloc[0]
            lines.append(f"| {r.ID} | {reg} | {cap_kw.get(r.ID, 0) / 1000:.0f} | {r.gwpt_name} |")
    (out / "join_report.md").write_text("\n".join(lines))
    print(f"metadata: {len(md)} plants -> {out / 'ar_md.csv'}")
    print(f"observations -> {out / 'ar_obs.csv'} | join report -> {out / 'join_report.md'}")


if __name__ == "__main__":
    main()
