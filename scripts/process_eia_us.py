#!/usr/bin/env python
"""Build the EIAUSSource inputs from the public federal data.

Reads (all local, downloaded by the user; all public-domain US government):
    one or more EIA-923 workbooks (xlsx; "Page 1 Generation and Fuel Data"
        sheet — the schedule preamble sits above the header, so --eia923-header
        selects the header row, 0-based)
    the EIA-860 plant table   (2___Plant_Y<year>.xlsx, "Plant" sheet)
    the EIA-860 generator table (3_1_Generator_Y<year>.xlsx, "Operable" sheet)
    the USWTDB turbine table   (uswtdb_*.csv)

Writes (under <out>, default input/turbine_level_data/US/):
    us_md.csv                 EIAUSSource plant metadata contract
    us_eia923_netgen.csv      long per-(plant, year, month) net generation
    join_report.md            fleet/coordinate/height-provenance report

Finalisation (CF arithmetic, annual-respondent screen, commissioning mask)
happens inside EIAUSSource at load time, through the same audited code path
the tests pin — this script only assembles the two on-disk tables.

    python scripts/process_eia_us.py \\
        --eia923 input/eia_raw/EIA923_2019.xlsx input/eia_raw/EIA923_2020.xlsx \\
        --eia860-plants input/eia_raw/2___Plant_Y2021.xlsx \\
        --eia860-generators input/eia_raw/3_1_Generator_Y2021.xlsx \\
        --uswtdb input/eia_raw/uswtdb_v6_1_20231128.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from vwf.datasets.eia_us import (
    build_us_metadata,
    filter_to_bbox,
    plant_hub_heights_from_uswtdb,
    wind_capacity_from_eia860,
    wind_generation_from_eia923,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eia923", nargs="+", required=True,
                    help="One or more EIA-923 workbooks (one per year)")
    ap.add_argument("--eia923-sheet", default="Page 1 Generation and Fuel Data")
    ap.add_argument("--eia923-header", type=int, default=5,
                    help="0-based header row of the Page 1 sheet (preamble above)")
    ap.add_argument("--eia860-plants", required=True, help="EIA-860 plant xlsx")
    ap.add_argument("--eia860-plants-sheet", default="Plant")
    ap.add_argument("--eia860-generators", required=True, help="EIA-860 generator xlsx")
    ap.add_argument("--eia860-generators-sheet", default="Operable")
    ap.add_argument("--eia860-header", type=int, default=1,
                    help="0-based header row of the EIA-860 sheets")
    ap.add_argument("--uswtdb", default=None, help="USWTDB CSV (hub heights, models)")
    ap.add_argument("--out", default="input/turbine_level_data/US")
    ap.add_argument("--bbox", type=float, nargs=4, default=None,
                    metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
                    help="Drop plants outside this box (use the region's "
                    "[era5] bbox, e.g. -125 -66 24 50 for CONUS). EIA covers "
                    "all states, so without this Alaska/Hawaii plants are "
                    "silently snapped to the nearest in-domain grid cell.")
    ap.add_argument("--default-height", type=float, default=100.0,
                    help="Uniform hub-height fallback, m, for plants USWTDB "
                    "does not cover")
    ap.add_argument("--model", default="2019COE_Market_Average_2.6MW_121",
                    help="Uniform power-curve key (must be a column of "
                    "power_curves.csv). Defaults to the bundled open library's "
                    "most recent market-average utility curve; override with a "
                    "specific reference or your own licensed key.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- generation ---------------------------------------------------------
    netgen_parts = []
    for wb in args.eia923:
        page1 = pd.read_excel(wb, sheet_name=args.eia923_sheet, header=args.eia923_header)
        netgen_parts.append(wind_generation_from_eia923(page1))
    netgen = pd.concat(netgen_parts, ignore_index=True)
    netgen = netgen.drop_duplicates(subset=["ID", "year", "month"], keep="last")
    netgen.to_csv(out / "us_eia923_netgen.csv", index=False)

    # --- capacity + coordinates --------------------------------------------
    plants = pd.read_excel(
        args.eia860_plants, sheet_name=args.eia860_plants_sheet, header=args.eia860_header
    )
    generators = pd.read_excel(
        args.eia860_generators,
        sheet_name=args.eia860_generators_sheet,
        header=args.eia860_header,
    )
    capacity = wind_capacity_from_eia860(generators, plants)

    # --- hub heights / models ----------------------------------------------
    hub_heights = None
    if args.uswtdb:
        uswtdb = pd.read_csv(args.uswtdb, low_memory=False)
        hub_heights = plant_hub_heights_from_uswtdb(uswtdb)

    metadata = build_us_metadata(
        capacity, hub_heights, default_height=args.default_height, model=args.model
    )

    # Keep the fleet inside the reanalysis domain (see filter_to_bbox).
    n_before = len(metadata)
    dropped_out_of_domain = pd.DataFrame(columns=metadata.columns)
    if args.bbox:
        dropped_out_of_domain = metadata[
            ~metadata["ID"].isin(filter_to_bbox(metadata, tuple(args.bbox))["ID"])
        ]
        metadata = filter_to_bbox(metadata, tuple(args.bbox))

    metadata.to_csv(out / "us_md.csv", index=False)

    # --- report -------------------------------------------------------------
    obs_plants = set(netgen["ID"])
    md_plants = set(metadata["ID"])
    gen_no_meta = sorted(obs_plants - md_plants)
    meta_no_gen = sorted(md_plants - obs_plants)
    from_uswtdb = int((metadata["height_source"] == "uswtdb-capacity-weighted").sum())
    lines = [
        "# US (EIA) fleet join report",
        "",
        f"- EIA-923 wind plant-months: {len(netgen)} "
        f"({netgen['ID'].nunique()} plants, "
        f"{sorted(netgen['year'].dropna().unique().tolist())})",
        f"- EIA-860 wind plants with coordinates: {len(metadata)}",
        f"- plants with generation but no metadata (dropped from md): {len(gen_no_meta)}",
        f"- plants with metadata but no generation in-window: {len(meta_no_gen)}",
        f"- hub height from USWTDB (capacity-weighted): {from_uswtdb} / {len(metadata)}; "
        f"the rest use the {args.default_height} m uniform default",
        (
            f"- OUT-OF-DOMAIN plants dropped by --bbox {list(args.bbox)}: "
            f"{len(dropped_out_of_domain)} of {n_before} "
            f"({dropped_out_of_domain['capacity'].sum() / 1e6:.2f} GW). EIA covers "
            "all states; the region's ERA5 box does not. Without this screen they "
            "are snapped to the nearest in-domain grid cell and simulated with "
            "the wrong wind."
            if args.bbox
            else "- NO --bbox screen applied: any plant outside the region's ERA5 "
            "box is snapped to the nearest in-domain grid cell (wrong wind). "
            "Pass --bbox with the region's [era5] bbox."
        ),
        "",
        "Height source is per-plant (height_source column). Model is a UNIFORM "
        f"default ({args.model}); the USWTDB manufacturer/model string travels in "
        "uswtdb_model for a future vintage-aware assignment but is never the curve key.",
        "Capacity is EIA-860 nameplate (static over the window); staged build-outs "
        "bias early months low (partial-staging follow-up, as for AU).",
        "EIA-923 Netgen is NET of station use; annual respondents are imputed and "
        "screened at load time.",
        "",
        "## Plants with generation but no EIA-860 metadata",
        "\n".join(gen_no_meta) if gen_no_meta else "(none)",
    ]
    (out / "join_report.md").write_text("\n".join(lines))
    print(f"netgen: {len(netgen)} plant-months -> {out / 'us_eia923_netgen.csv'}")
    print(f"metadata: {len(metadata)} plants -> {out / 'us_md.csv'}")
    print(f"join report -> {out / 'join_report.md'}")
    if gen_no_meta:
        print(f"NOTE: {len(gen_no_meta)} plants have generation but no coordinates "
              "and are EXCLUDED from the metadata; see the report.", file=sys.stderr)


if __name__ == "__main__":
    main()
