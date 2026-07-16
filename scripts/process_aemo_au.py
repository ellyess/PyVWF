#!/usr/bin/env python
"""Build the AEMONemSource inputs from the raw AEMO archives + GWPT.

Reads (all local, downloaded by the user):
    <raw>/scada/PUBLIC_DVD_DISPATCH_UNIT_SCADA_*.zip
    the AEMO "Generation Information" workbook (xlsx; needs openpyxl,
        part of the `data` extra)
    the Global Wind Power Tracker workbook (the user's public-source
        turbine compilation; coordinates)

Writes (under <out>, default input/turbine_level_data/AU_NEM/):
    au_nem_md.csv                          AEMONemSource metadata contract
    au_nem_scada_monthly_partials.csv      per-(DUID, UTC month) energy sums
    join_report.md                         match/unmatch/capacity-check report

The SCADA archives are cut on MARKET-time month boundaries and straddle UTC
months, so each archive is reduced to per-UTC-month partials
(vwf.sources.aemo.scada_partial_aggregate) and the partials are summed
across archives before any month is finalised. Finalisation itself
(coverage floor, commissioning mask) happens inside AEMONemSource at load
time, through the same audited code path the tests pin.

    python scripts/process_aemo_au.py \\
        --gen-info "input/aemo_raw/NEM Generation Information Apr 2026.xlsx" \\
        --gwpt input/Global-Wind-Power-Tracker-February-2026.xlsx
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

from vwf.datasets.aemo_au import (
    build_au_metadata,
    farms_from_gwpt,
    join_fleet_to_gwpt,
    parse_mms_table,
    wind_fleet_from_gen_info,
)
from vwf.sources.aemo import combine_partials, scada_partial_aggregate


def read_zipped_mms(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        member = zf.namelist()[0]
        with zf.open(member) as fh:
            return parse_mms_table(io.TextIOWrapper(fh, "utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="input/aemo_raw", help="Raw archive dir")
    ap.add_argument("--gen-info", required=True, help="Generation Information xlsx")
    ap.add_argument("--gwpt", required=True, help="Global Wind Power Tracker xlsx")
    ap.add_argument("--out", default="input/turbine_level_data/AU_NEM")
    ap.add_argument("--height", type=float, default=100.0,
                    help="Uniform hub height default, m (no per-farm data yet)")
    ap.add_argument("--model", default="Synthetic.Onshore2000",
                    help="Uniform power-curve model key (must be a column of "
                    "your power_curves.csv; pick a real-library key for real runs)")
    ap.add_argument("--aliases", default=None,
                    help="Optional CSV with columns gen_info_name,gwpt_name for "
                    "manual join overrides")
    args = ap.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- fleet metadata -----------------------------------------------------
    gen_info = pd.read_excel(args.gen_info, sheet_name="Generator Information", header=3)
    gwpt = pd.read_excel(args.gwpt, sheet_name="Data")
    aliases = None
    if args.aliases:
        alias_df = pd.read_csv(args.aliases)
        aliases = dict(zip(alias_df["gen_info_name"], alias_df["gwpt_name"]))

    fleet = wind_fleet_from_gen_info(gen_info)
    farms = farms_from_gwpt(gwpt)
    matched, unmatched_fleet, unmatched_gwpt = join_fleet_to_gwpt(fleet, farms, aliases)
    metadata = build_au_metadata(matched, height=args.height, model=args.model)
    metadata.to_csv(out / "au_nem_md.csv", index=False)

    # --- SCADA partials -----------------------------------------------------
    wind_duids = set(metadata["ID"])
    archives = sorted(raw.glob("scada/PUBLIC_DVD_DISPATCH_UNIT_SCADA_*.zip"))
    if not archives:
        print(f"No SCADA archives under {raw}/scada — metadata written, "
              "partials skipped.", file=sys.stderr)
    partials = []
    for i, archive in enumerate(archives, 1):
        table = read_zipped_mms(archive)
        table = table.rename(
            columns={"SETTLEMENTDATE": "timestamp", "DUID": "ID", "SCADAVALUE": "mw"}
        )
        table = table[table["ID"].isin(wind_duids)]
        partials.append(scada_partial_aggregate(table))
        print(f"[{i}/{len(archives)}] {archive.name}: {len(table)} wind rows")

    if partials:
        combined = combine_partials(partials)
        combined.to_csv(out / "au_nem_scada_monthly_partials.csv", index=False)
        print(f"partials: {len(combined)} (DUID, month) rows -> "
              f"{out / 'au_nem_scada_monthly_partials.csv'}")

    # --- join report ----------------------------------------------------------
    matched = matched.assign(
        capacity_ratio=matched["capacity_mw"] / matched["gwpt_capacity_mw"]
    )
    suspicious = matched[(matched["capacity_ratio"] < 0.6) | (matched["capacity_ratio"] > 1.7)]
    lines = [
        "# AU-NEM fleet join report",
        "",
        f"- Generation Information wind DUIDs (NEM, in service/commissioning): {len(fleet)}",
        f"- GWPT operating AU projects: {len(farms)}",
        f"- matched: {len(matched)}  |  unmatched DUIDs: {len(unmatched_fleet)}  |  "
        f"unmatched GWPT: {len(unmatched_gwpt)}",
        f"- capacity-suspicious matches (GI/GWPT ratio outside [0.6, 1.7]): {len(suspicious)}",
        "",
        "Height and model are UNIFORM DEFAULTS (see *_source columns): "
        f"height={args.height} m, model={args.model}.",
        "Static nameplate capacity: staged-commissioning months are biased; "
        "DUDETAILSUMMARY capacity histories are the named follow-up.",
        "",
        "## Unmatched Generation Information DUIDs (need aliases or have no GWPT entry)",
        unmatched_fleet[["ID", "site_name", "region", "capacity_mw"]].to_string(index=False)
        if len(unmatched_fleet) else "(none)",
        "",
        "## Capacity-suspicious matches",
        suspicious[["ID", "site_name", "gwpt_name", "capacity_mw", "gwpt_capacity_mw"]]
        .to_string(index=False) if len(suspicious) else "(none)",
        "",
        "## Unmatched GWPT projects (context only; many are non-NEM, e.g. WA)",
        unmatched_gwpt[["gwpt_name", "gwpt_capacity_mw"]].to_string(index=False)
        if len(unmatched_gwpt) else "(none)",
        "",
    ]
    (out / "join_report.md").write_text("\n".join(lines))
    print(f"metadata: {len(metadata)} farms -> {out / 'au_nem_md.csv'}")
    print(f"join report -> {out / 'join_report.md'}")
    if len(unmatched_fleet):
        print(f"NOTE: {len(unmatched_fleet)} DUIDs lack coordinates and are "
              "EXCLUDED from the metadata; see the report.", file=sys.stderr)


if __name__ == "__main__":
    main()
