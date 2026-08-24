#!/usr/bin/env python
"""Build the UK (european-turbine) inputs from REPD and Ofgem ROC data.

Two sub-commands, matching the two source components (docs/runbooks/UK.md):

    python scripts/process/uk.py metadata          # REPD -> open uk_md + divergence
    python scripts/process/uk.py observations --roc <ofgem export>   # ROCs -> ukobs

`metadata` reconstructs an OPEN per-turbine metadata table from the Renewable
Energy Planning Database (location, capacity, turbine count; uniform-default
model + hub height where REPD carries none), and writes a divergence report
against the committed curated `uk_md.csv`. It does NOT overwrite the curated
file; the open table ships alongside as `uk_md_open.csv` for comparison.

`observations` converts a per-station Ofgem ROC issuance export to the
pseudo-replicated `ukobs.csv` format: ROCs -> MWh (grandfathered banding) ->
equal split across each station's turbines. The station->turbine-count mapping
comes from the curated `uk_md.csv` (the accreditation-number key the open REPD
table does not carry); pass `--turbine-counts` to use a different source.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from vwf.datasets.uk_roc import (
    divergence_report,
    pseudo_replicate_metadata,
    pseudo_replicate_observations,
    read_ofgem_confidential_certificates,
    repd_wind_metadata,
    roc_issuance_to_station_monthly,
)

UK_DIR = Path("input/observations/turbine/UK")


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin-1", low_memory=False)


def cmd_metadata(args) -> None:
    repd = _read_any(Path(args.repd))
    station_md = repd_wind_metadata(repd, height_default=args.height, model=args.model)
    open_turb = pseudo_replicate_metadata(station_md)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    open_turb.to_csv(out / "uk_md_open.csv", index=False)

    curated_path = UK_DIR / "uk_md.csv"
    lines = [
        "# UK metadata: open REPD reconstruction vs curated table", "",
        f"- open (REPD): {len(station_md)} stations, {len(open_turb)} turbine rows",
        "- height: REPD tip height where present, else uniform default; "
        "NO turbine model or rotor diameter in REPD (uniform defaults used).",
    ]
    if curated_path.is_file():
        rep = divergence_report(open_turb, pd.read_csv(curated_path))
        lines += [
            "", "## fleet-level divergence (keys differ: REPD Ref ID vs ROC "
            "accreditation number, so this is not a row match)", "",
            "| metric | open (REPD, current fleet) | curated (2015-2019 RO set) |",
            "|---|---|---|",
            f"| turbine rows | {rep['open']['turbine_rows']} | {rep['curated']['turbine_rows']} |",
            f"| stations | {rep['open']['stations']} | {rep['curated']['stations']} |",
            f"| total capacity (GW) | {rep['open']['total_capacity_gw']} | {rep['curated']['total_capacity_gw']} |",
            f"| median height (m) | {rep['open']['median_height_m']} | {rep['curated']['median_height_m']} |",
            "",
            "The open table is the CURRENT operational fleet (REPD has no ROC "
            "accreditation key and no snapshot filter), so it is larger than "
            "the 2015-2019 ROC-accredited set the curated table covers. Use the "
            "open table for open/reproducible work; the curated table remains "
            "the committed input for reproducing published DK/DE/UK results.",
        ]
    (out / "uk_md_open_divergence.md").write_text("\n".join(lines))
    print(f"open metadata: {len(open_turb)} turbine rows -> {out/'uk_md_open.csv'}")
    print(f"divergence report -> {out/'uk_md_open_divergence.md'}")


def cmd_observations(args) -> None:
    if not args.ofgem_confidential and not args.roc:
        sys.exit("provide --roc (public RER export) or --ofgem-confidential "
                 "(the licensed certificate-warehouse CSVs).")
    if args.ofgem_confidential:
        import glob
        paths = sorted(p for g in args.ofgem_confidential for p in glob.glob(g))
        if not paths:
            sys.exit(f"no files matched {args.ofgem_confidential}")
        print("=" * 70)
        print("⚠  CONFIDENTIAL Ofgem certificate warehouse: differently licensed.")
        print("   Output is CONFIDENTIAL; input/ is git-ignored. Do NOT commit or")
        print("   redistribute the derived observations. (Open path: --roc, from")
        print("   the public RER export.)")
        print("=" * 70)
        station_monthly = read_ofgem_confidential_certificates(paths)
        out_name = "ukobs_confidential.csv"
    else:
        roc = _read_any(Path(args.roc))
        cols = {c.lower(): c for c in roc.columns}

        def pick(*cands, required=True):
            for c in cands:
                if c.lower() in cols:
                    return cols[c.lower()]
            if required:
                sys.exit(f"ROC export is missing a column like {cands}; found "
                         f"{list(roc.columns)}. Pass --station-col/--period-col/"
                         "--certs-col to name them explicitly.")
            return None

        station = args.station_col or pick("AccreditationNumber", "Accreditation Number",
                                           "Generating Station", "Station")
        period = args.period_col or pick("OutputPeriod", "Output Period", "Period", "Month")
        certs = args.certs_col or pick("Certificates", "ROCs", "No. of Certificates",
                                       "Number of Certificates")
        station_monthly = roc_issuance_to_station_monthly(
            roc, station_col=station, period_col=period, certs_col=certs)
        out_name = "ukobs_open.csv"

    tc_path = Path(args.turbine_counts) if args.turbine_counts else UK_DIR / "uk_md.csv"
    if not tc_path.is_file():
        sys.exit(f"{tc_path} not found; need a station->turbine-count source "
                 "(the curated uk_md.csv, or --turbine-counts).")
    md = pd.read_csv(tc_path)
    md["station"] = md["ID"].astype(str).str.replace(r"-\d+$", "", regex=True)
    counts = md.groupby("station").size()

    obs = pseudo_replicate_observations(station_monthly, counts)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    obs.to_csv(out / out_name, index=False)
    n_st = station_monthly["ID"].nunique()
    print(f"ROC stations: {n_st} | matched to turbine counts: "
          f"{station_monthly[station_monthly['ID'].isin(counts.index)]['ID'].nunique()}")
    print(f"observations: {len(obs)} turbine rows -> {out/out_name}")
    print(f"(saved as {out_name}, not overwriting the committed ukobs.csv)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("metadata", help="REPD -> open uk_md + divergence report")
    m.add_argument("--repd", default="input/raw/repd/repd_wind.csv")
    m.add_argument("--out-dir", default=str(UK_DIR))
    m.add_argument("--height", type=float, default=100.0)
    m.add_argument("--model", default="2019COE_Market_Average_2.6MW_121")
    m.set_defaults(func=cmd_metadata)

    o = sub.add_parser("observations", help="Ofgem ROC export -> ukobs")
    o.add_argument("--roc", help="Public Ofgem RER per-station issuance export")
    o.add_argument("--ofgem-confidential", nargs="+", metavar="GLOB",
                   help="CONFIDENTIAL Ofgem certificate-warehouse CSV(s) "
                   "(differently licensed; reproduces the committed ukobs exactly)")
    o.add_argument("--turbine-counts", default=None,
                   help="station->turbine-count source (default: curated uk_md.csv)")
    o.add_argument("--out-dir", default=str(UK_DIR))
    o.add_argument("--station-col", default=None)
    o.add_argument("--period-col", default=None)
    o.add_argument("--certs-col", default=None)
    o.set_defaults(func=cmd_observations)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
