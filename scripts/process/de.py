#!/usr/bin/env python
"""Stage + validate the CONFIDENTIAL WindStats Germany data for PyVWF.

⚠ CONFIDENTIAL / COMMERCIAL (WindStats). The German wind data is a licensed
WindStats extract, not open data. Nothing it produces may be committed or
redistributed; `input/` is git-ignored, and this script only moves files the
user already holds locally.

Germany is served by the shared `european-turbine` adapter, which reads the
raw WindStats layout directly:

    DE_md.csv               per-turbine metadata: V1 (id; first 5 chars = the
                            German postcode), Manufacturer, kW, Rotor..m.,
                            Tower..m., plus quarterly snapshot columns.
    geolocate.germany.csv   postcode -> lon/lat (this is why DE needs no
                            external coordinate source: the postcode in the
                            id geolocates it).
    DE_data.csv             monthly generation: ID, Year, Month, Output (kWh),
                            Downtime.

So "processing" Germany is validating those three files and staging them into
input/observations/turbine/DE/ (the format the adapter already consumes); this
script makes that reproducible and checks the schema, rather than a silent
manual copy.

    python scripts/process/de.py --src "<path to WindStats folder>"
    python scripts/process/de.py --src "<...>" --check-only   # validate, don't copy
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

FILES = {
    "DE_md.csv": {"V1", "Manufacturer", "kW", "Rotor..m.", "Tower..m."},
    "geolocate.germany.csv": {"postcode", "lon", "lat"},
    "DE_data.csv": {"ID", "Year", "Month", "Output"},
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="Directory holding the confidential WindStats DE files")
    ap.add_argument("--out-dir", default="input/observations/turbine/DE")
    ap.add_argument("--check-only", action="store_true",
                    help="Validate the source files without staging them")
    args = ap.parse_args()

    print("=" * 70)
    print("⚠  CONFIDENTIAL WindStats (Germany): commercial, not redistributable.")
    print("=" * 70)
    src = Path(args.src)
    problems = []
    for name, required in FILES.items():
        p = src / name
        if not p.is_file():
            problems.append(f"missing: {p}")
            continue
        cols = set(pd.read_csv(p, nrows=1).columns)
        miss = required - cols
        if miss:
            problems.append(f"{name}: missing columns {sorted(miss)}")
    if problems:
        print("VALIDATION FAILED:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    # Stats (no data echoed; aggregates only).
    md = pd.read_csv(src / "DE_md.csv")
    geo = pd.read_csv(src / "geolocate.germany.csv")
    data = pd.read_csv(src / "DE_data.csv", usecols=["ID", "Year", "Output"])
    md["postcode"] = md["V1"].astype(str).str[:5]
    matched = md["postcode"].isin(geo["postcode"].astype(str)).mean()
    print(f"turbines: {len(md):,} | capacity: {md['kW'].sum()/1e6:.2f} GW")
    print(f"postcode geolocation coverage: {matched:.1%}")
    print(f"generation: {len(data):,} turbine-months, "
          f"years {int(data['Year'].min())}-{int(data['Year'].max())}")

    if args.check_only:
        print("\n--check-only: validated, nothing staged.")
        return
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(src / name, out / name)
    print(f"\nstaged 3 files -> {out}  (european-turbine reads DE from here)")
    print("Confidential: do NOT commit input/ (it is git-ignored).")


if __name__ == "__main__":
    main()
