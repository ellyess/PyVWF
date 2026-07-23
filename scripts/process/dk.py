#!/usr/bin/env python
"""Build the Denmark (european-turbine) inputs from the ens.dk Excel files.

Reduces the two raw Danish Energy Agency workbooks (downloaded by
``scripts/fetch/dk.py``) to the CSVs the ``european-turbine`` adapter reads
for Denmark:

    anlaeg.xlsx                -> dk_md.csv               (per-turbine metadata)
    maanedsdata_2002_2020.xlsx -> dk_obs_2002_2020.csv    (monthly production)

    python scripts/process/dk.py
    python scripts/process/dk.py --metadata-only

The pure transforms live in ``vwf.datasets.process_dk_raw_data`` (the header
rows, the UTM->lon/lat conversion, the per-year sheet reshaping); this is the
thin CLI over them, matching the other ``scripts/process/<region>.py`` entry
points. Unlike the other regions Denmark keeps its shared ``european-turbine``
adapter (DK/DE/UK) — this only produces its input files.
"""
import argparse
import sys
from pathlib import Path

from vwf.datasets.process_dk_raw_data import (
    process_dk_metadata,
    process_dk_monthly_observations,
)

ANLAEG = "anlaeg.xlsx"
MAANEDSDATA = "maanedsdata_2002_2020.xlsx"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, default=Path("input/observations/turbine/DK"),
                    help="Directory holding the raw ens.dk .xlsx files")
    ap.add_argument("--out-dir", type=Path, default=Path("input/observations/turbine/DK"),
                    help="Directory for the processed CSVs")
    ap.add_argument("--metadata-only", action="store_true")
    ap.add_argument("--observations-only", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.observations_only:
        src = args.in_dir / ANLAEG
        if not src.is_file():
            sys.exit(f"{src} not found — run scripts/fetch/dk.py first.")
        process_dk_metadata(src, args.out_dir / "dk_md.csv", verbose=True)

    if not args.metadata_only:
        src = args.in_dir / MAANEDSDATA
        if not src.is_file():
            sys.exit(f"{src} not found — run scripts/fetch/dk.py first.")
        process_dk_monthly_observations(
            src, args.out_dir / "dk_obs_2002_2020.csv", verbose=True)

    print("\nDone. dk_md.csv / dk_obs_2002_2020.csv are what european-turbine reads for DK.")


if __name__ == "__main__":
    main()
