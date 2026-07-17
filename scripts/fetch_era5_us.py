#!/usr/bin/env python
"""Fetch the ERA5 subset for the contiguous US from the Copernicus CDS.

USER-EXECUTED. This script submits requests through your CDS credentials, so
running it is yours to do, like every other download on this project:

    pip install cdsapi              # not a PyVWF dependency
    # ~/.cdsapirc must hold your CDS url + key
    python scripts/fetch_era5_us.py            # all 48 months, 2019-2022
    python scripts/fetch_era5_us.py --dry-run  # list requests, submit nothing
    python scripts/fetch_era5_us.py --years 2021 2022

What it requests (per month, one CDS request each — a single multi-year
request exceeds the per-request item cap and gets rejected):

    dataset   reanalysis-era5-single-levels, hourly, all days/times
    variables 100m u/v (wind) + 10m u/v (needed for the roughness calc)
    area      N 50 / W -125 / S 24 / E -66   (configs/regions/us.toml, CONUS)
    grid      0.25 x 0.25
    format    netcdf, unarchived

The CONUS box is ~7x the area of the AU-NEM box, so each monthly file is
large (~hundreds of MB) and a four-year hourly load will not fit in memory —
run scripts/combine_era5_us_daily.py afterwards to reduce each year to a
daily file, exactly as the AU workflow does.

Output layout matches what the harness expects for the US region config
(era5_path = "era5/US"): <input-root>/era5/US/era5_us_<YYYY>_<MM>.nc, where
<input-root> is $PYVWF_INPUT if set, else ./input. vwf.datasets.era5 globs
*.nc in that directory and combines by coordinates; roughness is derived from
the 10m/100m shear at load time (calc_z0=True), so no preprocessing step is
required before running the harness (though the daily pre-combine is strongly
recommended for a box this size).

Requests run sequentially and the script is resumable: completed months are
skipped, partial downloads land in a .part file and are renamed only on
success. Expect the CDS queue, not bandwidth, to dominate wall-clock time.

If the very first request fails with an authentication error and your CDS
key predates the CDS account migration, regenerate the token in your CDS
profile — token format, not this script.
"""
import argparse
import os
import sys
import time
from pathlib import Path

DATASET = "reanalysis-era5-single-levels"
VARIABLES = [
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
#: North, West, South, East — the us.toml bounding box in CDS order (CONUS).
AREA = [50, -125, 24, -66]
GRID = [0.25, 0.25]
YEARS = (2019, 2020, 2021, 2022)


def output_dir() -> Path:
    root = Path(os.environ.get("PYVWF_INPUT", "input"))
    return root / "era5" / "US"


def month_request(year: int, month: int) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, 32)],  # CDS ignores invalid days
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": AREA,
        "grid": GRID,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, nargs="+", default=list(YEARS),
                    help="Years to fetch (default: 2019-2022)")
    ap.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)),
                    help="Months to fetch (default: all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the request plan and exit without submitting")
    args = ap.parse_args()

    out_dir = output_dir()
    plan = [
        (year, month, out_dir / f"era5_us_{year}_{month:02d}.nc")
        for year in sorted(args.years)
        for month in sorted(args.months)
    ]
    todo = [(y, m, p) for y, m, p in plan if not p.is_file()]
    print(f"Output directory: {out_dir}")
    print(f"{len(plan)} month(s) in plan, {len(plan) - len(todo)} already present, "
          f"{len(todo)} to fetch.")

    if args.dry_run:
        for year, month, path in todo:
            print(f"  would request {year}-{month:02d} -> {path.name}")
        return
    if not todo:
        return

    import cdsapi  # imported here so --dry-run works without it installed

    out_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    failures = []
    for i, (year, month, path) in enumerate(todo, 1):
        part = path.with_suffix(".nc.part")
        print(f"[{i}/{len(todo)}] {year}-{month:02d} -> {path.name}", flush=True)
        t0 = time.time()
        try:
            client.retrieve(DATASET, month_request(year, month), str(part))
            part.rename(path)
            print(f"    done in {time.time() - t0:.0f}s "
                  f"({path.stat().st_size / 1e6:.0f} MB)", flush=True)
        except Exception as exc:
            part.unlink(missing_ok=True)
            failures.append((year, month, str(exc)))
            print(f"    FAILED: {exc}", flush=True)
            if "auth" in str(exc).lower() or "401" in str(exc):
                print("    (Authentication failure on the first request usually "
                      "means a pre-migration CDS token: regenerate it in your "
                      "CDS profile and re-run — completed months are skipped.)")

    if failures:
        print(f"\n{len(failures)} month(s) failed; re-run to retry just those:")
        for year, month, err in failures:
            print(f"  {year}-{month:02d}: {err[:100]}")
        sys.exit(1)
    print("\nAll requested months present.")


if __name__ == "__main__":
    main()
