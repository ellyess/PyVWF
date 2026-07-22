#!/usr/bin/env python
"""Fetch the ERA5 subset for Chile from the Copernicus CDS.

USER-EXECUTED. This script submits requests through your CDS credentials, so
running it is yours to do, like every other download on this project:

    pip install cdsapi              # not a PyVWF dependency
    # ~/.cdsapirc must hold your CDS url + key
    python scripts/fetch_era5_cl.py            # 48 months, 2021-2024
    python scripts/fetch_era5_cl.py --dry-run  # list requests, submit nothing

What it requests (per month, one CDS request each — a single multi-year
request exceeds the per-request item cap and gets rejected):

    dataset   reanalysis-era5-single-levels, hourly, all days/times
    variables 100m u/v (wind) + 10m u/v (needed for the roughness calc)
    area      N -21 / W -75 / S -44 / E -68   (see bbox rationale below)
    grid      0.25 x 0.25
    format    netcdf, unarchived

**Why the box stops at -44 and not Cape Horn.** The Global Wind Power Tracker
puts 63 of Chile's 64 operating wind farms (6.29 of 6.30 GW) between -42.3 and
-22.5. The single exception is Vientos Patagónicos (10 MW, -52.94), separated
from the rest of the fleet by a 10.7-degree latitude gap — and Magallanes is
an isolated system, not part of the Coordinador-run SEN, so it does not appear
in the generation data this region trains on. Extending the box to reach it
would add ~30% to every ERA5 file for 0.16% of the fleet's capacity. This is
the US lesson (six Hawaii plants were being simulated with Californian wind
until the fleet was screened by bounding box) applied before it bites: if a
plant IS found south of -44 in the observations, it must be screened out
explicitly rather than silently matched to the nearest in-box grid cell.

At 29 x 93 cells per hour the monthly files are modest; no daily pre-combine
step is required for a box this size (unlike BR/US).

Output layout matches what the harness expects for the CL region config
(era5_path = "era5/CL"): <input-root>/era5/CL/era5_cl_<YYYY>_<MM>.nc, where
<input-root> is $PYVWF_INPUT if set, else ./input. Roughness is derived from
the 10m/100m shear at load time (calc_z0=True), so no preprocessing step is
required before running the harness.

Requests run sequentially and the script is resumable: completed months are
skipped, partial downloads land in a .part file and are renamed only on
success. Expect the CDS queue, not bandwidth, to dominate wall-clock time.
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
#: North, West, South, East — the planned cl.toml bounding box, CDS order.
#: (The region config ships with the adapter; this box is its source of truth.)
AREA = [-21, -75, -44, -68]
GRID = [0.25, 0.25]
YEARS = (2021, 2022, 2023, 2024)


def output_dir() -> Path:
    root = Path(os.environ.get("PYVWF_INPUT", "input"))
    return root / "era5" / "CL"


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
                    help="Years to fetch (default: 2021-2024)")
    ap.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)),
                    help="Months to fetch (default: all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the request plan and exit without submitting")
    args = ap.parse_args()

    out_dir = output_dir()
    plan = [
        (year, month, out_dir / f"era5_cl_{year}_{month:02d}.nc")
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

    if failures:
        print(f"\n{len(failures)} month(s) failed; re-run to retry just those:")
        for year, month, err in failures:
            print(f"  {year}-{month:02d}: {err[:100]}")
        sys.exit(1)
    print("\nAll requested months present.")


if __name__ == "__main__":
    main()
