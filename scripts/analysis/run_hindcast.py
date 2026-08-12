#!/usr/bin/env python
"""CLI: national monthly CF hindcast in historical context (offers 068 / 015).

Applies a trained correction over every ERA5 year on disk for the region and
writes a tidy monthly series with each month ranked against its own calendar
month across the record.

    PYVWF_INPUT=input/combined PYTHONPATH=src python \\
        scripts/analysis/run_hindcast.py \\
        --region configs/regions/nz.toml \\
        --train-run output/validation/.../train-nz \\
        --num-clu 5 --time-res fixed \\
        --out output/hindcast/NZ_national_monthly.csv

Context depth = ERA5 years available for the region. A multi-decade "is the wind
weakening" claim needs ERA5 back to 1979 (scripts/fetch/era5.py); the tool prints
how many years it used.
"""
import argparse
import warnings
from pathlib import Path

import sys
sys.path.insert(0, "src")

warnings.simplefilter("ignore")

from vwf.harness.hindcast import rank_in_context, run_hindcast  # noqa: E402
from vwf.harness.regions import load_region  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", required=True)
    ap.add_argument("--train-run", required=True)
    ap.add_argument("--num-clu", type=int, required=True)
    ap.add_argument("--time-res", required=True,
                    choices=["fixed", "season", "bimonth", "month"])
    ap.add_argument("--fleet-year", type=int, default=None)
    ap.add_argument("--mode", default="all", choices=["all", "onshore", "offshore"])
    ap.add_argument("--era5-dir", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = load_region(Path(args.region))
    df = run_hindcast(
        spec, args.train_run, num_clu=args.num_clu, time_res=args.time_res,
        fleet_year=args.fleet_year, mode=args.mode, era5_dir=args.era5_dir,
    )
    ranked = rank_in_context(df)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out, index=False)

    years = sorted(df["year"].unique())
    print(f"written: {out}  ({len(df)} months, {len(years)} years: {years[0]}-{years[-1]})")
    if len(years) < 20:
        print(f"NOTE: only {len(years)} ERA5 years on disk; a multi-decade resource "
              "claim needs ERA5 back to 1979 (scripts/fetch/era5.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
