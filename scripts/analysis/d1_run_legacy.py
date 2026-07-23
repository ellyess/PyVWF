"""Run the LEGACY PyVWF path for a D1 regression reference.

Half of the D1 method (docs/findings/d1_regression.md): generate the
reference from a git worktree of main by pointing PYTHONPATH at that tree's
src/, with PYVWF_INPUT at a staging directory holding the real curve
library. dask_n_workers=0 keeps the offset fit sequential and deterministic.

    PYVWF_INPUT=<stage> PYTHONPATH=<main-worktree>/src python \\
        scripts/analysis/d1_run_legacy.py --country DK --mode onshore \\
        --train-start 2015 --train-end 2019 --test-year 2020 \\
        --clusters 1 10 --time-res fixed season --out <ref-dir>

Turbine-level (DK, DE): PyVWF.train(dask_n_workers=0) + simulate_cf.
Country-level (NL, FR): load_country_data from the observations/country files
(pass --cl-data).

Writes factors_<slice>_<n>.csv and cor_cf_<slice>_<n>.csv (+ unc_cf, obs_cf)
into --out; diff against a harness run with scripts/analysis/d1_regression.py.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True)
    ap.add_argument("--mode", default="all")
    ap.add_argument("--obs-level", default="turbine", choices=["turbine", "country"])
    ap.add_argument("--train-start", type=int, required=True)
    ap.add_argument("--train-end", type=int, required=True)
    ap.add_argument("--test-year", type=int, required=True)
    ap.add_argument("--clusters", type=int, nargs="+", required=True)
    ap.add_argument("--time-res", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cl-data", default=None, help="observations/country dir")
    args = ap.parse_args()

    from vwf.vwf import PyVWF

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dir = out / "_pyvwf_run"

    t0 = time.time()
    model = PyVWF(
        str(run_dir), args.country, True, True, args.mode,
        args.clusters, args.time_res, obs_level=args.obs_level,
    )

    if args.obs_level == "country":
        cl = Path(args.cl_data)
        c = args.country.lower()
        gp = pd.read_csv(cl / "grid_points" / c / f"{c}_grid_points.csv")
        tr = pd.read_csv(
            cl / "observations" / c / f"{c}_train_{args.train_start}_{args.train_end}.csv",
            index_col=0, parse_dates=True,
        )
        te = pd.read_csv(
            cl / "observations" / c / f"{c}_test_{args.test_year}.csv",
            index_col=0, parse_dates=True,
        )
        model.load_country_data(gp, tr, te)

    model.train(dask_n_workers=0)
    model.simulate_cf(args.test_year)
    elapsed = time.time() - t0

    cf_dir = run_dir / f"{args.country}-{args.mode}-obs_{args.obs_level}-corrected-calc_z0"
    fac_dir = cf_dir / "training" / "correction-factors"
    res_dir = cf_dir / "results" / "capacity-factor"

    for f in fac_dir.glob(f"{args.country}_factors_*.csv"):
        parts = f.stem.split("_")  # {C}_factors_{slice}_{n}
        (out / f"factors_{parts[2]}_{parts[3]}.csv").write_bytes(f.read_bytes())
    for f in res_dir.glob("*_cor_cf.csv"):
        parts = f.stem.split("_")  # {C}_{year}_{slice}_{n}_cor_cf
        (out / f"cor_cf_{parts[2]}_{parts[3]}.csv").write_bytes(f.read_bytes())
    for tag in ("unc_cf", "obs_cf"):
        src = res_dir / f"{args.country}_{args.test_year}_{tag}.csv"
        if src.is_file():
            (out / f"{tag}.csv").write_bytes(src.read_bytes())

    print(f"LEGACY DONE {args.country} in {elapsed:.1f}s -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
