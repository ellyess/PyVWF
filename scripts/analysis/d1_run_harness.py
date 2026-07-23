"""Run the HARNESS path for a D1 regression comparison.

The other half of the D1 method (docs/findings/d1_regression.md): run the
branch's harness on the same config as the legacy reference, with
PYTHONPATH at this tree's src/ and PYVWF_INPUT at the same staging
directory, then diff with scripts/analysis/d1_regression.py.

    PYVWF_INPUT=<stage> PYTHONPATH=<branch>/src python \\
        scripts/analysis/d1_run_harness.py --country DK --mode onshore \\
        --source european-turbine --train-start 2015 --train-end 2019 \\
        --test-year 2020 --clusters 1 10 --time-res fixed season \\
        --bbox 7.5 13.5 54.0 58.2 --out <run-dir>

Builds a RegionSpec in-memory (NH seasons; D1 is European regions only),
runs driver.run_train + run_evaluate, and flattens factors_<slice>_<n>.csv
and cor_cf_<slice>_<n>.csv (+ unc_cf, harness_metrics) into --out.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

NH = {"winter": (12, 1, 2), "spring": (3, 4, 5), "summer": (6, 7, 8), "autumn": (9, 10, 11)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True)
    ap.add_argument("--mode", default="all")
    ap.add_argument("--obs-level", default="turbine", choices=["turbine", "country"])
    ap.add_argument("--source", required=True)
    ap.add_argument("--train-start", type=int, required=True)
    ap.add_argument("--train-end", type=int, required=True)
    ap.add_argument("--test-year", type=int, required=True)
    ap.add_argument("--clusters", type=int, nargs="+", required=True)
    ap.add_argument("--time-res", nargs="+", required=True)
    ap.add_argument("--bbox", type=float, nargs=4, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vwf.harness.driver import run_evaluate, run_train
    from vwf.harness.regions import RegionSpec

    spec = RegionSpec(
        code=args.country, name=args.country, source=args.source,
        obs_level=args.obs_level,
        obs_unit="turbine" if args.obs_level == "turbine" else "country",
        train_years=(args.train_start, args.train_end),
        test_years=(args.test_year,),
        era5_path="era5/EU", bbox=tuple(args.bbox), file_tag="EU",
        correction_model="affine-wind",
        cluster_list=tuple(args.clusters), time_slices=tuple(args.time_res),
        seasons=NH,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"

    t0 = time.time()
    train_dir = run_train(spec, work, mode=args.mode, run_name="train")
    eval_dir = run_evaluate(spec, train_dir, work, mode=args.mode, run_name="eval")
    elapsed = time.time() - t0

    for f in train_dir.glob("factors_*.csv"):
        shutil.copy(f, out / f.name)
    for f in eval_dir.glob("cor_cf_*.csv"):
        shutil.copy(f, out / f.name)
    if (eval_dir / "unc_cf.csv").is_file():
        shutil.copy(eval_dir / "unc_cf.csv", out / "unc_cf.csv")
    shutil.copy(eval_dir / "metrics.csv", out / "harness_metrics.csv")

    print(f"HARNESS DONE {args.country} in {elapsed:.1f}s -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
