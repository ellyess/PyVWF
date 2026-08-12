#!/usr/bin/env python
"""CLI: export a region's gridded correction-factor field (offer 103 artifact).

Thin wrapper over :func:`vwf.harness.export.export_correction_field`.

    PYVWF_INPUT=input/combined PYTHONPATH=src python \\
        scripts/analysis/export_correction_field.py \\
        --region configs/regions/dk.toml \\
        --train-run output/validation/.../train-k200 \\
        --num-clu 200 --time-res season \\
        --metrics output/validation/.../evaluate-2020-k200/metrics.csv \\
        --out output/exports/DK_correction_field_season_k200.nc
"""
import argparse
from pathlib import Path

import sys
sys.path.insert(0, "src")

from vwf.harness.export import export_correction_field
from vwf.harness.regions import load_region


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", required=True, help="path to configs/regions/<code>.toml")
    ap.add_argument("--train-run", required=True, help="train run dir with factors + fleet")
    ap.add_argument("--num-clu", type=int, required=True)
    ap.add_argument("--time-res", required=True,
                    choices=["fixed", "season", "bimonth", "month"])
    ap.add_argument("--metrics", default=None, help="evaluate metrics.csv to embed as note")
    ap.add_argument("--era5-dir", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = load_region(Path(args.region))
    out = export_correction_field(
        spec,
        args.train_run,
        num_clu=args.num_clu,
        time_res=args.time_res,
        out_path=args.out,
        metrics_csv=args.metrics,
        era5_dir=args.era5_dir,
    )
    print(f"written: {out} ({Path(out).stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
