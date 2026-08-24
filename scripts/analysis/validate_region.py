#!/usr/bin/env python
"""Thin CLI over the validation-harness driver (vwf.harness.driver).

Examples:
    python scripts/analysis/validate_region.py train --region configs/regions/dk.toml
    python scripts/analysis/validate_region.py evaluate --region configs/regions/dk.toml \
        --train-run output/validation/DK/train-20260715T120000Z
    python scripts/analysis/validate_region.py transfer --region configs/regions/uk.toml \
        --source-region configs/regions/au_nem.toml \
        --source-run output/validation/AU-NEM/train-20260715T120000Z

All logic lives in vwf.harness.driver so it is importable and tested; this
file only parses arguments.
"""
import argparse

from vwf.harness.driver import run_evaluate, run_train, run_transfer
from vwf.harness.regions import load_region


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--region", required=True, help="Region config TOML")
    common.add_argument("--out", default="output/validation", help="Output root")
    common.add_argument("--mode", default="all", choices=["all", "onshore", "offshore"])
    common.add_argument("--run-name", default=None, help="Run directory suffix")

    sub.add_parser("train", parents=[common])

    p_eval = sub.add_parser("evaluate", parents=[common])
    p_eval.add_argument("--train-run", required=True, help="Training run directory")
    p_eval.add_argument("--year", type=int, default=None)

    p_tr = sub.add_parser("transfer", parents=[common])
    p_tr.add_argument("--source-region", required=True, help="Source region config TOML")
    p_tr.add_argument("--source-run", required=True, help="Source training run directory")
    p_tr.add_argument("--year", type=int, default=None)

    args = parser.parse_args()
    spec = load_region(args.region)

    if args.command == "train":
        run_dir = run_train(spec, args.out, mode=args.mode, run_name=args.run_name)
    elif args.command == "evaluate":
        run_dir = run_evaluate(
            spec, args.train_run, args.out,
            year=args.year, mode=args.mode, run_name=args.run_name,
        )
    else:
        run_dir = run_transfer(
            load_region(args.source_region), args.source_run, spec, args.out,
            year=args.year, mode=args.mode, run_name=args.run_name,
        )

    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
