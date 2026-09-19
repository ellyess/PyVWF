"""``pyvwf-validate``: train, evaluate or transfer one region through the harness.

The validation harness (:mod:`vwf.harness.driver`) runs one region at a time
from its config in ``configs/regions/``. Each command writes a run directory
under ``--out``, with a manifest recording the version, the git state, the
config and the curve library.

Examples:
    pyvwf-validate train --region configs/regions/dk.toml
    pyvwf-validate evaluate --region configs/regions/dk.toml \\
        --train-run output/validation/DK/train-20260715T120000Z
    pyvwf-validate transfer --region configs/regions/uk.toml \\
        --source-region configs/regions/au_nem.toml \\
        --source-run output/validation/AU-NEM/train-20260715T120000Z

``scripts/analysis/validate_region.py`` runs the same command from a checkout.
"""

from __future__ import annotations

import argparse

from vwf.cli.common import add_region, entry, make_parser, run


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser(__doc__, prog="pyvwf-validate")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    add_region(common)
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
    return parser


def execute(args: argparse.Namespace) -> None:
    """Run the command ``args`` names; print the run directory."""
    from vwf.harness.driver import run_evaluate, run_train, run_transfer
    from vwf.harness.regions import load_region

    spec = load_region(args.region)
    if args.command == "train":
        run_dir = run_train(spec, args.out, mode=args.mode, run_name=args.run_name)
    elif args.command == "evaluate":
        run_dir = run_evaluate(
            spec,
            args.train_run,
            args.out,
            year=args.year,
            mode=args.mode,
            run_name=args.run_name,
        )
    else:
        run_dir = run_transfer(
            load_region(args.source_region),
            args.source_run,
            spec,
            args.out,
            year=args.year,
            mode=args.mode,
            run_name=args.run_name,
        )
    print(f"Run complete: {run_dir}")


def main(argv: list[str] | None = None) -> int:
    return run(execute, build_parser(), argv)


if __name__ == "__main__":
    entry(execute, build_parser())
