"""``pyvwf-train``: train bias corrections and simulate a year of CF for one country.

Mirrors the ``examples/quick_run.py`` flow:
    1) instantiate PyVWF with an output directory
    2) train the correction factors
    3) simulate the capacity factor time series for the test year

For batch runs across many countries/configurations, use the research driver
``scripts/train_all_bias_corrections.py`` instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vwf.vwf import PyVWF


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pyvwf-train",
        description="Train PyVWF bias corrections and simulate a year of capacity factor.",
    )
    p.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory (PyVWF creates subfolders/files here).",
    )
    p.add_argument("--country", type=str, default="DK", help='Country code (e.g. "DK", "DE")')
    p.add_argument("--year-test", type=int, default=2020, help="Year to simulate (e.g. 2020)")
    p.add_argument(
        "--calc-z0",
        action="store_true",
        help="Compute surface roughness from 10m/100m wind (default: off).",
    )
    p.add_argument(
        "--cluster-mode",
        type=str,
        default="onshore",
        choices=["all", "onshore", "offshore"],
        help="Turbine subset used in clustering.",
    )
    p.add_argument(
        "--cluster-list",
        type=int,
        nargs="*",
        default=[1, 2, 5, 7, 10],
        help="Cluster counts to evaluate.",
    )
    p.add_argument(
        "--time-res-list",
        type=str,
        nargs="*",
        default=["fixed", "season", "bimonth", "month"],
        help="Time-resolution modes to evaluate.",
    )
    p.add_argument(
        "--add-nan",
        type=float,
        default=None,
        help="Optional fraction/percent of observations to drop.",
    )
    p.add_argument(
        "--interp-nan",
        type=int,
        default=None,
        help="Optional monthly interpolation limit for NaNs.",
    )
    p.add_argument(
        "--fix-turb",
        type=str,
        default=None,
        help='Optional fixed turbine model/id, e.g. "2019COE_Market_Average_2.6MW_121".',
    )
    p.add_argument(
        "--train-plots",
        action="store_true",
        help="Emit verbose training plots (maps to PyVWF.train(True)).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[pyvwf-train] Output directory: {outdir}")

    vwf_model = PyVWF(
        str(outdir),
        args.country,
        True,
        calc_z0=args.calc_z0,
        cluster_mode=args.cluster_mode,
        cluster_list=args.cluster_list,
        time_res_list=args.time_res_list,
        add_nan=args.add_nan,
        interp_nan=args.interp_nan,
        fix_turb=args.fix_turb,
    )
    vwf_model.train(args.train_plots)
    vwf_model.simulate_cf(args.year_test)


if __name__ == "__main__":
    main()
