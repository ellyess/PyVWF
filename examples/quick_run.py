"""
Quick run example for PyVWF (research workflow).

This script mirrors a typical research run:
1) Instantiate PyVWF with an output directory (created if missing)
2) Train the model(s)
3) Simulate capacity factor (CF) time series for a chosen year

The first PyVWF argument is the OUTPUT PATH where folders/files will be created.
"""

import argparse
from pathlib import Path

import vwf.vwf as model


def parse_args():
    p = argparse.ArgumentParser(description="PyVWF quick run example (train + simulate CF)")

    p.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory (PyVWF will create subfolders/files here).",
    )

    p.add_argument("--country", type=str, default="DK", help='Country code (e.g., "DK", "DE")')
    p.add_argument("--year-test", type=int, default=2020, help="Year to simulate (e.g., 2020)")

    p.add_argument(
        "--calc-z0",
        action="store_true",
        help="Enable z0 calculation (default: False)",
    )
    p.add_argument(
        "--cluster-mode",
        type=str,
        default="onshore",
        choices=["all", "onshore", "offshore"],
        help="Turbine subset used in clustering",
    )

    p.add_argument(
        "--cluster-list",
        type=int,
        nargs="*",
        default=[1, 2, 5, 7, 10]
        help="List of cluster counts to evaluate",
    )
    p.add_argument(
        "--time-res-list",
        type=str,
        nargs="*",
        default=["fixed", "season", "bimonth", "month"],
        help="Time-resolution modes to evaluate",
    )

    p.add_argument("--add-nan", type=float, default=None, help="Fraction/percent of data to remove (optional)")
    p.add_argument("--interp-nan", type=int, default=None, help="Monthly interpolation limit for NaNs (optional)")
    p.add_argument(
        "--fix-turb",
        type=str,
        default=None,
        help='Fix to a single turbine model/id (optional), e.g. "Vestas.V66.2000"',
    )

    # Mirrors your vwf_model.train(False)
    p.add_argument(
        "--train-plots",
        action="store_true",
        help="If set, enables any plotting/verbose outputs during training (maps to train(True)).",
    )

    return p.parse_args()


def main():
    args = parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[PyVWF] Output directory: {outdir}")

    vwf_model = model.PyVWF(
        str(outdir),          # OUTPUT PATH (folders created inside here)
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
