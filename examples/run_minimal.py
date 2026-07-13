"""Minimal, self-contained PyVWF example: runs end-to-end in under a minute on
bundled synthetic data, with no ERA5 download and no private turbine data.

It demonstrates the full core workflow:

  ERA5-shaped winds -> hub-height extrapolation -> per-cluster linear bias
  correction (w_corrected = scalar*w + offset) trained against observed
  capacity factors -> corrected simulation -> error reduction.

All bundled data under ``examples/data/`` is synthetic (see
``examples/data/README.md``); the shipped power curves are synthetic
placeholders too (see ``input/README.md``). Regenerate the data with
``python examples/data/generate_example_data.py``.

Run from the repository root:

    python examples/run_minimal.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from vwf.config import PyVWFPaths
from vwf.correction import calculate_scalar, find_offset
from vwf.datasets.era5 import prep_era5
from vwf.wind import train_simulate_wind

REPO = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    # 1. Point the ERA5 loader at the bundled synthetic NetCDF and prep it. This
    #    computes 100 m wind speed and derives surface roughness from the
    #    10 m/100 m shear, the inputs to hub-height extrapolation.
    PyVWFPaths.ERA5_DATA = DATA / "era5"
    reanalysis = prep_era5("example", train=False, calc_z0=True, bbox=None)

    turbines = pd.read_csv(DATA / "turbines_example.csv")
    observations = pd.read_csv(DATA / "observations_example.csv").set_index("cluster")
    power_curves = pd.read_csv(REPO / "input" / "power_curves.csv")

    rows = []
    for cluster, cl in turbines.groupby("cluster"):
        obs_cf = float(observations.loc[cluster, "obs_cf"])

        # Uncorrected simulated capacity factor for this cluster.
        sim_cf = train_simulate_wind(reanalysis, cl, power_curves, 1.0, 0.0)

        # Train the linear correction: scalar is the capacity-weighted obs/sim
        # ratio; offset is fit numerically so the corrected CF matches observed.
        bias = pd.DataFrame(
            {
                "fixed": ["1/1"],
                "cluster": [cluster],
                "year": [2020],
                "obs": [obs_cf],
                "sim": [sim_cf],
                "capacity": [cl["capacity"].sum()],
            }
        )
        scalar = float(calculate_scalar(bias, "fixed")["scalar"].iloc[0])
        row = pd.Series(
            {
                "obs": obs_cf,
                "sim": sim_cf,
                "scalar": scalar,
                "year": 2020,
                "cluster": cluster,
                "time_slice": "1/1",
            }
        )
        offset = float(find_offset(row, turbines, reanalysis, power_curves))

        corr_cf = train_simulate_wind(reanalysis, cl, power_curves, scalar, offset)
        rows.append(
            {
                "cluster": int(cluster),
                "observed": obs_cf,
                "uncorrected": sim_cf,
                "corrected": corr_cf,
                "scalar": scalar,
                "offset": offset,
            }
        )

    results = pd.DataFrame(rows)
    unc_mae = (results["uncorrected"] - results["observed"]).abs().mean()
    cor_mae = (results["corrected"] - results["observed"]).abs().mean()

    print("\nPer-cluster capacity factors (synthetic example):\n")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(
        f"\nMean |CF error| vs observed:  uncorrected {unc_mae:.4f}  ->  "
        f"corrected {cor_mae:.4f}"
    )
    if unc_mae > 0:
        print(
            f"The trained bias correction reduced the mean capacity-factor error "
            f"by {100 * (1 - cor_mae / unc_mae):.0f}%."
        )

    return unc_mae, cor_mae


if __name__ == "__main__":
    _unc_mae, _cor_mae = main()
    # Sanity guard (also used by CI): the trained correction must reduce error.
    assert _cor_mae < _unc_mae, (
        f"bias correction did not reduce error: {_cor_mae} !< {_unc_mae}"
    )
