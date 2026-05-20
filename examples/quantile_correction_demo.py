"""Demonstrate distribution-aware bias correction vs. linear mean-matching.

This self-contained example uses *synthetic* data (no ERA5 / ENTSO-E needed) so
it runs anywhere. It builds a reanalysis-like wind series whose distribution is
biased the way ERA5 typically is - inflated variability and a mean offset -
then converts to capacity factor and compares three series against "observed":

    1. uncorrected
    2. linear correction   (alpha * w + beta, mean-matched: the PyVWF default)
    3. quantile mapping     (vwf.quantile_correction, the experimental method)

It prints a distributional skill table. The key result: the linear correction
matches the mean but leaves variance/tails biased, while quantile mapping
repairs the full distribution.

Run:
    python examples/quantile_correction_demo.py
"""
import sys
from pathlib import Path

import numpy as np

# Allow running directly (python examples/quantile_correction_demo.py) without
# an editable install by putting the repo root on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vwf.quantile_correction import empirical_quantile_mapping
from vwf.distribution_metrics import distribution_report


def synthetic_power_curve(speed):
    """Logistic-ish capacity-factor curve (cut-in 3, rated 13, cut-out 25)."""
    cut_in, rated, cut_out = 3.0, 13.0, 25.0
    cf = np.zeros_like(speed, dtype=float)
    ramp = (speed >= cut_in) & (speed < rated)
    cf[ramp] = ((speed[ramp] - cut_in) / (rated - cut_in)) ** 3
    cf[(speed >= rated) & (speed <= cut_out)] = 1.0
    return np.clip(cf, 0.0, 1.0)


def main():
    rng = np.random.default_rng(2024)
    n = 8760  # one year, hourly

    # "True" wind the turbine experiences (the observed world).
    obs_wind_train = np.clip(rng.weibull(2.0, n) * 8.0, 0, None)
    obs_wind_test = np.clip(rng.weibull(2.0, n) * 8.0, 0, None)

    # Reanalysis-like model wind: over-dispersed (x1.6 about its mean) + offset.
    def biased(w):
        m = w.mean()
        return np.clip(m + 1.2 + (w - m) * 1.6, 0, None)

    mod_wind_train = biased(obs_wind_train)
    mod_wind_test = biased(obs_wind_test)

    # --- Corrections (trained on the training period) ---------------------
    # Linear mean-matching in wind-speed space (alpha=1, beta = mean gap).
    beta = obs_wind_train.mean() - mod_wind_train.mean()
    lin_wind_test = mod_wind_test + beta

    # Quantile mapping in wind-speed space.
    qm_wind_test = empirical_quantile_mapping(
        mod_wind_train, obs_wind_train, mod_wind_test, n_quantiles=200
    )

    # --- Convert everything to capacity factor ----------------------------
    obs_cf = synthetic_power_curve(obs_wind_test)
    unc_cf = synthetic_power_curve(mod_wind_test)
    lin_cf = synthetic_power_curve(lin_wind_test)
    qm_cf = synthetic_power_curve(qm_wind_test)

    report = distribution_report(
        {"uncorrected": unc_cf, "linear": lin_cf, "quantile_mapping": qm_cf},
        obs_cf,
        quantiles=(0.1, 0.5, 0.9),
    )

    pretty = report[
        ["mean_bias", "std_ratio", "variance_ratio", "wasserstein",
         "ks", "q10_bias", "q90_bias", "ramp_std_ratio"]
    ].round(4)

    print("\nDistributional skill of capacity-factor corrections")
    print("(targets: mean_bias~0, std_ratio~1, variance_ratio~1, wasserstein~0)\n")
    print(pretty.to_string())
    print(
        "\nReading: linear correction removes the mean bias but the variance/"
        "tail\nratios stay off; quantile mapping drives variance_ratio and the "
        "tail\nbiases back toward their ideal values."
    )


if __name__ == "__main__":
    main()
