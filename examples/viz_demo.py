"""Render the vwf.viz distributional diagnostics on synthetic data.

Builds a reanalysis-like over-dispersed wind series, applies a mean-matching
linear correction (the PyVWF default), converts to capacity factor with a
synthetic power curve, then draws the two diagnostic figures and saves them
under ``docs/img/``. No external data required.

Run:
    python examples/viz_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vwf.viz import plot_cf_distribution, plot_qq


def _power_curve(speed: np.ndarray) -> np.ndarray:
    cut_in, rated, cut_out = 3.0, 13.0, 25.0
    cf = np.zeros_like(speed, dtype=float)
    ramp = (speed >= cut_in) & (speed < rated)
    cf[ramp] = ((speed[ramp] - cut_in) / (rated - cut_in)) ** 3
    cf[(speed >= rated) & (speed <= cut_out)] = 1.0
    return np.clip(cf, 0.0, 1.0)


def main() -> None:
    rng = np.random.default_rng(2024)
    n = 8760

    obs_wind_train = np.clip(rng.weibull(2.0, n) * 8.0, 0, None)
    obs_wind_test = np.clip(rng.weibull(2.0, n) * 8.0, 0, None)

    def biased(w):
        m = w.mean()
        return np.clip(m + 1.2 + (w - m) * 1.6, 0, None)

    mod_wind_test = biased(obs_wind_test)

    # Linear correction: mean-match in wind-speed space (alpha=1, beta = mean gap),
    # which is the same form PyVWF's per-cluster correction uses.
    beta = obs_wind_train.mean() - biased(obs_wind_train).mean()
    lin_wind_test = mod_wind_test + beta

    series = {
        "uncorrected": _power_curve(mod_wind_test),
        "linear": _power_curve(lin_wind_test),
    }
    obs_cf = _power_curve(obs_wind_test)

    out_dir = Path(__file__).resolve().parent.parent / "docs" / "img"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1 = plot_cf_distribution(obs_cf, series)
    fig1.savefig(out_dir / "viz_distribution.png", dpi=150)
    print(f"  -> {out_dir / 'viz_distribution.png'}")

    fig2 = plot_qq(obs_cf, series)
    fig2.savefig(out_dir / "viz_qq.png", dpi=150)
    print(f"  -> {out_dir / 'viz_qq.png'}")


if __name__ == "__main__":
    main()
