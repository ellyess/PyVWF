"""Render the vwf.viz diagnostics on synthetic data.

Builds a reanalysis-like over-dispersed wind series, applies a mean-matching
linear correction (the PyVWF default), converts to capacity factor with a
synthetic power curve, then draws the diagnostic figures and saves them
under ``docs/img/``: the two distributional diagnostics, a map of learned
correction factors, and the error-vs-clusters model-selection plot. No
external data required.

Run:
    python examples/viz_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vwf.viz import (
    plot_cf_distribution,
    plot_correction_factor_map,
    plot_error_vs_clusters,
    plot_factor_joint,
    plot_qq,
    plot_sim_vs_obs,
)


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

    # --- Correction-factor map: synthetic fleet + per-cluster factors -----
    n_turb, n_clu = 80, 6
    fleet = pd.DataFrame({
        "lat": rng.uniform(55.0, 57.0, n_turb),
        "lon": rng.uniform(8.0, 10.0, n_turb),
    })
    factors = pd.DataFrame({
        "cluster": np.arange(n_clu),
        "scalar": rng.uniform(0.85, 1.15, n_clu),
        "offset": rng.uniform(-0.6, 0.6, n_clu),
    })
    from shapely.geometry import box

    fig3 = plot_correction_factor_map(
        factors, fleet, boundary=box(8.0, 55.0, 10.0, 57.0)
    )
    fig3.savefig(out_dir / "viz_factor_map.png", dpi=150)
    print(f"  -> {out_dir / 'viz_factor_map.png'}")

    # --- Factor joint distribution: seasonal factors, one point per slice --
    seasons = ["winter", "spring", "summer", "autumn"]
    seasonal_factors = pd.DataFrame({
        "cluster": np.repeat(np.arange(n_clu), len(seasons)),
        "season": seasons * n_clu,
        "scalar": rng.normal(0.97, 0.08, n_clu * len(seasons)),
        "offset": rng.normal(0.3, 0.35, n_clu * len(seasons)),
    })
    fig4 = plot_factor_joint(seasonal_factors)
    fig4.savefig(out_dir / "viz_factor_joint.png", dpi=150)
    print(f"  -> {out_dir / 'viz_factor_joint.png'}")

    # --- Per-turbine sim vs obs bias scatter -------------------------------
    ids = [f"T{i:03d}" for i in range(n_turb)]
    base = rng.uniform(0.20, 0.45, n_turb)
    obs_wide = pd.DataFrame(
        np.clip(base + rng.normal(0, 0.08, (365, n_turb)), 0, 1), columns=ids
    )
    sim_wide = pd.DataFrame(
        np.clip(obs_wide.to_numpy() + rng.normal(0.04, 0.02, (365, n_turb)), 0, 1),
        columns=ids,
    )
    turb_types = pd.DataFrame({
        "ID": ids,
        "type": rng.choice(["onshore", "offshore"], n_turb, p=[0.8, 0.2]),
    })
    fig5 = plot_sim_vs_obs(sim_wide, obs_wide, turb_info=turb_types)
    fig5.savefig(out_dir / "viz_sim_vs_obs.png", dpi=150)
    print(f"  -> {out_dir / 'viz_sim_vs_obs.png'}")

    # --- Error vs clusters: synthetic evaluation metrics ------------------
    rows = []
    for tres, base in [("fixed", 0.160), ("season", 0.152),
                       ("bimonth", 0.147), ("month", 0.143)]:
        for n in [1, 5, 10, 50, 100, 500, 1000]:
            err = base / (1 + 0.30 * np.log10(n + 1))
            rows.append({
                "correction_type": "corrected", "time_res": tres,
                "n_clusters": n,
                "rmse": err + rng.normal(0, 0.001),
                "mae": 0.8 * err + rng.normal(0, 0.001),
            })
    rows.append({
        "correction_type": "uncorrected", "time_res": None,
        "n_clusters": None, "rmse": 0.185, "mae": 0.150,
    })

    fig6 = plot_error_vs_clusters(pd.DataFrame(rows))
    fig6.savefig(out_dir / "viz_error_vs_clusters.png", dpi=150)
    print(f"  -> {out_dir / 'viz_error_vs_clusters.png'}")


if __name__ == "__main__":
    main()
