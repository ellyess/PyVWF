"""Distributional skill metrics for bias-correction evaluation.

:mod:`vwf.metrics` summarises agreement *in the mean* (MAE, RMSE, MBE on
time-aggregated capacity factors). Those metrics are blind to whether a
corrected series reproduces the observed *distribution* and *temporal
variability*. A correction that nails the monthly mean can still over- or
under-disperse hourly output, mis-place the calm/high-wind tails, or smear out
ramps - all of which propagate into energy-system results.

This module provides complementary, distribution-focused diagnostics:

* ``variance_ratio`` / ``std_ratio`` - dispersion bias (target 1.0).
* ``quantile_bias`` - bias at chosen quantiles (e.g. the P10 calm tail and the
  P90 high-wind tail).
* ``wasserstein`` - earth-mover distance between the full marginal
  distributions (target 0).
* ``ks_statistic`` - Kolmogorov-Smirnov distance (target 0).
* ``ramp_metrics`` - skill on the distribution of first differences
  (hour-to-hour ramps), which the mean metrics ignore entirely.
* ``distribution_report`` - one-call summary table bundling the above, suitable
  for comparing the linear vs quantile-mapping corrections side by side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "variance_ratio",
    "std_ratio",
    "quantile_bias",
    "wasserstein",
    "ks_statistic",
    "ramp_metrics",
    "distribution_report",
]


def _align(sim, obs):
    """Return finite-only sim/obs arrays (independently cleaned)."""
    s = np.asarray(sim, dtype=float).ravel()
    o = np.asarray(obs, dtype=float).ravel()
    return s[np.isfinite(s)], o[np.isfinite(o)]


def variance_ratio(sim, obs) -> float:
    """Ratio of simulated to observed variance (1.0 == unbiased dispersion)."""
    s, o = _align(sim, obs)
    ov = np.var(o)
    return float(np.var(s) / ov) if ov > 0 else np.nan


def std_ratio(sim, obs) -> float:
    """Ratio of simulated to observed standard deviation (1.0 == unbiased)."""
    s, o = _align(sim, obs)
    osd = np.std(o)
    return float(np.std(s) / osd) if osd > 0 else np.nan


def quantile_bias(sim, obs, quantiles=(0.1, 0.5, 0.9)) -> dict:
    """Signed bias ``q(sim) - q(obs)`` at each requested quantile."""
    s, o = _align(sim, obs)
    return {
        f"q{int(round(q * 100)):02d}_bias": float(np.quantile(s, q) - np.quantile(o, q))
        for q in quantiles
    }


def wasserstein(sim, obs) -> float:
    """1-Wasserstein (earth-mover) distance between marginal distributions."""
    s, o = _align(sim, obs)
    if s.size == 0 or o.size == 0:
        return np.nan
    return float(stats.wasserstein_distance(s, o))


def ks_statistic(sim, obs) -> float:
    """Two-sample Kolmogorov-Smirnov distance (max CDF gap)."""
    s, o = _align(sim, obs)
    if s.size == 0 or o.size == 0:
        return np.nan
    return float(stats.ks_2samp(s, o).statistic)


def ramp_metrics(sim, obs) -> dict:
    """Skill on the hour-to-hour ramp (first-difference) distribution.

    ``sim`` and ``obs`` are treated as time-ordered series. Returns the ratio of
    ramp standard deviations (dispersion of ramps) and the Wasserstein distance
    between the two ramp distributions.
    """
    s = np.asarray(sim, dtype=float).ravel()
    o = np.asarray(obs, dtype=float).ravel()
    ds = np.diff(s)
    do = np.diff(o)
    ds = ds[np.isfinite(ds)]
    do = do[np.isfinite(do)]
    if ds.size == 0 or do.size == 0:
        return {"ramp_std_ratio": np.nan, "ramp_wasserstein": np.nan}
    osd = np.std(do)
    return {
        "ramp_std_ratio": float(np.std(ds) / osd) if osd > 0 else np.nan,
        "ramp_wasserstein": float(stats.wasserstein_distance(ds, do)),
    }


def distribution_report(series: dict, obs, quantiles=(0.1, 0.5, 0.9)) -> pd.DataFrame:
    """Build a comparison table of distributional metrics.

    Args:
        series: Mapping of label -> simulated/corrected time series (e.g.
            ``{"uncorrected": cf_unc, "linear": cf_lin, "quantile": cf_qm}``).
        obs: Observed reference time series.
        quantiles: Quantiles at which to report bias.

    Returns:
        DataFrame indexed by label with columns: ``mean_bias``, ``rmse``,
        ``std_ratio``, ``variance_ratio``, ``wasserstein``, ``ks``,
        per-quantile biases, and ramp metrics.
    """
    rows = {}
    for label, sim in series.items():
        s, o = _align(sim, obs)
        n = min(s.size, o.size)
        row = {
            "mean_bias": float(np.mean(s) - np.mean(o)) if n else np.nan,
            "rmse": _paired_rmse(sim, obs),
            "std_ratio": std_ratio(sim, obs),
            "variance_ratio": variance_ratio(sim, obs),
            "wasserstein": wasserstein(sim, obs),
            "ks": ks_statistic(sim, obs),
        }
        row.update(quantile_bias(sim, obs, quantiles))
        row.update(ramp_metrics(sim, obs))
        rows[label] = row
    return pd.DataFrame(rows).T


def _paired_rmse(sim, obs) -> float:
    """RMSE over the time-aligned overlap of two equal-length series."""
    s = np.asarray(sim, dtype=float).ravel()
    o = np.asarray(obs, dtype=float).ravel()
    n = min(s.size, o.size)
    if n == 0:
        return np.nan
    d = s[:n] - o[:n]
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d**2))) if d.size else np.nan
