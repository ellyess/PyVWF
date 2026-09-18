"""Paired bootstrap intervals for skill metrics.

Several studies ask whether a difference in RMSE between two conditions of
the same row can be told apart from zero. They resample the test year, the
same way for every condition, and read percentile intervals off the draws:

- a **country-level** row has one national series, so its months are the
  resampling unit: :func:`resample_indices`, then :func:`rmse_over_rows`;
- a **turbine-level** row resamples its units: :func:`resample_counts` gives
  how often each unit is drawn, and :func:`weighted_rmse` and
  :func:`weighted_mean` weight each unit's summed errors by those counts.

Pairing comes from reuse: one set of indices or counts serves every condition
of a row. The seed and the number of draws are the caller's, because a study
fixes them in its pre-registration. For a given seed, ``n`` and ``n_draws``
the draws are exactly those of ``numpy.random.default_rng(seed).integers(0,
n, size=(n_draws, n))``, so an interval computed before this module existed
reproduces bit for bit.

The intervals understate the uncertainty: they are conditional on one test
year, months are not independent, and neighbouring units share weather. The
studies that use them say so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def resample_indices(n: int, *, seed: int, n_draws: int) -> np.ndarray:
    """Row indices for each draw: shape ``(n_draws, n)``, with replacement."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_draws, n))


def resample_counts(n: int, *, seed: int, n_draws: int) -> np.ndarray:
    """How often each of ``n`` units is drawn, per draw: shape ``(n_draws, n)``.

    The same draws as :func:`resample_indices`, counted per unit, as floats so
    they can weight sums directly.
    """
    draws = resample_indices(n, seed=seed, n_draws=n_draws)
    return np.stack([np.bincount(r, minlength=n) for r in draws]).astype(float)


def rmse_over_rows(errors: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """RMSE of ``errors`` under each draw of :func:`resample_indices`."""
    return np.sqrt((errors[indices] ** 2).mean(axis=1))


def weighted_rmse(counts: np.ndarray, squared: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Weighted RMSE under each draw of :func:`resample_counts`.

    Args:
        counts: Draw counts, ``(n_draws, n_units)``.
        squared: Per unit, the sum of ``weight * error**2`` over its rows.
        weight: Per unit, the sum of the weights over its rows.
    """
    return np.sqrt((counts @ squared) / (counts @ weight))


def weighted_mean(counts: np.ndarray, value: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Weighted mean under each draw: ``counts @ value / (counts @ weight)``."""
    return (counts @ value) / (counts @ weight)


def unit_sums(frame: pd.DataFrame, units: np.ndarray) -> pd.DataFrame:
    """Per-unit capacity weight ``w`` and weighted squared error ``e``.

    Units absent from ``frame``, for example those with no complete row in
    this condition, get zero weight, which is what the point metrics do.

    Args:
        frame: Rows with ``ID``, ``capacity``, ``cf_sim`` and ``cf_obs``.
        units: The unit order the draw counts follow.
    """
    g = frame.assign(w=frame["capacity"],
                     e=frame["capacity"] * (frame["cf_sim"] - frame["cf_obs"]) ** 2)
    return g.groupby("ID")[["w", "e"]].sum().reindex(units, fill_value=0.0)


def percentile_interval(draws: np.ndarray, q: tuple[float, float] = (2.5, 97.5)) -> tuple[float, float]:
    """The percentile interval of ``draws``; 95% by default.

    The percentiles are given as numbers rather than derived from a level,
    because ``(1 - 0.95) / 2 * 100`` is not exactly 2.5 in floating point.
    """
    lo, hi = np.percentile(draws, list(q))
    return float(lo), float(hi)
