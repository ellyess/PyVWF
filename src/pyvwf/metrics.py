"""Capacity-weighted means shared across the package.

:func:`weighted_mean` and :func:`weighted_mean_by` are imported by ten modules,
so that a missing value is treated the same way across them. Harness metrics
are computed in :mod:`pyvwf.harness.skill`.

Until 2026-10-28 this module also carried ``calculate_error``,
``prepare_monthly_data`` and ``weighted_average_vectorized``, used only by the
gridded correction, which was archived with ``pyvwf.extensions``; they are at
tag ``archive/research-2026-10-27``.
"""

import numpy as np


def weighted_mean(values, weights, *, axis=None):
    """The weighted mean of the entries that have both a value and a weight.

    Every weighted mean in the package goes through this function, so that a
    missing value is treated the same way everywhere. Two rules:

    - **A missing entry leaves both the sum and the weights.** Keeping its
      weight in the denominator would scale the result down by the missing
      share of weight, which is not a mean of anything.
    - **Where nothing has a value the result is NaN, never zero.** An empty
      numpy or pandas sum is 0.0, and a zero is a value: downstream code
      scores it instead of skipping it. That is how four UK stations came to
      be scored at zero output against an observed 0.38 for a whole year
      (:func:`pyvwf.harness.skill.collapse_pseudo_replicates`).

    A non-finite or non-positive total weight also gives NaN, so a group with
    no capacity behind it does not divide by zero.

    Args:
        values: Values to average. Array-like, or any shape broadcastable
            against ``weights``.
        weights: Weights, of the same shape as ``values`` or broadcastable to
            it (capacity, in most of this package).
        axis: Axis to reduce, as in numpy. ``None`` reduces everything and
            returns a float; an int returns an array over the other axes.

    Returns:
        float or numpy.ndarray: The weighted mean, NaN where nothing has a
        value and a weight.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    # numpy broadcasts w against v where the shapes differ; broadcasting it
    # explicitly first would change how the products associate in the sum, and
    # the results of runs already recorded would move in their last digit.
    present = np.isfinite(v) & np.isfinite(w)
    total = np.where(present, w, 0.0).sum(axis=axis)
    numerator = np.where(present, v * w, 0.0).sum(axis=axis)
    usable = total > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(usable, numerator / np.where(usable, total, 1.0), np.nan)
    return float(out) if axis is None else out


def weighted_mean_by(frame, value_col, weight_col, by):
    """:func:`weighted_mean` per group of ``frame``, as a Series indexed by ``by``.

    A thin wrapper: the rule lives in :func:`weighted_mean` and this only
    groups the rows.
    """
    grouped = frame.groupby(by, sort=True)
    return grouped.apply(
        lambda g: weighted_mean(g[value_col], g[weight_col]),
        include_groups=False,
    )
