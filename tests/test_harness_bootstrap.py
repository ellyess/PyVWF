"""The paired bootstrap (vwf.harness.bootstrap) matches the code it replaced.

Five study scripts carried their own copy of these few lines before they were
promoted. Every interval those scripts recorded depends on the draws being
exactly ``default_rng(seed).integers(0, n, size=(n_draws, n))``, so each
function here is compared with the inline expression it replaced, with exact
equality. The recorded outputs themselves are pinned in
tests/test_pin_bootstrap_reproduction.py, which needs local data.
"""

import numpy as np
import pandas as pd
import pytest

from vwf.harness import bootstrap as bs

SEED, N_DRAWS = 20260911, 1000


def test_resample_indices_are_the_inline_draws():
    want = np.random.default_rng(SEED).integers(0, 12, size=(N_DRAWS, 12))
    np.testing.assert_array_equal(bs.resample_indices(12, seed=SEED, n_draws=N_DRAWS), want)


def test_resample_counts_are_the_inline_counts():
    draws = np.random.default_rng(SEED).integers(0, 37, size=(N_DRAWS, 37))
    want = np.stack([np.bincount(r, minlength=37) for r in draws]).astype(float)
    got = bs.resample_counts(37, seed=SEED, n_draws=N_DRAWS)
    np.testing.assert_array_equal(got, want)
    assert got.dtype == float and (got.sum(axis=1) == 37).all()


def test_rmse_over_rows_is_the_inline_expression():
    d = np.random.default_rng(1).normal(size=12)
    idx = bs.resample_indices(12, seed=SEED, n_draws=N_DRAWS)
    np.testing.assert_array_equal(bs.rmse_over_rows(d, idx), np.sqrt((d[idx] ** 2).mean(axis=1)))


def test_weighted_statistics_are_the_inline_expressions():
    rng = np.random.default_rng(2)
    e, a, w = rng.random(37), rng.random(37), rng.random(37) + 0.5
    counts = bs.resample_counts(37, seed=SEED, n_draws=N_DRAWS)
    np.testing.assert_array_equal(
        bs.weighted_rmse(counts, e, w), np.sqrt((counts @ e) / (counts @ w))
    )
    np.testing.assert_array_equal(bs.weighted_mean(counts, a, w), (counts @ a) / (counts @ w))


def test_unit_sums_weight_missing_units_zero():
    frame = pd.DataFrame(
        {
            "ID": ["a", "a", "b"],
            "capacity": [2.0, 2.0, 1.0],
            "cf_sim": [0.5, 0.3, 0.2],
            "cf_obs": [0.4, 0.4, 0.1],
        }
    )
    got = bs.unit_sums(frame, np.array(["a", "b", "c"]))
    assert got.index.tolist() == ["a", "b", "c"]
    assert got.loc["a", "w"] == 4.0 and got.loc["c", "w"] == 0.0
    assert got.loc["a", "e"] == pytest.approx(2 * 0.1**2 + 2 * 0.1**2)


def test_percentile_interval_uses_literal_percentiles():
    x = np.random.default_rng(3).normal(size=N_DRAWS)
    lo, hi = np.percentile(x, [2.5, 97.5])
    assert bs.percentile_interval(x) == (float(lo), float(hi))
    # The reason the percentiles are literal: derived from a level they are not 2.5.
    assert (1 - 0.95) / 2 * 100 != 2.5
