"""One rule for every weighted mean in the package (`vwf.metrics.weighted_mean`).

Each weighted mean used to be written by hand, and they did not agree on what
a missing value means. Two cases decide it, and both are tested here against
the arithmetic the old hand-written forms produced:

- **Partial:** a missing entry leaves the sum AND the weights. Keeping its
  weight in the denominator scales the result down by the missing share.
- **Empty:** where nothing has a value the result is NaN, never zero, because
  an empty numpy or pandas sum is 0.0 and a zero is a value that downstream
  code scores instead of skipping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vwf.metrics import weighted_average_vectorized, weighted_mean, weighted_mean_by


def test_the_plain_case_is_the_weighted_mean():
    assert weighted_mean([0.2, 0.4, 0.4], [2.0, 1.0, 1.0]) == pytest.approx(0.3)
    # ...and not the unweighted mean, so the fixture can tell them apart.
    assert weighted_mean([0.2, 0.4, 0.4], [2.0, 1.0, 1.0]) != pytest.approx(1 / 3)


def test_partial_a_missing_value_leaves_the_weights_too():
    # The remaining row carries 0.4. Dividing by the whole weight, as the
    # hand-written forms did, would give 0.4 * 1 / 4 = 0.1.
    got = weighted_mean([np.nan, np.nan, 0.4], [2.0, 1.0, 1.0])
    assert got == pytest.approx(0.4)
    assert got != pytest.approx(0.1)


def test_partial_a_missing_weight_drops_its_value():
    # A value with no weight cannot be weighted, so it leaves both sides.
    assert weighted_mean([0.2, 0.8], [1.0, np.nan]) == pytest.approx(0.2)


def test_empty_is_nan_not_zero():
    assert np.isnan(weighted_mean([np.nan, np.nan], [2.0, 1.0]))
    assert np.isnan(weighted_mean([], []))
    # The old idiom returned 0.0 here, which downstream code scores as output.
    assert float(np.nansum(np.array([np.nan, np.nan]) * np.array([2.0, 1.0]))) == 0.0


def test_zero_or_negative_total_weight_is_nan_not_a_division():
    assert np.isnan(weighted_mean([0.3, 0.5], [0.0, 0.0]))
    assert np.isnan(weighted_mean([0.3, 0.5], [-1.0, -1.0]))


def test_an_axis_reduces_row_by_row():
    values = np.array([[1.0, np.nan], [2.0, 4.0], [np.nan, np.nan]])
    weights = np.array([1.0, 3.0])
    got = weighted_mean(values, weights, axis=1)
    assert got[0] == pytest.approx(1.0)  # only the first column has a value
    assert got[1] == pytest.approx((2.0 * 1 + 4.0 * 3) / 4)
    assert np.isnan(got[2])


def test_weights_broadcast_against_a_wide_frame():
    values = np.array([[0.2, 0.4], [0.6, 0.8]])
    got = weighted_mean(values, [3.0, 1.0], axis=1)
    assert got == pytest.approx([(0.2 * 3 + 0.4) / 4, (0.6 * 3 + 0.8) / 4])


def test_grouped_wrapper_applies_the_same_rule():
    frame = pd.DataFrame(
        {
            "g": ["a", "a", "b", "b"],
            "v": [0.2, np.nan, np.nan, np.nan],
            "w": [1.0, 9.0, 1.0, 1.0],
        }
    )
    got = weighted_mean_by(frame, "v", "w", "g")
    assert got["a"] == pytest.approx(0.2)  # not 0.2 * 1 / 10
    assert np.isnan(got["b"])  # not 0.0


def test_the_legacy_name_delegates():
    frame = pd.DataFrame({"v": [np.nan, 0.4], "w": [3.0, 1.0]})
    assert weighted_average_vectorized(frame, "v", "w") == pytest.approx(0.4)
