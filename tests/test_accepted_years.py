"""The accepted-year rule for factors (issue #28).

A factor is the mean of per-year fits. It averages its scalar and its offset
over one set of years, the years whose offset was fitted and accepted, and is
refused unless that set holds at least two thirds of the training years. Each
test is built so the rule before #28 would give a different answer: a mean
over every year's scalar, a zero-observation year counted in, a partial set
averaged silently, or a cluster with no fit given the identity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vwf.data import format_bc_factors, min_accepted_years
from vwf.harness.corrections import fit_quality
from vwf.harness.driver import _record_accepted_years


def _fits(rows):
    """Per-year fits in the column order format_bc_factors expects."""
    return pd.DataFrame(
        rows, columns=["year", "time_slice", "cluster", "obs", "sim", "scalar", "offset"]
    )


def _one_cluster(years, obs, scalars, offsets):
    return _fits(
        [(y, "1/1", 0, o, 0.3, s, off) for y, o, s, off in zip(years, obs, scalars, offsets)]
    )


@pytest.mark.parametrize("n, need", [(1, 1), (2, 2), (3, 2), (4, 3), (5, 4), (6, 4)])
def test_the_minimum_is_two_thirds_rounded_up(n, need):
    assert min_accepted_years(n) == need


def test_every_year_accepted_gives_the_plain_mean():
    f = format_bc_factors(
        _one_cluster([2019, 2020, 2021], [0.3] * 3, [1.0, 1.2, 1.4], [0.1, 0.2, 0.3]), "fixed"
    )
    assert list(f.columns) == ["cluster", "fixed", "scalar", "offset", "n_years"]
    assert f.loc[0, "scalar"] == pytest.approx(1.2)
    assert f.loc[0, "offset"] == pytest.approx(0.2)
    assert f.loc[0, "n_years"] == 3


def test_scalar_and_offset_share_one_set_of_years():
    # 2021's offset was refused. The old rule averaged the scalar over all
    # three years (1.2) and the offset over two (0.15): a pair from two sets.
    f = format_bc_factors(
        _one_cluster([2019, 2020, 2021], [0.3] * 3, [1.0, 1.2, 1.4], [0.1, 0.2, np.nan]), "fixed"
    )
    assert f.loc[0, "scalar"] == pytest.approx(1.1)
    assert f.loc[0, "offset"] == pytest.approx(0.15)
    assert f.loc[0, "n_years"] == 2


def test_one_year_of_three_is_refused():
    # The US cluster 15 case: the old rule kept a factor from one year.
    f = format_bc_factors(
        _one_cluster([2019, 2020, 2021], [0.3] * 3, [2.9, 3.5, 3.4], [-8.4, np.nan, np.nan]),
        "fixed",
    )
    assert np.isnan(f.loc[0, "scalar"]) and np.isnan(f.loc[0, "offset"])
    assert f.loc[0, "n_years"] == 1


def test_a_zero_observation_year_is_not_a_fit():
    # A zero observation gets offset 0.0 without being fitted. The old rule
    # counted it, pulling the offset mean toward zero (0.2 over three years).
    f = format_bc_factors(
        _one_cluster([2019, 2020, 2021], [0.3, 0.3, 0.0], [1.0, 1.2, 0.0], [0.3, 0.3, 0.0]),
        "fixed",
    )
    assert f.loc[0, "offset"] == pytest.approx(0.3)
    assert f.loc[0, "scalar"] == pytest.approx(1.1)
    assert f.loc[0, "n_years"] == 2


def test_a_cluster_with_no_fit_is_refused_not_the_identity():
    # The old rule turned a NaN scalar into scalar 1, offset 0.
    f = format_bc_factors(
        _one_cluster([2019, 2020, 2021], [np.nan] * 3, [np.nan] * 3, [np.nan] * 3), "fixed"
    )
    assert np.isnan(f.loc[0, "scalar"]) and np.isnan(f.loc[0, "offset"])
    assert f.loc[0, "n_years"] == 0


def test_five_training_years_need_four():
    years = [2015, 2016, 2017, 2018, 2019]
    three = format_bc_factors(
        _one_cluster(years, [0.3] * 5, [1.0] * 5, [0.1, 0.1, 0.1, np.nan, np.nan]), "fixed"
    )
    four = format_bc_factors(
        _one_cluster(years, [0.3] * 5, [1.0] * 5, [0.1, 0.1, 0.1, 0.1, np.nan]), "fixed"
    )
    assert np.isnan(three.loc[0, "offset"])
    assert four.loc[0, "offset"] == pytest.approx(0.1)


def test_each_season_has_its_own_set():
    rows = [(y, "winter", 0, 0.3, 0.3, 1.0, 0.1) for y in (2019, 2020, 2021)]
    rows += [(2019, "summer", 0, 0.3, 0.3, 1.0, 0.1)]
    rows += [(y, "summer", 0, 0.3, 0.3, 1.0, np.nan) for y in (2020, 2021)]
    f = format_bc_factors(_fits(rows), "season").set_index("season")
    assert f.loc["winter", "n_years"] == 3 and f.loc["winter", "offset"] == pytest.approx(0.1)
    assert f.loc["summer", "n_years"] == 1 and np.isnan(f.loc["summer", "offset"])


def test_fit_quality_counts_a_refused_factor_as_failed():
    f = format_bc_factors(
        pd.concat(
            [
                _one_cluster([2019, 2020, 2021], [0.3] * 3, [1.0] * 3, [0.1] * 3),
                _one_cluster(
                    [2019, 2020, 2021], [0.3] * 3, [1.0] * 3, [0.1, np.nan, np.nan]
                ).assign(cluster=1),
            ],
            ignore_index=True,
        ),
        "fixed",
    )
    q = fit_quality(f)
    assert q["n_failed_offset"] == 1
    assert q["degenerate_clusters"] == "1"


def test_the_manifest_record_carries_every_count():
    f = format_bc_factors(
        pd.concat(
            [
                _one_cluster([2019, 2020, 2021], [0.3] * 3, [1.0] * 3, [0.1] * 3),
                _one_cluster([2019, 2020, 2021], [0.3] * 3, [1.0] * 3, [0.1, 0.1, np.nan]).assign(
                    cluster=1
                ),
                _one_cluster(
                    [2019, 2020, 2021], [0.3] * 3, [1.0] * 3, [0.1, np.nan, np.nan]
                ).assign(cluster=2),
            ],
            ignore_index=True,
        ),
        "fixed",
    )
    record = _record_accepted_years(f, "fixed", 3)
    assert record["training_years"] == 3 and record["min_accepted_years"] == 2
    assert record["per_factor"] == {"1/1": {"0": 3, "1": 2, "2": 1}}
    assert record["n_partial"] == 1 and record["n_refused"] == 1
