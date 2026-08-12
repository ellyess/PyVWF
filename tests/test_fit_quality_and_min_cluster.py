"""Guards for the two fit-robustness features.

Both exist because of the same incident: a Chilean cluster of ONE plant fitted a
wind scalar of 80 with an offset that never converged, and the monthly skill
metric still reported the region as a corrected win. See
docs/findings/hourly_resolution_test.md.

- ``fit_quality`` makes such a fit visible.
- ``min_cluster_size`` stops a one-plant cluster being fitted in the first place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vwf.clustering import cluster_turbines
from vwf.harness.corrections import PLAUSIBLE_SCALAR, fit_quality


def _factors(scalars, offsets=None):
    n = len(scalars)
    return pd.DataFrame({
        "cluster": range(n),
        "fixed": ["1/1"] * n,
        "scalar": scalars,
        "offset": offsets if offsets is not None else [0.0] * n,
    })


def test_clean_fit_flags_nothing():
    q = fit_quality(_factors([0.8, 1.0, 1.4]))
    assert q["n_implausible_scalar"] == 0
    assert q["n_failed_offset"] == 0
    assert q["degenerate_clusters"] == ""
    assert q["n_clusters"] == 3


def test_implausible_scalar_is_flagged():
    q = fit_quality(_factors([1.0, 80.2, 0.9]))
    assert q["n_implausible_scalar"] == 1
    assert q["degenerate_clusters"] == "1"
    assert q["max_scalar"] == pytest.approx(80.2)


def test_scalar_below_the_lower_bound_is_flagged_too():
    low = PLAUSIBLE_SCALAR[0]
    q = fit_quality(_factors([low / 2, 1.0]))
    assert q["n_implausible_scalar"] == 1
    assert q["degenerate_clusters"] == "0"


def test_failed_offset_is_flagged_even_with_a_sane_scalar():
    """A NaN offset means the solver failed; those sites vanish from a run."""
    q = fit_quality(_factors([1.0, 1.1], offsets=[0.2, np.nan]))
    assert q["n_failed_offset"] == 1
    assert q["n_implausible_scalar"] == 0
    assert q["degenerate_clusters"] == "1"


def test_empty_factors_do_not_raise():
    q = fit_quality(pd.DataFrame())
    assert q["n_clusters"] == 0 and q["degenerate_clusters"] == ""


# --- min_cluster_size ------------------------------------------------------

def _fleet(seed=0, n=40):
    rng = np.random.default_rng(seed)
    # Three tight blobs plus two far outliers that k-means will isolate.
    core = np.vstack([
        rng.normal([10.0, 55.0], 0.15, size=(n // 3, 2)),
        rng.normal([11.0, 56.0], 0.15, size=(n // 3, 2)),
        rng.normal([12.0, 57.0], 0.15, size=(n // 3, 2)),
    ])
    outliers = np.array([[20.0, 62.0], [21.5, 63.0]])
    xy = np.vstack([core, outliers])
    return pd.DataFrame({
        "ID": [str(i) for i in range(len(xy))],
        "lon": xy[:, 0], "lat": xy[:, 1],
        "capacity": 1000.0, "height": 100.0,
    })


def test_default_is_a_no_op():
    """Default must reproduce the legacy partition exactly."""
    a = cluster_turbines(6, _fleet(), True)
    b = cluster_turbines(6, _fleet(), True, min_cluster_size=1)
    assert (a["cluster"].to_numpy() == b["cluster"].to_numpy()).all()


def test_undersized_clusters_are_merged_away():
    merged = cluster_turbines(6, _fleet(), True, min_cluster_size=4)
    sizes = merged.groupby("cluster").size()
    assert sizes.min() >= 4, f"undersized cluster survived: {sizes.to_dict()}"
    assert merged["cluster"].nunique() < 6, "nothing was merged"


def test_merging_never_loses_a_site():
    fleet = _fleet()
    merged = cluster_turbines(6, fleet.copy(), True, min_cluster_size=4)
    assert len(merged) == len(fleet)
    assert merged["cluster"].notna().all()


def test_train_and_apply_agree_on_labels():
    """Apply-side labels must match the training partition, or factors misalign."""
    train = _fleet()
    trained = cluster_turbines(6, train.copy(), True, min_cluster_size=4)
    applied = cluster_turbines(6, train.copy(), False, train.copy(), min_cluster_size=4)
    assert (trained["cluster"].to_numpy() == applied["cluster"].to_numpy()).all()


def test_min_cluster_size_larger_than_fleet_collapses_to_one():
    merged = cluster_turbines(6, _fleet(), True, min_cluster_size=10_000)
    assert merged["cluster"].nunique() == 1
