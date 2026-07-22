"""Spatial clustering: the partition must not be a lottery.

`cluster_turbines` feeds every downstream correction factor, so if its
partition depends on the KMeans seed then held-out skill does too. Measured on
the real DK fleet at a FIXED k=500, varying only the seed moved MAE across
0.0514-0.0842 — a wider range than the entire k=10..1000 curve. The k-sweeps
were measuring which local optimum KMeans fell into, not k.

These tests pin the property that matters: same data, same k, same partition.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from vwf.clustering import cluster_turbines

SEEDS = (0, 1, 7, 42)


def _blobs(n_blobs=40, per_blob=5, spread=0.02, seed=0):
    """Tight, well-separated blobs — the partition is unambiguous.

    Any competent initialisation recovers exactly these blobs, so a seed
    dependence here is initialisation luck, not genuine ambiguity in the data.
    """
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-10.0, 10.0, (n_blobs, 2))
    pts, truth = [], []
    for i, c in enumerate(centres):
        pts.append(c + rng.normal(0.0, spread, (per_blob, 2)))
        truth.extend([i] * per_blob)
    xy = np.vstack(pts)
    df = pd.DataFrame({
        "ID": [str(i) for i in range(len(xy))],
        "lat": xy[:, 0],
        "lon": xy[:, 1],
    })
    return df, np.asarray(truth)


def _labels_for_seed(df, k, seed):
    return cluster_turbines(k, df.copy(), True, random_state=seed)["cluster"].to_numpy()


def test_partition_is_stable_across_seeds():
    """Same points, same k, different seed -> the same partition.

    With ``init="random"`` this fails at ARI ~0.68: KMeans settles into a
    different local optimum per seed, and ``n_init=10`` picks among them by
    INERTIA, which is not the quantity we care about.
    """
    df, _ = _blobs()
    labels = [_labels_for_seed(df, 40, s) for s in SEEDS]
    agreement = [adjusted_rand_score(labels[0], other) for other in labels[1:]]
    assert min(agreement) > 0.99, (
        f"partition depends on the KMeans seed (min ARI {min(agreement):.4f}); "
        "held-out skill will inherit that noise"
    )


def test_partition_recovers_the_true_blobs():
    """Must-distinguish: stability alone is not enough.

    A degenerate initialisation that always collapsed every point into one
    cluster would be perfectly seed-stable and perfectly useless. Pin that the
    stable partition is also the CORRECT one.
    """
    df, truth = _blobs()
    labels = _labels_for_seed(df, 40, 42)
    assert adjusted_rand_score(truth, labels) > 0.99
    assert len(np.unique(labels)) == 40  # not collapsed


@pytest.mark.parametrize("k", [10, 40])
def test_predict_path_is_stable_across_seeds(k):
    """The evaluate-time path (fit on train fleet, predict on another) too.

    `run_evaluate` refits the clustering and predicts labels for the
    evaluation fleet, then joins factors on those labels. If that refit is
    seed-dependent, factors attach to the wrong turbines.

    The fixture uses ``n_blobs == k`` deliberately. Asking for FEWER clusters
    than there are blobs is genuinely ambiguous — which blobs share a cluster
    is arbitrary, so seeds may legitimately disagree and a stability assertion
    would be testing the data, not the initialisation.
    """
    train_df, _ = _blobs(n_blobs=k)
    target_df, _ = _blobs(n_blobs=k, seed=1)
    out = [
        cluster_turbines(k, train_df.copy(), False, target_df.copy(), random_state=s)[
            "cluster"
        ].to_numpy()
        for s in SEEDS
    ]
    agreement = [adjusted_rand_score(out[0], other) for other in out[1:]]
    assert min(agreement) > 0.99
