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


def _capacity_gradient(n=21):
    """Evenly spaced sites on a line, with capacity concentrated at one end.

    Two far-apart blobs would NOT discriminate: the obvious split is the same
    weighted or not. A contiguous line does, because weighting decides where
    the boundary falls rather than whether there is one.
    """
    return pd.DataFrame({
        "ID": [str(i) for i in range(n)],
        "lat": np.linspace(0.0, 20.0, n),
        "lon": np.zeros(n),
        "capacity": [1.0] * (n // 2 + 1) + [100.0] * (n // 2),
    })


def _lowest_lat_of_top_cluster(df, labels):
    """Latitude at which the highest cluster begins — i.e. the split point."""
    top = labels[np.argmax(df["lat"].to_numpy())]
    return df["lat"].to_numpy()[labels == top].min()


def test_capacity_weighting_moves_the_boundary_toward_capacity():
    """Must-distinguish: weighting must move the split, in the right direction.

    Capacity sits at the high-latitude end, so the weighted fit should spend
    its clusters there and push the boundary up. A no-op implementation leaves
    the boundary where it was and fails.
    """
    df = _capacity_gradient()
    u = cluster_turbines(2, df.copy(), True)["cluster"].to_numpy()
    w = cluster_turbines(2, df.copy(), True, weight_col="capacity")["cluster"].to_numpy()

    u_split = _lowest_lat_of_top_cluster(df, u)
    w_split = _lowest_lat_of_top_cluster(df, w)
    assert w_split > u_split, (
        f"weighted split {w_split} should sit above unweighted {u_split}; "
        "capacity weighting had no effect"
    )


def test_capacity_weighting_falls_back_when_weights_unusable():
    df = _capacity_gradient()
    df.loc[0, "capacity"] = np.nan
    with pytest.warns(UserWarning, match="falling back to an unweighted fit"):
        out = cluster_turbines(2, df.copy(), True, weight_col="capacity")
    baseline = cluster_turbines(2, df.copy(), True)
    assert np.array_equal(
        pd.factorize(out["cluster"])[0], pd.factorize(baseline["cluster"])[0]
    )


def test_missing_weight_column_is_not_an_error():
    df = _capacity_gradient().drop(columns=["capacity"])
    out = cluster_turbines(2, df.copy(), True, weight_col="capacity")
    assert "cluster" in out.columns


def _square_km_grid(lat0=50.0, half_km=60.0, n=7):
    """Points on a grid that is SQUARE in kilometres, centred at lat0.

    At 50N a degree of longitude is ~71 km against ~111 km for latitude, so a
    square in kilometres spans MORE degrees of longitude than of latitude
    (1.677 vs 1.085). In degree space it therefore looks like a wide rectangle,
    and k-means splits it across longitude — an artefact of the units, since
    the region is square on the ground.
    """
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * np.cos(np.radians(lat0))
    offs = np.linspace(-half_km, half_km, n)
    lats, lons = [], []
    for dy in offs:
        for dx in offs:
            lats.append(lat0 + dy / km_per_deg_lat)
            lons.append(dx / km_per_deg_lon)
    return pd.DataFrame({
        "ID": [str(i) for i in range(len(lats))],
        "lat": lats,
        "lon": lons,
    })


def _split_axis(df, labels):
    """Whether a 2-cluster split separates points by latitude or longitude."""
    a, b = (df[labels == c] for c in np.unique(labels))
    return "lat" if abs(a["lat"].mean() - b["lat"].mean()) > abs(
        a["lon"].mean() - b["lon"].mean()
    ) else "lon"


def test_degree_space_splits_a_square_along_the_wrong_axis():
    """Must-distinguish: the units, not the geography, decide the split.

    The fixture is square on the ground. At 50N degree space stretches it in
    longitude (1.677 deg vs 1.085 deg), so k=2 cuts it across LONGITUDE — a
    pure artefact. Clustering on the sphere removes the stretch, and the two
    partitions must differ, which a no-op transform could not produce.
    """
    df = _square_km_grid(lat0=50.0)
    degrees = cluster_turbines(2, df.copy(), True)["cluster"].to_numpy()
    geographic = cluster_turbines(2, df.copy(), True, geographic=True)["cluster"].to_numpy()

    assert _split_axis(df, degrees) == "lon", (
        "fixture is not exercising the degree-space distortion"
    )
    assert _split_axis(df, geographic) == "lat", (
        "geographic fit still split along the degree-stretched axis"
    )
    assert not np.array_equal(
        pd.factorize(degrees)[0], pd.factorize(geographic)[0]
    ), "geographic projection did not change the partition"


def test_the_two_conventions_agree_where_there_is_no_distortion():
    """Sanity: at the equator a degree of lon ~= a degree of lat.

    With no stretch to correct, both conventions must choose the same split
    axis. Disagreeing here would mean the transform is wrong rather than
    merely different.
    """
    df = _square_km_grid(lat0=0.0)
    degrees = cluster_turbines(2, df.copy(), True)["cluster"].to_numpy()
    geographic = cluster_turbines(2, df.copy(), True, geographic=True)["cluster"].to_numpy()
    assert _split_axis(df, degrees) == _split_axis(df, geographic)


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
