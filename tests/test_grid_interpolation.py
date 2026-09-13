"""Spatial interpolation of correction factors (src/vwf/extensions/grid/interpolation.py).

Ported from a driver script on the `development` branch, where it carried no
tests. Two registered studies use these as their single definition, so the
tests that matter most are the two at the end: that the port reproduces the
chapter's own published cross-validation scores, and that the grid-wise and
point-wise routes cannot disagree because there is only one implementation.
"""
import numpy as np
import pandas as pd
import pytest

from vwf.extensions.grid import interpolation as interp


def points(lons=(0.0, 1.0, 2.0), lats=(50.0, 50.0, 50.0),
           scalar=(1.0, 2.0, 3.0), offset=(0.0, 1.0, 2.0)):
    return pd.DataFrame({"lon": list(lons), "lat": list(lats),
                         "scalar": list(scalar), "offset": list(offset)})


def test_a_target_on_a_control_point_takes_its_value():
    s, o = interp.idw_at(points(), [1.0], [50.0])
    assert s[0] == 2.0 and o[0] == 1.0


def test_the_exact_match_rule_holds_past_the_first_batch():
    """The original corrected exact matches only in the first batch of 10,000
    targets, because it sliced a per-batch mask with the global offset. A
    control point repeated past that boundary is where it showed."""
    targets_lon = [9.9] * 10_001 + [1.0]
    targets_lat = [50.0] * 10_002
    s, o = interp.idw_at(points(), targets_lon, targets_lat)
    assert s[-1] == 2.0 and o[-1] == 1.0
    assert len(s) == 10_002


def test_interpolation_between_control_points_is_bounded_by_them():
    s, _ = interp.idw_at(points(), [0.5, 1.5], [50.0, 50.0])
    assert 1.0 < s[0] < 2.0 and 2.0 < s[1] < 3.0


def test_restricting_to_the_k_nearest_changes_the_answer():
    far = points(lons=(0.0, 1.0, 40.0), scalar=(1.0, 2.0, 99.0))
    everything = interp.idw_at(far, [0.5], [50.0])[0][0]
    two_nearest = interp.idw_at(far, [0.5], [50.0], k=2)[0][0]
    assert everything != two_nearest
    assert 1.0 < two_nearest < 2.0


def test_nearest_neighbour_assigns_a_whole_control_point():
    s, o = interp.nearest_at(points(), [1.9, 0.1], [50.0, 50.0])
    assert list(s) == [3.0, 1.0] and list(o) == [2.0, 0.0]


def test_distance_is_euclidean_in_degrees_not_great_circle():
    """The chapter's choice, reproduced deliberately: one degree of longitude
    counts the same as one of latitude, which it is not in kilometres."""
    d = interp.degree_distances(np.array([[0.0, 60.0]]), np.array([[1.0, 60.0], [0.0, 61.0]]))
    assert d[0, 0] == pytest.approx(1.0) and d[0, 1] == pytest.approx(1.0)


def test_distance_to_nearest_reports_degrees():
    got = interp.distance_to_nearest(points(), [0.0, 5.0], [50.0, 50.0])
    assert got[0] == pytest.approx(0.0) and got[1] == pytest.approx(3.0)


def test_a_grid_comes_back_shaped_by_latitude_then_longitude():
    s, o = interp.to_grid(interp.idw_at, points(), [0.0, 1.0, 2.0], [50.0, 51.0])
    assert s.shape == (2, 3) and o.shape == (2, 3)


def test_the_grid_route_and_the_point_route_agree_because_they_are_one():
    grid_s, _ = interp.to_grid(interp.idw_at, points(), [0.3, 1.7], [50.0])
    point_s, _ = interp.idw_at(points(), [0.3, 1.7], [50.0, 50.0])
    assert np.allclose(grid_s.ravel(), point_s)


def test_rbf_runs_and_is_exact_at_its_control_points():
    s, _ = interp.rbf_at(points(lons=(0.0, 1.0, 2.0, 3.0), lats=(50.0, 50.5, 51.0, 50.2),
                                scalar=(1.0, 2.0, 3.0, 4.0), offset=(0.0, 1.0, 2.0, 3.0)),
                         [1.0], [50.5])
    assert s[0] == pytest.approx(2.0, abs=1e-6)


def test_kriging_needs_its_extra_and_returns_a_value_per_target():
    pytest.importorskip("pykrige", reason="kriging is in the 'grid' extra")
    frame = points(lons=(0.0, 1.0, 2.0, 3.0, 1.5), lats=(50.0, 50.5, 51.0, 50.2, 50.8),
                   scalar=(1.0, 2.0, 3.0, 4.0, 2.5), offset=(0.0, 1.0, 2.0, 3.0, 1.5))
    (s, o) = interp.kriging_at(frame, [1.2, 2.2], [50.4, 50.6])
    assert len(s) == 2 and len(o) == 2 and np.isfinite(s).all()


def test_kriging_can_return_its_variance_for_the_mask():
    pytest.importorskip("pykrige", reason="kriging is in the 'grid' extra")
    frame = points(lons=(0.0, 1.0, 2.0, 3.0, 1.5), lats=(50.0, 50.5, 51.0, 50.2, 50.8),
                   scalar=(1.0, 2.0, 3.0, 4.0, 2.5), offset=(0.0, 1.0, 2.0, 3.0, 1.5))
    (s, _), (var_s, _) = interp.kriging_at(frame, [1.2], [50.4], with_variance=True)
    assert len(s) == 1 and len(var_s) == 1 and var_s[0] >= 0


@pytest.mark.parametrize("fn", [interp.idw_at, interp.nearest_at, interp.rbf_at])
def test_a_frame_missing_a_required_column_is_refused(fn):
    with pytest.raises(ValueError, match="missing"):
        fn(points().drop(columns=["offset"]), [0.5], [50.0])


@pytest.mark.parametrize("fn", [interp.idw_at, interp.nearest_at])
def test_an_empty_control_set_is_refused(fn):
    with pytest.raises(ValueError, match="empty"):
        fn(points().iloc[0:0], [0.5], [50.0])


def test_it_reproduces_the_chapter_s_published_cross_validation(tmp_path):
    """The check that matters: the same control points, the same folds and the
    same arithmetic as thesis chapter 4, against the scores it published.

    Skipped where the research outputs are absent, which is CI. The data is
    git-ignored, so this is a local regression check on the port rather than a
    gate.
    """
    from pathlib import Path
    pool = Path("output/pyvwf_to_grid/all_corrections_centroids.csv")
    scores = Path("output/pyvwf_to_grid/grid_comparison/cv_scores.csv")
    if not (pool.exists() and scores.exists()):
        pytest.skip("chapter 4 outputs are not present")

    d = pd.read_csv(pool)
    published = pd.read_csv(scores, index_col=0).loc["idw"]
    order = np.argsort(d["lon"].to_numpy(), kind="stable")
    n, size = len(d), len(d) // 5
    scalar_mae, offset_mae = [], []
    for i in range(5):
        a, b = i * size, ((i + 1) * size if i < 4 else n)
        train = d.iloc[np.concatenate([order[:a], order[b:]])]
        test = d.iloc[order[a:b]]
        s, o = interp.idw_at(train, test["lon"].to_numpy(), test["lat"].to_numpy())
        scalar_mae.append(np.abs(s - test["scalar"].to_numpy()).mean())
        offset_mae.append(np.abs(o - test["offset"].to_numpy()).mean())

    assert np.mean(scalar_mae) == pytest.approx(published["scalar_mae_mean"], abs=1e-12)
    assert np.mean(offset_mae) == pytest.approx(published["offset_mae_mean"], abs=1e-12)


def test_the_two_distance_metrics_differ_where_longitude_is_short():
    """One degree of longitude is about half a degree of latitude in km at 60
    north. Euclidean degrees cannot see that; great circle can."""
    targets = np.array([[0.0, 60.0]])
    coords = np.array([[1.0, 60.0], [0.0, 61.0]])
    degrees = interp.degree_distances(targets, coords, "degrees")
    km = interp.degree_distances(targets, coords, "great_circle")
    assert degrees[0, 0] == pytest.approx(degrees[0, 1])
    assert km[0, 0] == pytest.approx(55.6, abs=1.0)
    assert km[0, 1] == pytest.approx(111.2, abs=1.0)


def test_the_metric_defaults_to_the_chapter_s_and_must_be_asked_for():
    assert interp.DEFAULT_METRIC == "degrees"
    frame = points(lons=(0.0, 1.0), lats=(60.0, 60.0), scalar=(1.0, 2.0), offset=(0.0, 1.0))
    same = interp.idw_at(frame, [0.5], [60.5])[0][0]
    assert interp.idw_at(frame, [0.5], [60.5], metric="degrees")[0][0] == same


def test_a_metric_that_is_not_offered_is_refused():
    with pytest.raises(ValueError, match="unknown metric"):
        interp.degree_distances(np.array([[0.0, 50.0]]), np.array([[1.0, 50.0]]), "manhattan")
    with pytest.raises(ValueError, match="unknown metric"):
        interp.idw_at(points(), [0.5], [50.0], metric="manhattan")


def test_the_metric_reaches_nearest_neighbour_and_the_distance_report():
    """At 60 north a point one degree east is nearer in km than one degree
    north, and the two metrics disagree about which control point wins."""
    frame = points(lons=(1.05, 0.0), lats=(60.0, 61.0), scalar=(1.0, 2.0), offset=(0.0, 1.0))
    by_degrees = interp.nearest_at(frame, [0.0], [60.0], metric="degrees")[0][0]
    by_km = interp.nearest_at(frame, [0.0], [60.0], metric="great_circle")[0][0]
    assert by_degrees == 2.0 and by_km == 1.0
    assert interp.distance_to_nearest(frame, [0.0], [60.0], metric="great_circle")[0] > 50
