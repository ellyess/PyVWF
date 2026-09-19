"""Onshore and offshore classification (src/vwf/geospatial.py).

Ported from the `development` branch, where it carried no tests at all.

One of these pins a defect found in the port and checked against the original:
a point matching two overlapping polygons of one file produced a join result
longer than the frame, and the original then labelled an unrelated offshore
point as onshore. The repeated-index test below pins an invariant instead: the
original happens to survive the case, and it is kept because the positional
implementation is what makes that true rather than luck.

The two implementations are checked against each other on the same points,
which is the independent route: they share only the shape loading.
"""

import json

import pandas as pd
import pytest

from vwf import geospatial


def square(lon0, lat0, lon1, lat1):
    return {
        "type": "Polygon",
        "coordinates": [[[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]],
    }


def geojson(tmp_path, name, *polygons):
    path = tmp_path / f"{name}.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {"n": i}, "geometry": g}
                    for i, g in enumerate(polygons)
                ],
            }
        )
    )
    return path


@pytest.fixture
def shapes(tmp_path):
    """Onshore 0 to 2 east, offshore 1 to 3 east: they overlap from 1 to 2."""
    return (
        geojson(tmp_path, "onshore", square(0, 50, 2, 52)),
        geojson(tmp_path, "offshore", square(1, 50, 3, 52)),
    )


def points(lons, index=None):
    return pd.DataFrame(
        {"lon": list(lons), "lat": [51.0] * len(lons)},
        index=index if index is not None else range(len(lons)),
    )


@pytest.mark.parametrize("method", ["spatial_join", "point_in_polygon"])
def test_a_point_is_labelled_by_the_shapes_it_falls_in(shapes, method):
    on, off = shapes
    got = geospatial.add_domain_column(
        points([0.5, 2.5, 9.0]), onshore_geojson=on, offshore_geojson=off, method=method
    )
    assert list(got["domain"]) == ["onshore", "offshore", "unknown"]


@pytest.mark.parametrize("method", ["spatial_join", "point_in_polygon"])
def test_a_point_in_both_domains_takes_the_preference(shapes, method):
    on, off = shapes
    kw = dict(onshore_geojson=on, offshore_geojson=off, method=method, overwrite=True)
    both = points([1.5])
    assert (
        geospatial.add_domain_column(both.copy(), prefer_onshore=True, **kw)["domain"][0]
        == "onshore"
    )
    assert (
        geospatial.add_domain_column(both.copy(), prefer_onshore=False, **kw)["domain"][0]
        == "offshore"
    )


def test_the_two_implementations_agree(shapes):
    """They share only the shape loading, so agreement is evidence about the
    join rather than about a helper they both call."""
    on, off = shapes
    frame = points([0.2, 0.5, 1.2, 1.8, 2.4, 2.9, 5.0, -1.0])
    a = geospatial.categorize_points_spatial_join(frame, onshore_geojson=on, offshore_geojson=off)
    b = geospatial.categorize_points_by_region(frame, onshore_geojson=on, offshore_geojson=off)
    assert list(a) == list(b)


def test_a_point_matching_two_polygons_of_one_file_stays_one_row(tmp_path):
    """The join returns a row per matching polygon. Reading it through the
    frame's index gave a mask longer than the frame. Checked against the
    original, which returns ['onshore', 'onshore'] here: the offshore point at
    8.5 is mislabelled because the masks no longer line up with the frame."""
    on = geojson(tmp_path, "onshore", square(0, 50, 2, 52), square(1, 50, 3, 52))
    off = geojson(tmp_path, "offshore", square(8, 50, 9, 52))
    got = geospatial.categorize_points_spatial_join(
        points([1.5, 8.5]), onshore_geojson=on, offshore_geojson=off
    )
    assert list(got) == ["onshore", "offshore"] and len(got) == 2


def test_a_frame_with_repeated_index_labels_is_classified_positionally(shapes):
    """Fleet frames arrive with a reset index most of the time and not always.
    A duplicated label must not merge two units' answers. The original passes
    this case too; it is pinned because the positional join is what keeps it
    true, and an index-keyed one would not."""
    on, off = shapes
    frame = points([0.5, 2.5], index=["a", "a"])
    got = geospatial.categorize_points_spatial_join(frame, onshore_geojson=on, offshore_geojson=off)
    assert list(got) == ["onshore", "offshore"]


def test_a_boundary_point_is_unknown_rather_than_either(shapes):
    """Strictly inside is the rule, so a point on the edge is a third answer.
    It matters because a coastline is where the units are."""
    on, off = shapes
    got = geospatial.categorize_points_spatial_join(
        points([0.0]), onshore_geojson=on, offshore_geojson=off
    )
    assert list(got) == ["unknown"]


def test_an_empty_shape_file_is_refused(tmp_path, shapes):
    _, off = shapes
    empty = tmp_path / "empty.geojson"
    empty.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    with pytest.raises(ValueError, match="No geometries"):
        geospatial.categorize_points_spatial_join(
            points([0.5]), onshore_geojson=empty, offshore_geojson=off
        )


def test_an_existing_domain_column_is_not_overwritten_silently(shapes):
    on, off = shapes
    frame = points([0.5]).assign(domain="set by hand")
    with pytest.raises(ValueError, match="already exists"):
        geospatial.add_domain_column(frame, onshore_geojson=on, offshore_geojson=off)
    got = geospatial.add_domain_column(
        frame, onshore_geojson=on, offshore_geojson=off, overwrite=True
    )
    assert list(got["domain"]) == ["onshore"]


def test_an_unknown_method_or_domain_is_refused(shapes):
    on, off = shapes
    with pytest.raises(ValueError, match="Unknown method"):
        geospatial.add_domain_column(
            points([0.5]), onshore_geojson=on, offshore_geojson=off, method="kriging"
        )
    with pytest.raises(ValueError, match="Unknown domain"):
        geospatial.filter_by_domain(pd.DataFrame({"domain": ["onshore"]}), "offshore-ish")


def test_filtering_selects_one_domain_or_both_known_ones():
    frame = pd.DataFrame({"domain": ["onshore", "offshore", "unknown", "onshore"]})
    assert len(geospatial.filter_by_domain(frame, "onshore")) == 2
    assert len(geospatial.filter_by_domain(frame, "offshore")) == 1
    assert len(geospatial.filter_by_domain(frame, "known")) == 3
    with pytest.raises(ValueError, match="not found"):
        geospatial.filter_by_domain(pd.DataFrame({"x": [1]}), "onshore")
