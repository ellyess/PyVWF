"""Joining factors to cluster geometries (src/vwf/extensions/grid/geodataframes.py).

Ported from the `development` branch, where it carried no tests. Most of these
pin refusals that replace silent failures in the original: a left join that
mapped a missing fit as missing values, a dropped cluster, an unrecognised
output suffix written as GeoJSON anyway, and a time-slice filter that matched
nothing and returned an empty frame.
"""
import json

import pandas as pd
import pytest

from vwf.extensions.grid import geodataframes as gdf


def square(i):
    return {"type": "Polygon", "coordinates": [[
        [i, 50], [i + 1, 50], [i + 1, 51], [i, 51], [i, 50]]]}


def geoms(tmp_path, clusters=(0, 1, 2)):
    path = tmp_path / "clusters.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"cluster": c}, "geometry": square(c)}
        for c in clusters]}))
    return path


def factors(tmp_path, clusters=(0, 1, 2), slices=("1/1",), column="fixed", name="factors"):
    rows = [{"cluster": c, column: s, "scalar": 1.0 + c / 10, "offset": -c / 10}
            for s in slices for c in clusters]
    path = tmp_path / f"{name}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_factors_are_joined_to_the_geometry_of_their_own_cluster(tmp_path):
    got = gdf.correction_geodataframe(factors(tmp_path), geoms(tmp_path))
    assert len(got) == 3
    assert list(got["cluster"]) == [0, 1, 2]
    assert got.loc[got["cluster"] == 2, "scalar"].iloc[0] == pytest.approx(1.2)
    assert got.geometry.notna().all()


def test_a_cluster_with_a_geometry_and_no_factors_is_refused(tmp_path):
    """The original left-joined, so this cluster came back as a row of missing
    values and mapped as a hole rather than as an error."""
    with pytest.raises(ValueError, match="different fits"):
        gdf.correction_geodataframe(factors(tmp_path, clusters=(0, 1)),
                                    geoms(tmp_path, clusters=(0, 1, 2)))


def test_a_cluster_with_factors_and_no_geometry_is_refused(tmp_path):
    """The original dropped it, so a fit silently left the map."""
    with pytest.raises(ValueError, match="different fits"):
        gdf.correction_geodataframe(factors(tmp_path, clusters=(0, 1, 2)),
                                    geoms(tmp_path, clusters=(0, 1)))


def test_one_time_slice_can_be_selected(tmp_path):
    path = factors(tmp_path, slices=("winter", "summer"), column="season")
    got = gdf.correction_geodataframe(path, geoms(tmp_path), time_slice="winter")
    assert len(got) == 3 and set(got["season"]) == {"winter"}


def test_every_slice_is_kept_when_none_is_asked_for(tmp_path):
    path = factors(tmp_path, slices=("winter", "summer"), column="season")
    got = gdf.correction_geodataframe(path, geoms(tmp_path))
    assert len(got) == 6


def test_a_slice_that_is_not_there_is_refused_with_what_is(tmp_path):
    """It used to filter to nothing and join an empty frame."""
    path = factors(tmp_path, slices=("winter", "summer"), column="season")
    with pytest.raises(ValueError, match="not in the 'season' column"):
        gdf.correction_geodataframe(path, geoms(tmp_path), time_slice="autumn")


def test_asking_for_a_slice_of_a_table_that_has_none_is_refused(tmp_path):
    path = tmp_path / "noslice.csv"
    pd.DataFrame({"cluster": [0, 1, 2], "scalar": [1.0] * 3,
                  "offset": [0.0] * 3}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="none of"):
        gdf.correction_geodataframe(path, geoms(tmp_path), time_slice="winter")


def test_a_frame_without_a_cluster_column_is_refused(tmp_path):
    path = tmp_path / "nocluster.csv"
    pd.DataFrame({"scalar": [1.0], "offset": [0.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no 'cluster' column"):
        gdf.correction_geodataframe(path, geoms(tmp_path))


@pytest.mark.parametrize("suffix", [".geojson", ".gpkg"])
def test_it_writes_the_format_the_suffix_names(tmp_path, suffix):
    out = tmp_path / f"written{suffix}"
    gdf.correction_geodataframe(factors(tmp_path), geoms(tmp_path), output_path=out)
    assert out.exists() and out.stat().st_size > 0


def test_an_unrecognised_suffix_is_refused_rather_than_written_as_geojson(tmp_path):
    with pytest.raises(ValueError, match="unsupported output suffix"):
        gdf.correction_geodataframe(factors(tmp_path), geoms(tmp_path),
                                    output_path=tmp_path / "written.nc")


def test_every_factors_table_of_a_row_is_joined(tmp_path):
    d = tmp_path / "factors"
    d.mkdir()
    factors(d, slices=("1/1",), column="fixed", name="NL_factors_fixed_3")
    factors(d, slices=("winter", "summer"), column="season", name="NL_factors_season_3")
    got = gdf.country_correction_geodataframes("NL", factors_dir=d,
                                               geometry_file=geoms(tmp_path))
    assert set(got) == {"NL_factors_fixed_3", "NL_factors_season_3"}
    assert len(got["NL_factors_fixed_3"]) == 3
    assert len(got["NL_factors_season_3"]) == 6


def test_a_row_with_no_factors_table_is_refused(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no factors table"):
        gdf.country_correction_geodataframes("NL", factors_dir=empty,
                                             geometry_file=geoms(tmp_path))


def test_the_harness_naming_is_found_as_well_as_the_chapter_s(tmp_path):
    """Harness runs write factors_<slice>_<n>.csv with no code prefix."""
    d = tmp_path / "factors"
    d.mkdir()
    factors(d, slices=("1/1",), column="fixed", name="factors_fixed_3")
    got = gdf.country_correction_geodataframes("NL", factors_dir=d,
                                               geometry_file=geoms(tmp_path))
    assert set(got) == {"factors_fixed_3"}
