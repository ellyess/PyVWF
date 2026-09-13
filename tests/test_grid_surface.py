"""Gridded correction surfaces (src/vwf/extensions/grid/surface.py).

Ported from the `development` branch, where it carried no tests. The test that
matters most is the first: the control-point set is an argument, so a holdout
is the same call with rows removed and runs through the route the product uses.
Both registered studies depend on that being true.
"""
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from vwf.extensions.grid import surface


def shapes(tmp_path):
    """Onshore 0 to 4 east, offshore 4 to 8 east, both 50 to 54 north."""
    def write(name, lon0, lon1):
        path = tmp_path / f"{name}.geojson"
        path.write_text(json.dumps({"type": "FeatureCollection", "features": [{
            "type": "Feature", "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [[
                [lon0, 50], [lon1, 50], [lon1, 54], [lon0, 54], [lon0, 50]]]}}]}))
        return path
    return write("onshore", 0, 4), write("offshore", 4, 8)


def control(n_on=8, n_off=6, scalar_on=0.8, scalar_off=1.2):
    rng = np.random.default_rng(0)
    on = pd.DataFrame({
        "lon": rng.uniform(0.5, 3.5, n_on), "lat": rng.uniform(50.5, 53.5, n_on),
        "scalar": scalar_on + rng.normal(0, 0.02, n_on),
        "offset": rng.normal(0.3, 0.02, n_on), "cluster_mode": "onshore"})
    off = pd.DataFrame({
        "lon": rng.uniform(4.5, 7.5, n_off), "lat": rng.uniform(50.5, 53.5, n_off),
        "scalar": scalar_off + rng.normal(0, 0.02, n_off),
        "offset": rng.normal(-0.3, 0.02, n_off), "cluster_mode": "offshore"})
    return pd.concat([on, off], ignore_index=True)


GRID_LON = np.arange(-1.0, 9.01, 0.5)
GRID_LAT = np.arange(49.0, 55.01, 0.5)


def build(tmp_path, points=None, **kw):
    on, off = shapes(tmp_path)
    return surface.correction_surface(
        control() if points is None else points, GRID_LON, GRID_LAT,
        onshore_geojson=on, offshore_geojson=off, method=kw.pop("method", "idw"), **kw)


def test_the_control_point_set_is_an_argument_so_a_holdout_is_the_same_call(tmp_path):
    """The capability both registered studies were blocked on. A subset runs
    through the same function, not a second path beside it."""
    full = control()
    held_out = full[~((full["lon"] > 1.0) & (full["lon"] < 2.5))]
    a = build(tmp_path, full)
    b = build(tmp_path, held_out)
    assert a.attrs["n_control_points"] == len(full)
    assert b.attrs["n_control_points"] == len(held_out)
    assert not np.allclose(a["scalar"].values, b["scalar"].values)


def test_cells_outside_every_area_are_neutral_rather_than_extrapolated(tmp_path):
    got = build(tmp_path)
    outside = ~(got["is_onshore_area"] | got["is_offshore_area"])
    assert outside.values.any()
    assert (got["scalar"].values[outside.values] == surface.NEUTRAL_SCALAR).all()
    assert (got["offset"].values[outside.values] == surface.NEUTRAL_OFFSET).all()


def test_each_domain_is_interpolated_from_its_own_points(tmp_path):
    """An onshore cell must not take an offshore correction. The two pools
    differ by 0.4 in scalar here, so a leak is visible."""
    got = build(tmp_path)
    on = got["scalar"].values[got["is_onshore_area"].values]
    off = got["scalar"].values[got["is_offshore_area"].values]
    assert on.max() < 1.0 and off.min() > 1.0


def test_a_domain_with_fewer_than_five_points_is_refused(tmp_path):
    """Where Denmark offshore's documented failure sits: two points."""
    points = control(n_off=2)
    with pytest.raises(ValueError, match="offshore pool has 2 control points"):
        build(tmp_path, points)


def test_country_level_points_join_the_onshore_pool(tmp_path):
    """The chapter's rule, reproduced: cluster_mode 'all' is onshore."""
    points = control()
    points.loc[points["cluster_mode"] == "onshore", "cluster_mode"] = surface.COUNTRY_MODE
    got = build(tmp_path, points)
    assert got.attrs["n_control_points_onshore"] == 8
    assert got.attrs["n_control_points_offshore"] == 6


def test_a_missing_domain_column_is_refused_with_what_to_do(tmp_path):
    with pytest.raises(ValueError, match="declared split needs one"):
        build(tmp_path, control().drop(columns=["cluster_mode"]))


def test_missing_value_columns_are_refused(tmp_path):
    with pytest.raises(ValueError, match="missing \\['offset'\\]"):
        build(tmp_path, control().drop(columns=["offset"]))


def test_the_attributes_say_the_correction_applies_to_wind_speed(tmp_path):
    """The original said 'applied to wind power output', which is wrong and is
    the kind of wrong a user acts on."""
    got = build(tmp_path)
    assert "WIND SPEED" in got["scalar"].attrs["description"]
    assert "before the power curve" in got.attrs["usage"]
    assert got["offset"].attrs["units"] == "m s-1"


def test_thinning_happens_only_above_its_threshold_and_is_recorded(tmp_path):
    got = build(tmp_path, thin_onshore_above=4, thin_bin_ddeg=2.0)
    assert "onshore_thinned_from" in got.attrs
    assert got.attrs["onshore_thinned_from"] == "8"
    assert build(tmp_path).attrs.get("onshore_thinned_from") is None


def test_an_unknown_method_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown method"):
        build(tmp_path, method="rbf")


def test_kriging_runs_through_the_shared_definition(tmp_path):
    pytest.importorskip("pykrige", reason="kriging is in the 'grid' extra")
    got = build(tmp_path, method="kriging", n_closest_onshore=None,
                n_closest_offshore=None)
    assert got.attrs["method"] == "kriging"
    assert got.attrs["coordinates_type"] == "geographic"
    assert np.isfinite(got["scalar"].values).all()


def test_a_cutout_gives_up_its_axes_with_or_without_the_extra_variables():
    """The original dropped lat, lon and height unconditionally and failed on a
    cutout that had none of them."""
    bare = xr.Dataset(coords={"x": [1.0, 2.0], "y": [50.0, 51.0]})
    lon, lat = surface.cutout_lonlat(bare)
    assert list(lon) == [1.0, 2.0] and list(lat) == [50.0, 51.0]

    full = xr.Dataset(
        {"height": (("y", "x"), np.zeros((2, 2))),
         "lat": (("y", "x"), np.zeros((2, 2))), "lon": (("y", "x"), np.zeros((2, 2)))},
        coords={"x": [1.0, 2.0], "y": [50.0, 51.0]})
    lon, lat = surface.cutout_lonlat(full)
    assert list(lon) == [1.0, 2.0]

    with pytest.raises(KeyError, match="coordinate"):
        surface.cutout_lonlat(xr.Dataset(coords={"a": [1.0]}))


def test_a_cell_inside_two_overlapping_polygons_of_one_file_matches_once(tmp_path):
    """offshore_shapes.geojson holds 44 overlapping pairs. Unioning before the
    join is what keeps this free of the defect fixed in vwf.geospatial."""
    path = tmp_path / "overlapping.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[
            [0, 50], [4, 50], [4, 54], [0, 54], [0, 50]]]}},
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[
            [2, 50], [6, 50], [6, 54], [2, 54], [2, 50]]]}}]}))
    mask = surface.area_mask(GRID_LON, GRID_LAT, path, name="m")
    assert mask.shape == (len(GRID_LAT), len(GRID_LON))
    assert bool(mask.sel(lon=3.0, lat=52.0)) and not bool(mask.sel(lon=8.0, lat=52.0))


def test_a_written_surface_carries_the_axis_names_atlite_reads(tmp_path):
    got = build(tmp_path)
    path = surface.export_correction_surface(tmp_path / "out.nc", got)
    written = xr.open_dataset(path)
    assert set(written["scalar"].dims) == {"y", "x"}
    assert written["scalar"].dims == ("y", "x")
    written.close()

    plain = surface.export_correction_surface(tmp_path / "plain.nc", got, for_atlite=False)
    written = xr.open_dataset(plain)
    assert set(written["scalar"].dims) == {"lat", "lon"}
    written.close()


def test_the_declared_and_shape_splits_disagree_and_it_is_only_reported(tmp_path):
    """Thirty of the chapter's 1,729 points disagree. The surface uses the
    declared mode; this function exists so the disagreement is visible."""
    on, off = shapes(tmp_path)
    points = control()
    points.loc[0, "lon"] = 6.0          # declared onshore, inside the offshore shape
    differ = surface.domain_disagreement(points, onshore_geojson=on, offshore_geojson=off)
    assert len(differ) >= 1
    assert differ.iloc[0]["declared"] == "onshore"
    assert differ.iloc[0]["by_shapes"] == "offshore"
    got = surface.correction_surface(points, GRID_LON, GRID_LAT, onshore_geojson=on,
                                     offshore_geojson=off, method="idw")
    assert got.attrs["n_control_points_onshore"] == 8
