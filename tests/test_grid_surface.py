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
        path.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    [[lon0, 50], [lon1, 50], [lon1, 54], [lon0, 54], [lon0, 50]]
                                ],
                            },
                        }
                    ],
                }
            )
        )
        return path

    return write("onshore", 0, 4), write("offshore", 4, 8)


def control(n_on=8, n_off=6, scalar_on=0.8, scalar_off=1.2):
    rng = np.random.default_rng(0)
    on = pd.DataFrame(
        {
            "lon": rng.uniform(0.5, 3.5, n_on),
            "lat": rng.uniform(50.5, 53.5, n_on),
            "scalar": scalar_on + rng.normal(0, 0.02, n_on),
            "offset": rng.normal(0.3, 0.02, n_on),
            "cluster_mode": "onshore",
        }
    )
    off = pd.DataFrame(
        {
            "lon": rng.uniform(4.5, 7.5, n_off),
            "lat": rng.uniform(50.5, 53.5, n_off),
            "scalar": scalar_off + rng.normal(0, 0.02, n_off),
            "offset": rng.normal(-0.3, 0.02, n_off),
            "cluster_mode": "offshore",
        }
    )
    return pd.concat([on, off], ignore_index=True)


GRID_LON = np.arange(-1.0, 9.01, 0.5)
GRID_LAT = np.arange(49.0, 55.01, 0.5)


def build(tmp_path, points=None, **kw):
    on, off = shapes(tmp_path)
    return surface.correction_surface(
        control() if points is None else points,
        GRID_LON,
        GRID_LAT,
        onshore_geojson=on,
        offshore_geojson=off,
        method=kw.pop("method", "idw"),
        **kw,
    )


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


def test_every_cell_is_corrected_including_outside_every_area(tmp_path):
    """Since 2026-09-15 the surface answers everywhere. A cell filled with unity
    is indistinguishable in the file from a cell whose correction happens to be
    the identity, which is what the support variables and the flag replace."""
    got = build(tmp_path)
    outside = ~(got["is_onshore_area"] | got["is_offshore_area"])
    assert outside.values.any()
    values = got["scalar"].values[outside.values]
    assert np.isfinite(values).all()
    assert not (values == surface.NEUTRAL_SCALAR).all()
    assert got.attrs["outside_areas"].startswith("corrected from the nearest")


def test_the_old_neutral_fill_is_still_available_by_name(tmp_path):
    got = build(tmp_path, neutral_outside_areas=True)
    outside = ~(got["is_onshore_area"] | got["is_offshore_area"])
    assert (got["scalar"].values[outside.values] == surface.NEUTRAL_SCALAR).all()
    assert (got["offset"].values[outside.values] == surface.NEUTRAL_OFFSET).all()
    assert got.attrs["outside_areas"] == "neutral"


def test_a_cell_outside_both_areas_takes_its_nearest_control_point_domain(tmp_path):
    """The two pools differ by 0.4 in scalar, so which surface a cell took is
    visible. South of both shapes, the nearer pool is whichever is nearer in
    longitude."""
    got = build(tmp_path)
    west = got.sel(lon=1.0, lat=49.0, method="nearest")  # below the onshore box
    east = got.sel(lon=7.0, lat=49.0, method="nearest")  # below the offshore box
    assert not bool(west["is_onshore_area"]) and not bool(west["is_offshore_area"])
    assert bool(west["nearest_is_onshore"]) and not bool(east["nearest_is_onshore"])
    assert float(west["scalar"]) < 1.0 and float(east["scalar"]) > 1.0


def test_the_support_variables_say_how_much_data_a_cell_rests_on(tmp_path):
    got = build(tmp_path)
    for name in ("distance_to_control_deg", "distance_to_control_km", "n_control_within_horizon"):
        assert name in got
    near = got.sel(lon=2.0, lat=52.0, method="nearest")
    far = got.sel(lon=9.0, lat=49.0, method="nearest")
    assert float(near["distance_to_control_deg"]) < float(far["distance_to_control_deg"])
    assert int(near["n_control_within_horizon"]) >= int(far["n_control_within_horizon"])
    # The two metrics are not interchangeable and the file says so.
    assert got["distance_to_control_km"].attrs["units"] == "km"
    assert got["distance_to_control_deg"].attrs["units"] == "degree"
    assert float(got["distance_to_control_km"].max()) > float(got["distance_to_control_deg"].max())


def test_the_horizon_is_recorded_as_provenance_not_as_safety(tmp_path):
    got = build(tmp_path)
    assert got.attrs["information_horizon_deg"] == 5.0
    meaning = got.attrs["information_horizon_meaning"]
    assert "carries no information" in meaning
    assert "not about safety" in meaning
    assert "method-distance-mask" in meaning
    assert "plausible" in got.attrs["recommended_filter"]


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
    got = build(tmp_path, method="kriging", n_closest_onshore=None, n_closest_offshore=None)
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
        {
            "height": (("y", "x"), np.zeros((2, 2))),
            "lat": (("y", "x"), np.zeros((2, 2))),
            "lon": (("y", "x"), np.zeros((2, 2))),
        },
        coords={"x": [1.0, 2.0], "y": [50.0, 51.0]},
    )
    lon, lat = surface.cutout_lonlat(full)
    assert list(lon) == [1.0, 2.0]

    with pytest.raises(KeyError, match="coordinate"):
        surface.cutout_lonlat(xr.Dataset(coords={"a": [1.0]}))


def test_a_cell_inside_two_overlapping_polygons_of_one_file_matches_once(tmp_path):
    """offshore_shapes.geojson holds 44 overlapping pairs. Unioning before the
    join is what keeps this free of the defect fixed in vwf.geospatial."""
    path = tmp_path / "overlapping.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 50], [4, 50], [4, 54], [0, 54], [0, 50]]],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[2, 50], [6, 50], [6, 54], [2, 54], [2, 50]]],
                        },
                    },
                ],
            }
        )
    )
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


def test_the_flag_catches_a_defect_sitting_on_top_of_the_control_points(tmp_path):
    """An offset of -5 against a scalar of 0.8 crosses zero at 6.25 m/s, so the
    pair refuses ordinary winds rather than correcting them. It sits right on
    top of its control points, where no distance or variance threshold looks."""
    points = control()
    points.loc[points["cluster_mode"] == "onshore", "offset"] = -5.0
    points.loc[points["cluster_mode"] == "onshore", "scalar"] = 0.8
    got = build(tmp_path, points)
    onshore = got["is_onshore_area"].values
    assert not got["plausible"].values[onshore].any()
    assert float(got["zero_crossing_speed"].values[onshore].max()) > 4.0
    # The cells the flag rejects are the close ones, which is the point.
    rejected = ~got["plausible"].values
    assert (
        got["distance_to_control_deg"].values[rejected].mean()
        < got["distance_to_control_deg"].values[~rejected].mean()
    )


def test_a_degenerate_scalar_is_flagged_by_the_projects_own_bounds(tmp_path):
    points = control(scalar_off=5.0)
    got = build(tmp_path, points)
    assert not got["plausible"].values[got["is_offshore_area"].values].all()
    assert "degenerate fit" in got["plausible"].attrs["description"]


def test_an_ordinary_surface_is_plausible_everywhere(tmp_path):
    """The offshore pool's offsets are negative here, so crossings exist. A
    crossing is not itself a defect: it is one below the operating range."""
    got = build(tmp_path)
    assert got["plausible"].values.all()
    crossings = got["zero_crossing_speed"].values
    assert np.isfinite(crossings).any()
    assert np.nanmax(crossings) <= surface.MAX_ZERO_CROSSING_SPEED


def test_kriging_carries_its_own_variance_onto_the_grid(tmp_path):
    pytest.importorskip("pykrige", reason="kriging is in the 'grid' extra")
    got = build(tmp_path, method="kriging", n_closest_onshore=None, n_closest_offshore=None)
    assert "scalar_variance" in got and "offset_variance" in got
    assert (got["scalar_variance"].values >= 0).all()
    # Variance grows away from the points, which is why it is not the guard:
    # it ranks cells the same way distance does.
    near = float(got["scalar_variance"].sel(lon=2.0, lat=52.0, method="nearest"))
    far = float(got["scalar_variance"].sel(lon=9.0, lat=49.0, method="nearest"))
    assert far > near


def test_idw_carries_no_variance_because_it_has_none(tmp_path):
    got = build(tmp_path, method="idw")
    assert "scalar_variance" not in got


def test_the_declared_and_shape_splits_disagree_and_it_is_only_reported(tmp_path):
    """Thirty of the chapter's 1,729 points disagree. The surface uses the
    declared mode; this function exists so the disagreement is visible."""
    on, off = shapes(tmp_path)
    points = control()
    points.loc[0, "lon"] = 6.0  # declared onshore, inside the offshore shape
    differ = surface.domain_disagreement(points, onshore_geojson=on, offshore_geojson=off)
    assert len(differ) >= 1
    assert differ.iloc[0]["declared"] == "onshore"
    assert differ.iloc[0]["by_shapes"] == "offshore"
    got = surface.correction_surface(
        points, GRID_LON, GRID_LAT, onshore_geojson=on, offshore_geojson=off, method="idw"
    )
    assert got.attrs["n_control_points_onshore"] == 8


def test_the_two_screens_differ_only_on_a_missing_scalar():
    # plausible; scalar too low; scalar too high; crossing too fast (6 m/s);
    # crossing within bounds (2 m/s); missing scalar.
    scalar = pd.Series([1.0, 0.1, 3.5, 1.0, 1.0, np.nan])
    offset = pd.Series([0.5, 0.0, 0.0, -6.0, -2.0, 0.0])
    crossing = surface.zero_crossing_speed(scalar, offset)
    assert crossing.isna().tolist() == [True, True, True, False, False, True]
    assert crossing[3] == 6.0 and crossing[4] == 2.0
    plausible = surface.within_plausible_bounds(scalar, offset)
    flagged = surface.flag_implausible(scalar, offset)
    assert plausible.tolist() == [True, False, False, False, True, False]
    assert flagged.tolist() == [False, True, True, True, False, False]
