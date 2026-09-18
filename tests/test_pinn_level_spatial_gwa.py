"""The level and spatial split of a fleet's error, and the wind-atlas ratio.

The turbine-only study's primary metric is the spatial part of the per-unit
error, which a national series cannot see, and one of its arms scales ERA5 by a
Global Wind Atlas ratio. Both are pinned here against hand computations.
Nothing here imports torch; the atlas tests need rasterio and skip without it.
"""
import numpy as np
import pandas as pd
import pytest

from vwf.pinn.runs import level_spatial


def test_level_and_spatial_parts_match_a_hand_computation():
    # Two units over two months. Unit a (capacity 1) errs +0.10 then +0.20,
    # unit b (capacity 3) errs -0.10 then 0.00.
    frame = pd.DataFrame({
        "ID": ["a", "b", "a", "b"], "year": 2020, "month": [1, 1, 2, 2],
        "cf_obs": [0.3, 0.3, 0.3, 0.3], "cf_sim": [0.4, 0.2, 0.5, 0.3],
        "capacity": [1.0, 3.0, 1.0, 3.0],
    })
    m = level_spatial(frame)
    level = {1: (0.10 * 1 - 0.10 * 3) / 4, 2: (0.20 * 1 + 0.0 * 3) / 4}      # -0.05, 0.05
    s_a = np.mean([0.10 - level[1], 0.20 - level[2]])                          # 0.125
    s_b = np.mean([-0.10 - level[1], 0.0 - level[2]])                          # -0.05
    spatial = np.sqrt((1 * 2 * s_a**2 + 3 * 2 * s_b**2) / (1 * 2 + 3 * 2))
    lvl = np.sqrt((4 * level[1] ** 2 + 4 * level[2] ** 2) / 8)
    rmse = np.sqrt((1 * 0.01 + 3 * 0.01 + 1 * 0.04 + 3 * 0.0) / 8)
    assert m["spatial_rmse"] == pytest.approx(spatial)
    assert m["level_rmse"] == pytest.approx(lvl)
    assert m["rmse"] == pytest.approx(rmse)
    assert (m["n_units"], m["n_samples"]) == (2, 4)


def test_a_pure_level_error_has_no_spatial_part():
    frame = pd.DataFrame({"ID": ["a", "b"] * 3, "year": 2020, "month": [1, 1, 2, 2, 3, 3],
                          "cf_obs": 0.3, "cf_sim": [0.35, 0.35, 0.25, 0.25, 0.4, 0.4],
                          "capacity": [1.0, 2.0] * 3})
    m = level_spatial(frame)
    assert m["spatial_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert m["level_rmse"] == pytest.approx(m["rmse"])


def _raster(path, values, west=10.0, north=50.0, res=0.01, nodata=-9999.0):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    values = np.asarray(values, dtype="float32")
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
                       count=1, dtype="float32", crs="EPSG:4326", nodata=nodata,
                       transform=from_origin(west, north, res, res)) as dst:
        dst.write(values, 1)
    return path


def test_the_atlas_mean_uses_only_cells_within_the_radius(tmp_path):
    from vwf.pinn.gwa import KM_PER_DEG, atlas_means
    # 9 x 9 cells of 0.01 deg; the centre cell holds 9 and every other cell 3,
    # except one nodata cell next to the centre.
    grid = np.full((9, 9), 3.0)
    grid[4, 4] = 9.0
    grid[4, 5] = -9999.0
    path = _raster(tmp_path / "a.tif", grid)
    lon, lat = 10.045, 49.955                     # the centre of cell (4, 4)
    # A radius smaller than one cell spacing takes the centre cell alone.
    assert atlas_means(path, [lon], [lat], radius_km=0.3)[0] == pytest.approx(9.0)
    # Longitude is scaled by cos(lat), so at 50 N the east and west
    # neighbours sit 0.0064 deg away, north and south 0.01 deg, and the
    # diagonals 0.0119 deg. A radius of 0.0105 deg takes the four orthogonal
    # neighbours and no diagonal; the eastern one is nodata and is ignored.
    got = atlas_means(path, [lon], [lat], radius_km=0.0105 * KM_PER_DEG)[0]
    assert got == pytest.approx((9.0 + 3.0 * 3) / 4)
    # At 0.012 deg the four diagonals come in too: 7 valid cells of 3 and one of 9.
    got = atlas_means(path, [lon], [lat], radius_km=0.012 * KM_PER_DEG)[0]
    assert got == pytest.approx((9.0 + 3.0 * 7) / 8)


def test_the_ratio_is_clipped_and_a_point_off_the_atlas_is_neutral(tmp_path):
    from vwf.pinn.gwa import gwa_ratio
    path = _raster(tmp_path / "b.tif", np.full((5, 5), 8.0))
    table = gwa_ratio(["in", "high", "off"], [10.02, 10.02, 30.0], [49.98, 49.98, 10.0],
                      era5_mean=[8.0 / 1.25, 2.0, 7.0], raster_path=path, radius_km=1.0)
    t = table.set_index("ID")
    assert t.loc["in", "ratio"] == pytest.approx(1.25)
    assert not t.loc["in", "clipped"] and not t.loc["in", "neutral"]
    assert t.loc["high", "ratio_raw"] == pytest.approx(4.0)
    assert t.loc["high", "ratio"] == pytest.approx(2.46) and t.loc["high", "clipped"]
    assert t.loc["off", "ratio"] == 1.0 and t.loc["off", "neutral"]
