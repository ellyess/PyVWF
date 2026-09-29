"""Offshore projects get their own grid points; onshore ones keep the land lattice.

``scripts/region_tools/weight_country_grid_points.py`` builds the per-year
country grids. The lattices cover land only, so until 2026-09-29 an offshore
project was summed onto the nearest land point: Belgium's whole offshore fleet
sat on two coastal points. These cases check the helpers on synthetic grids.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/region_tools/weight_country_grid_points.py"
    spec = importlib.util.spec_from_file_location("weight_country_grid_points", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def land_grid() -> pd.DataFrame:
    """Two coastal land points in different clusters, and one inland."""
    return pd.DataFrame(
        {
            "lat": [51.25, 51.25, 50.5],
            "lon": [2.75, 3.25, 4.5],
            "weight": [0.0, 0.0, 0.0],
            "ID": ["g0", "g1", "g2"],
            "height": [100.0] * 3,
            "model": ["Vestas.V90.3000"] * 3,
            "capacity": [0.0] * 3,
            "type": ["onshore"] * 3,
            "cluster": [0, 1, 2],
        }
    )


def fleet() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lat": [51.20, 50.55, 51.65, 51.65, 51.54],
            "lon": [2.80, 4.40, 2.83, 2.83, 3.30],
            "mw": [10.0, 20.0, 165.0, 165.0, 30.0],
            "offshore": [False, False, True, True, True],
        }
    )


def test_each_offshore_site_gets_its_own_point_and_phases_share_it():
    m = _module()
    f = fleet()
    points = m.offshore_sites(f[f["offshore"]], land_grid(), "BE")
    assert list(points.columns) == list(land_grid().columns)
    # Two sites: the two 165 MW phases at one position share a point.
    assert points[["lat", "lon"]].values.tolist() == [[51.54, 3.3], [51.65, 2.83]]
    assert points["type"].tolist() == ["offshore", "offshore"]
    assert points["height"].tolist() == [100.0, 100.0]
    assert points["model"].tolist() == ["Vestas.V90.3000"] * 2
    w = m.site_weights(points, f[f["offshore"]])
    assert w.tolist() == [30.0, 330.0]


def test_an_offshore_point_takes_the_nearest_land_points_cluster():
    m = _module()
    f = fleet()
    points = m.offshore_sites(f[f["offshore"]], land_grid(), "BE")
    # (51.54, 3.30) is nearest g1 (cluster 1); (51.65, 2.83) nearest g0 (cluster 0).
    assert points["cluster"].tolist() == [1, 0]


def test_onshore_capacity_stays_on_land_and_the_total_is_kept():
    m = _module()
    f = fleet()
    grid = land_grid()
    onshore, offshore = f[~f["offshore"]], f[f["offshore"]]
    on_land = m.assign_to_grid(grid, onshore)
    at_sea = m.site_weights(m.offshore_sites(offshore, grid, "BE"), offshore)
    assert on_land.tolist() == [10.0, 0.0, 20.0]
    assert on_land.sum() + at_sea.sum() == f["mw"].sum()
    # Known positive for the old behaviour: without the split, the offshore
    # capacity lands on the coastal land points.
    snapped = m.assign_to_grid(grid, f)
    assert snapped[grid["ID"] == "g0"].item() > 10.0


def test_no_offshore_projects_adds_no_points():
    m = _module()
    f = fleet()
    points = m.offshore_sites(f[~f["offshore"] & f["offshore"]], land_grid(), "BE")
    assert points.empty
    assert list(points.columns) == list(land_grid().columns)


def test_a_grid_with_two_heights_is_refused():
    m = _module()
    grid = land_grid()
    grid.loc[0, "height"] = 80.0
    f = fleet()
    with pytest.raises(ValueError, match="height"):
        m.offshore_sites(f[f["offshore"]], grid, "BE")
