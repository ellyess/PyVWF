"""Region shapes must not silently omit a country's islands.

``country_shapes.geojson`` carries only the largest landmasses: the DK polygon
holds Jutland, Zealand and Funen and nothing else, 39,208 km² against a
published land area of 42,943, which leaves 12.9% of the DK onshore fleet
standing outside its own country polygon. Greece (-10.1%) and Croatia (-9.0%)
are truncated the same way.

``repair_region_shape`` merges the missing landmasses back in. The property
worth pinning is *how it decides who owns an island*: attribution follows
territorial waters, not proximity to land. Bornholm is 34 km from Sweden and
143 km from the Danish mainland, so a nearest-land rule hands it to Sweden;
it sits inside Denmark's EEZ, so the EEZ rule gets it right. The synthetic
fixture below reproduces exactly that inversion.
"""
import json

import numpy as np
import pytest

pytest.importorskip("geopandas")
pytest.importorskip("shapely")

from shapely.geometry import box  # noqa: E402

import vwf.clustering as clustering  # noqa: E402
from vwf.clustering import get_country_shape, repair_region_shape  # noqa: E402
from vwf.config import PyVWFPaths  # noqa: E402


# Region A's mainland. The repair window is bounds +3 deg lon / +2 deg lat.
A_MAINLAND = (10.0, 55.0, 12.0, 57.0)
# Nearer to B's land (0.6 deg) than to A's (1.6 deg), but inside A's waters.
# This is the Bornholm case: land distance and sovereignty disagree.
ISLAND_A = (14.0, 55.8, 14.4, 56.2)
# Inside B's waters; must not be claimed by A.
ISLAND_B = (14.7, 55.8, 14.9, 56.2)
B_MAINLAND = (15.0, 55.0, 17.0, 57.0)
A_WATERS = (12.0, 54.0, 14.6, 58.0)
B_WATERS = (14.6, 54.0, 15.5, 58.0)


def _ring(bounds):
    """A closed LineString ring, which is what polygonize consumes."""
    return list(box(*bounds).exterior.coords)


def _write_geojson(path, features):
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"name": name},
             "geometry": geom}
            for name, geom in features
        ],
    }))


@pytest.fixture
def shape_files(tmp_path, monkeypatch):
    """Point PyVWF's shape paths at a synthetic two-country world."""
    coast = tmp_path / "coastlines.geojson"
    _write_geojson(coast, [
        ("c", {"type": "LineString", "coordinates": _ring(A_MAINLAND)}),
        ("c", {"type": "LineString", "coordinates": _ring(ISLAND_A)}),
        ("c", {"type": "LineString", "coordinates": _ring(ISLAND_B)}),
    ])
    countries = tmp_path / "country_shapes.geojson"
    _write_geojson(countries, [
        ("A", box(*A_MAINLAND).__geo_interface__),
        ("B", box(*B_MAINLAND).__geo_interface__),
    ])
    offshore = tmp_path / "offshore_shapes.geojson"
    _write_geojson(offshore, [
        ("A", box(*A_WATERS).__geo_interface__),
        ("B", box(*B_WATERS).__geo_interface__),
    ])

    monkeypatch.setattr(PyVWFPaths, "COASTLINES", coast)
    monkeypatch.setattr(PyVWFPaths, "COUNTRY_SHAPES", countries)
    monkeypatch.setattr(PyVWFPaths, "OFFSHORE_SHAPES", offshore)
    # load_region_shapes memoises in module globals; clear so the fixture wins.
    monkeypatch.setattr(clustering, "_COUNTRY_SHAPES", None)
    monkeypatch.setattr(clustering, "_OFFSHORE_SHAPES", None)
    clustering._REPAIRED_SHAPES.clear()
    yield tmp_path
    clustering._REPAIRED_SHAPES.clear()


def test_island_follows_territorial_waters_not_nearest_land(shape_files):
    """The island in A's waters is claimed by A even though B's land is nearer."""
    base = box(*A_MAINLAND)
    repaired = repair_region_shape(base, "A")

    island_a = box(*ISLAND_A)
    assert repaired.contains(island_a.centroid), (
        "island inside A's EEZ was not claimed; attribution has fallen back to "
        "nearest land, which is the bug that loses Bornholm"
    )
    # Sanity: the fixture really does encode the inversion being guarded.
    assert island_a.distance(box(*B_MAINLAND)) < island_a.distance(base)


def test_island_in_other_countrys_waters_is_not_claimed(shape_files):
    """Repair must not annex a neighbour's islands."""
    repaired = repair_region_shape(box(*A_MAINLAND), "A")
    assert not repaired.contains(box(*ISLAND_B).centroid)


def test_fleet_points_claim_a_landmass(shape_files):
    """A landmass carrying the fleet is kept whatever the EEZ test decides."""
    fleet = np.array([[14.8, 56.0]])   # sits on ISLAND_B, in B's waters
    repaired = repair_region_shape(box(*A_MAINLAND), "A", fleet_xy=fleet)
    assert repaired.contains(box(*ISLAND_B).centroid), (
        "a turbine's own island was dropped; the fleet override is what keeps "
        "sites from being orphaned when the EEZ files disagree"
    )


def test_repair_is_opt_in(shape_files):
    """Existing callers keep the exact geometry they have always received."""
    plain = get_country_shape("A")
    repaired = get_country_shape("A", repair=True)
    assert plain.equals(box(*A_MAINLAND))
    assert repaired.area > plain.area


def test_missing_coastline_returns_input_unchanged(shape_files, monkeypatch, tmp_path):
    """No coastline file is a warning and a no-op, never an exception."""
    monkeypatch.setattr(PyVWFPaths, "COASTLINES", tmp_path / "absent.geojson")
    base = box(*A_MAINLAND)
    with pytest.warns(UserWarning, match="Coastline file not found"):
        assert repair_region_shape(base, "A") is base


# --------------------------------------------------------------- real data ---

_REAL = PyVWFPaths.COUNTRY_SHAPES.exists() and PyVWFPaths.COASTLINES.exists()
pytestmark_real = pytest.mark.skipif(not _REAL, reason="bundled shape files absent")


@pytestmark_real
def test_denmark_regains_bornholm():
    """The concrete case: DK gains Bornholm and lands near its published area."""
    clustering._REPAIRED_SHAPES.clear()
    plain = get_country_shape("DK")
    repaired = get_country_shape("DK", repair=True)

    from shapely.geometry import Point
    bornholm = Point(14.9, 55.13)
    assert not plain.contains(bornholm)
    assert repaired.contains(bornholm)

    import geopandas as gpd
    areas = gpd.GeoSeries([plain, repaired], crs="EPSG:4326").to_crs("EPSG:3035").area / 1e6
    # Published land area is 42,943 km2; unrepaired is 39,208.
    assert areas.iloc[0] < 40_000
    assert 42_000 < areas.iloc[1] < 43_500
