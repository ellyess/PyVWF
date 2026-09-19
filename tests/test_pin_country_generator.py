"""Pin the country-level data generator before it is split.

``vwf/datasets/generate_country_level_training_data.py`` is one 1,400-line
file with three groups that only ``main`` joins: grid-point generation, the
ENTSO-E observation fetch, and the writing of ``pyvwf_config.py``. Phase 3
splits it along those lines. Before that, ``main`` is pinned end to end.

The ENTSO-E fetch needs an API key and the network, so ``FakeFetcher`` stands
in for ``ENTSOEWindDataFetcher``: a deterministic hourly series per country or
zone, with the columns the real one returns. Everything the generator does
with that series (the train and test split, the zone aggregation for Norway
and Sweden, the file layout) is then pinned exactly.

Every CSV and the generated ``pyvwf_config.py`` are pinned by sha256. The
correction-region GeoJSON files are pinned by a per-cluster summary (area,
bounds, vertex count), because GeoJSON writers differ between environments in
ways that do not change a geometry.

Norway and Sweden build their grids from bidding-zone files that are local
only, so their grid case skips in CI; their observation case runs everywhere.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import vwf.datasets.generate_country_level_training_data as gen

ROOT = Path(__file__).resolve().parents[1]
PINS = Path(__file__).resolve().parent / "data" / "pins" / "country_generator"

KMEANS_COUNTRIES = ["NL", "FR", "BE", "ES", "IT", "PT", "IE"]
ZONE_COUNTRIES = ["NO", "SE"]
YEARS = ["--train-years", "2015", "2016", "--test-year", "2017"]

CASES = {
    # case: (arguments, needs the local bidding-zone files)
    "kmeans_grids_and_observations": (["--countries", *KMEANS_COUNTRIES, *YEARS], False),
    "zone_observations": (["--countries", *ZONE_COUNTRIES, "--skip-grids", *YEARS], False),
    "zone_grids": (["--countries", *ZONE_COUNTRIES, "--skip-observations", *YEARS], True),
}


class FakeFetcher:
    """Deterministic stand-in for ``ENTSOEWindDataFetcher``.

    Built from integer arithmetic and exact divisions only. A first version
    used ``np.sin`` and ``np.cos``, whose last bit differs between the macOS
    and Linux maths libraries, so its hashes held locally and failed in CI.
    """

    def calculate_capacity_factor(self, country, start, end, psr_type="all"):
        index = pd.date_range(start, end, freq="h")
        seed = sum(ord(c) for c in f"{country}{psr_type}")
        t = np.arange(len(index), dtype=np.int64)
        capacity = 800.0 + (seed % 97) * 10.0 + (t // 720) * 2.0
        cf = 0.15 + ((t * 7 + seed) % 24) / 64.0 + ((t // 24 + seed) % 9) / 128.0
        frame = pd.DataFrame({"generation_mw": capacity * cf, "capacity_mw": capacity}, index=index)
        frame["capacity_factor"] = frame["generation_mw"] / frame["capacity_mw"]
        return frame


def run_main(args: list[str], out: Path, monkeypatch) -> None:
    monkeypatch.setattr(gen, "ENTSOEWindDataFetcher", FakeFetcher)
    monkeypatch.setenv("ENTSOE_API_KEY", "not-a-real-key")
    monkeypatch.chdir(ROOT)  # the zone files are read by relative path
    monkeypatch.setattr(sys, "argv", ["generate", *args, "--output-dir", str(out)])
    assert gen.main() in (None, 0)


def geometry_summary(path: Path) -> pd.DataFrame:
    import geopandas as gpd

    g = gpd.read_file(path)
    b = g.geometry.bounds
    return pd.DataFrame(
        {
            "row": range(len(g)),
            "area": g.geometry.area.round(9),
            "minx": b.minx.round(9),
            "miny": b.miny.round(9),
            "maxx": b.maxx.round(9),
            "maxy": b.maxy.round(9),
            "vertices": g.geometry.apply(
                lambda s: (
                    len(s.exterior.coords)
                    if s.geom_type == "Polygon"
                    else sum(len(p.exterior.coords) for p in s.geoms)
                )
            ),
        }
    )


def outputs(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(sha256 per data file, geometry summary per GeoJSON)."""
    files = sorted(p for p in out.rglob("*") if p.is_file())
    digests = pd.DataFrame(
        [
            (p.relative_to(out).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest())
            for p in files
            if p.suffix != ".geojson"
        ],
        columns=["path", "sha256"],
    )
    geoms = [
        geometry_summary(p).assign(path=p.relative_to(out).as_posix())
        for p in files
        if p.suffix == ".geojson"
    ]
    geometry = (
        pd.concat(geoms, ignore_index=True)[
            ["path", "row", "area", "minx", "miny", "maxx", "maxy", "vertices"]
        ]
        if geoms
        else pd.DataFrame()
    )
    return digests, geometry


def _zone_files_present() -> bool:
    shapes = ROOT / "input" / "reference" / "shapes"
    return all(
        (shapes / f).is_file() for f in ("no_bidding_zones.geojson", "se_bidding_zones.geojson")
    )


@pytest.mark.parametrize(
    "case",
    [pytest.param(case, marks=pytest.mark.realdata) if CASES[case][1] else case for case in CASES],
)
def test_generator_main(case, tmp_path, monkeypatch):
    args, needs_zones = CASES[case]
    if needs_zones and not _zone_files_present():
        pytest.skip("the bidding-zone geometries are local only")
    run_main(args, tmp_path / "out", monkeypatch)
    digests, geometry = outputs(tmp_path / "out")
    pd.testing.assert_frame_equal(digests, pd.read_csv(PINS / f"{case}_sha256.csv"))
    if (PINS / f"{case}_geometry.csv").is_file():
        pd.testing.assert_frame_equal(
            geometry,
            pd.read_csv(PINS / f"{case}_geometry.csv"),
            check_dtype=False,
            rtol=0,
            atol=1e-9,
        )
    else:
        assert geometry.empty
