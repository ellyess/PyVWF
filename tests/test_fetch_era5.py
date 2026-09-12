"""The ERA5 fetch script's box, tag and year resolution (scripts/fetch/era5.py).

A download need not belong to a region. The extended European box serves
eleven regions and is none of them, so it has no config to read a box or a
year span from, and the script grew ``--bbox``, ``--years``, ``--code`` and
``--file-tag`` to fetch one. The same flags override a region config field by
field. These tests pin what is resolved from where, and that an unusable box
stops the run before a request is submitted rather than after.

The script submits through the caller's CDS credentials, so nothing here
touches the network: ``cdsapi`` is imported inside ``main`` only when there is
something to fetch.
"""
import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "fetch_era5",
    Path(__file__).resolve().parents[1] / "scripts" / "fetch" / "era5.py",
)
fetch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fetch)

EU_BOX = [-12.0, 31.5, 36.0, 72.0]


def args(**kwargs):
    base = {"region": None, "code": None, "bbox": None, "file_tag": None, "years": None}
    return Namespace(**{**base, **kwargs})


def test_a_bare_box_needs_only_the_flags():
    spec, years = fetch.resolve_spec(
        args(code="eu", bbox=EU_BOX, file_tag="EU_2026-09", years=[2016, 2015]))
    assert (spec.code, spec.file_tag) == ("eu", "EU_2026-09")
    assert spec.bbox == (-12.0, 31.5, 36.0, 72.0)
    assert years == [2015, 2016]


def test_the_file_tag_defaults_to_the_code():
    spec, _ = fetch.resolve_spec(args(code="eu", bbox=EU_BOX, years=[2015]))
    assert spec.file_tag == "EU"


@pytest.mark.parametrize("given, missing", [
    ({"bbox": EU_BOX, "years": [2015]}, "--code"),
    ({"code": "eu", "years": [2015]}, "--bbox"),
    ({"code": "eu", "bbox": EU_BOX}, "--years"),
])
def test_a_bare_box_cannot_fall_back_to_a_config(given, missing):
    """Nothing supplies these without a region, so the run stops naming them."""
    with pytest.raises(SystemExit) as e:
        fetch.resolve_spec(args(**given))
    assert missing in str(e.value)


def test_a_region_supplies_every_default():
    spec, years = fetch.resolve_spec(args(region="nz"))
    assert (spec.code, spec.file_tag) == ("NZ", "NZ")
    assert spec.bbox == (166.0, 179.0, -48.0, -36.0)
    assert years == [2019, 2020, 2021, 2022, 2023, 2024]


def test_a_flag_replaces_one_field_and_leaves_the_rest():
    spec, years = fetch.resolve_spec(
        args(region="nz", bbox=[160.0, 180.0, -50.0, -30.0], file_tag="NZ_wide"))
    assert spec.bbox == (160.0, 180.0, -50.0, -30.0)
    assert spec.file_tag == "NZ_wide"
    assert spec.code == "NZ"                      # not overridden
    assert years == [2019, 2020, 2021, 2022, 2023, 2024]   # not overridden


@pytest.mark.parametrize("bbox", [
    [31.5, -12.0, 36.0, 72.0],   # W east of E
    [-12.0, 31.5, 72.0, 36.0],   # S north of N
    [-12.0, 31.5, 36.0, 95.0],   # off the globe
])
def test_an_unusable_box_stops_the_run(bbox):
    with pytest.raises(SystemExit):
        fetch.check_bbox(bbox)


def test_the_request_carries_the_overridden_box_in_cds_order():
    spec, _ = fetch.resolve_spec(args(code="eu", bbox=EU_BOX, years=[2015]))
    request = fetch.chunk_request(spec, 2015, [1, 2, 3])
    assert request["area"] == [72.0, -12.0, 36.0, 31.5]   # N, W, S, E
    assert request["year"] == ["2015"] and request["month"] == ["01", "02", "03"]
    assert len(request["time"]) == 24 and len(request["day"]) == 31
    assert request["data_format"] == "netcdf"


def test_the_year_span_fixes_the_request_count():
    """Nine years at three months a request is the 36 the European box needs."""
    _, years = fetch.resolve_spec(
        args(code="eu", bbox=EU_BOX, years=list(range(2015, 2024))))
    todo = [(y, m, Path(f"{y}_{m}.nc")) for y in years for m in range(1, 13)]
    chunks = fetch.plan_chunks(todo, 3)
    assert len(chunks) == 36
    assert all(len({y for (y, _, _) in chunk}) == 1 for _, chunk in chunks)
