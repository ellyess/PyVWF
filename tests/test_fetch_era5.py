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


# --- the worker pool -------------------------------------------------------
#
# The CDS queue dominates wall-clock time, so requests overlap through a thread
# pool. Three properties make that safe, and each is pinned below: a worker
# never raises into the pool, so one refused chunk cannot lose the others; a
# chunk's temporary file is named for its own year and first month, so two
# workers cannot collide; and each worker builds its own client, because
# requests.Session is not thread-safe.

def _chunk(tmp_path, year, months):
    return [(year, m, tmp_path / f"era5_eu_{year}_{m:02d}.nc") for m in months]


def _months_file(path, year, months):
    """A stand-in for a downloaded chunk: hourly steps across the months."""
    import numpy as np
    import xarray as xr

    times = np.concatenate([
        np.arange(f"{year}-{m:02d}-01", f"{year}-{m:02d}-02", dtype="datetime64[h]")
        for m in months
    ])
    xr.Dataset(
        {"u10": (("valid_time",), np.arange(len(times), dtype="float32"))},
        coords={"valid_time": times},
    ).to_netcdf(path)


class FakeClient:
    """Writes the target file the way ``retrieve`` does, or refuses."""

    def __init__(self, year, months, refuse_months=()):
        self.year, self.months, self.refuse = year, months, set(refuse_months)
        self.calls = []

    def retrieve(self, dataset, request, target):
        got = [int(m) for m in request["month"]]
        self.calls.append(got)
        if self.refuse & set(got):
            raise RuntimeError("cost limits exceeded")
        _months_file(target, int(request["year"][0]), got)


def test_a_worker_reports_a_refusal_instead_of_raising(tmp_path):
    """A worker that raised would take the other chunks down with it."""
    spec, _ = fetch.resolve_spec(args(code="eu", bbox=EU_BOX, years=[2015]))
    client = FakeClient(2015, [1], refuse_months=[1])
    result = fetch.fetch_chunk(lambda: client, spec, tmp_path, "eu", 2015,
                               _chunk(tmp_path, 2015, [1]))
    assert "cost limits exceeded" in result["error"]
    assert result["written"] == []
    assert list(tmp_path.glob("*.nc")) == []          # no month file
    assert list(tmp_path.glob("*.part")) == []        # and no partial left behind


def test_a_worker_writes_its_months_and_clears_its_part_file(tmp_path):
    spec, _ = fetch.resolve_spec(args(code="eu", bbox=EU_BOX, years=[2015]))
    chunk = _chunk(tmp_path, 2015, [1, 2, 3])
    result = fetch.fetch_chunk(lambda: FakeClient(2015, [1, 2, 3]), spec, tmp_path,
                               "eu", 2015, chunk)
    assert result["error"] is None
    assert [p.name for p in result["written"]] == [p.name for (_, _, p) in chunk]
    assert all(p.is_file() for (_, _, p) in chunk)
    assert list(tmp_path.glob("*.part")) == []
    assert result["span"] == "2015-01..03"


def test_a_split_that_fails_part_way_leaves_what_it_wrote(tmp_path):
    """Resumable, not corrupt: the months already renamed stay, the rest are
    absent, and the next run fetches only those."""
    spec, _ = fetch.resolve_spec(args(code="eu", bbox=EU_BOX, years=[2015]))
    chunk = _chunk(tmp_path, 2015, [1, 2, 3])

    class ShortClient(FakeClient):
        def retrieve(self, dataset, request, target):      # month 3 never arrives
            _months_file(target, 2015, [1, 2])

    result = fetch.fetch_chunk(lambda: ShortClient(2015, [1, 2, 3]), spec, tmp_path,
                               "eu", 2015, chunk)
    assert "missing month 03" in result["error"]
    written = sorted(p.name for p in tmp_path.glob("*.nc"))
    assert written == ["era5_eu_2015_01.nc", "era5_eu_2015_02.nc"]
    assert list(tmp_path.glob("*.part")) == []


def test_each_thread_builds_its_own_client():
    """requests.Session is not thread-safe, so a client is never shared."""
    from concurrent.futures import ThreadPoolExecutor

    built = []

    def factory():
        built.append(object())
        return built[-1]

    clients = fetch.thread_local_clients(factory)
    with ThreadPoolExecutor(max_workers=3) as pool:
        got = list(pool.map(lambda _: id(clients()), range(24)))
    assert len(set(got)) == len(built) <= 3     # one per thread, not one per task
    assert len(built) >= 2                      # and more than one thread ran


def test_a_sequential_run_still_builds_exactly_one_client():
    built = []
    clients = fetch.thread_local_clients(lambda: built.append(1) or "client")
    clients(), clients(), clients()
    assert len(built) == 1


def test_one_refused_chunk_does_not_lose_the_others(tmp_path):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    spec, _ = fetch.resolve_spec(args(code="eu", bbox=EU_BOX, years=[2015]))
    chunks = [(2015, _chunk(tmp_path, 2015, ms))
              for ms in ([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12])]
    clients = fetch.thread_local_clients(
        lambda: FakeClient(2015, list(range(1, 13)), refuse_months=[7, 8, 9]))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fetch.fetch_chunk, clients, spec, tmp_path, "eu", y, c)
                   for y, c in chunks]
        results = [f.result() for f in as_completed(futures)]

    failed = [r for r in results if r["error"]]
    assert len(failed) == 1 and failed[0]["months"] == [7, 8, 9]
    assert len(sorted(tmp_path.glob("*.nc"))) == 9     # the other three chunks
    assert list(tmp_path.glob("*.part")) == []


@pytest.mark.parametrize("workers", ["0", "7"])
def test_the_worker_count_is_bounded(workers, monkeypatch):
    monkeypatch.setattr("sys.argv", ["era5.py", "--region", "nz", "--workers", workers])
    with pytest.raises(SystemExit) as e:
        fetch.main()
    assert "--workers must be 1.." in str(e.value)
