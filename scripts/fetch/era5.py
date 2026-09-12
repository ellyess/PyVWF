#!/usr/bin/env python
"""Fetch the ERA5 subset for a region from the Copernicus CDS.

USER-EXECUTED. This script submits requests through your CDS credentials, so
running it is yours to do, like every other download on this project:

    pip install cdsapi              # not a PyVWF dependency
    # ~/.cdsapirc must hold your CDS url + key
    python scripts/fetch/era5.py --region cl            # all months, from config
    python scripts/fetch/era5.py --region cl --dry-run  # list requests only
    python scripts/fetch/era5.py --region ar --years 2024

One script for every region: the bounding box comes from the region TOML
(``[era5] bbox`` = [W, E, S, N]) and the year span from the training/test
window (``train_years[0]`` .. ``test_years[-1]``), so there is nothing
region-specific to hardcode here: a new region needs only its config file.

A download need not belong to a region. ``--bbox``, ``--years``, ``--code``
and ``--file-tag`` fetch a bare box, with no config at all, which is how the
extended European box was fetched: it serves eleven regions and is not one:

    python scripts/fetch/era5.py --code eu --file-tag EU_2026-09 \
        --bbox -12 31.5 36 72 --years 2015 2016 2017 2018 2019 2020 2021 2022 2023

The same options override a region config field by field, so a region can be
re-fetched over a wider box without editing its TOML. Overriding leaves the
config untouched, so a run from the config and a run from the override are not
the same download; the directory names them apart.

This replaced six near-identical ``fetch_era5_<code>.py`` scripts; each box's
rationale now lives in the comments of its region TOML (e.g. why Chile stops
at -44 and excludes Magallanes, why Argentina spans Patagonia + Pampas).

Requests are BATCHED across months to cut queue waits. One CDS request can
carry several months of a single year (``--chunk-months``, default 3). The
ceiling is the CDS *cost* limit, which counts fields = variables x days x
hours and ignores the area, so it is the same for every region: 3 months
(~8.6k fields) is accepted, 6 months (~17k fields) is rejected with a 403
"cost limits exceeded / request too large". Note the interactive CDS web form
allows larger selections than the API does, so a size that works in the
browser can still be refused here. The multi-month netcdf that comes back is
then SPLIT into the same per-month files a month-at-a-time run would have
produced, so nothing downstream changes:

    dataset   reanalysis-era5-single-levels, hourly, all days/times
    variables 100m u/v (wind) + 10m u/v (needed for the roughness calc)
    area      from the config bbox, converted to CDS [N, W, S, E] order
    grid      0.25 x 0.25
    format    netcdf, unarchived

A single multi-YEAR request would exceed the per-request field cap and be
rejected, so chunks never cross a year boundary: the ``year`` field stays a
single year and only ``month`` carries the list. ``--chunk-months 1`` restores
the old one-request-per-month behaviour byte-for-byte (no split step).

Output goes to ``<input-root>/era5/<file_tag>/era5_<code>_<YYYY>_<MM>.nc``
(``<input-root>`` is $PYVWF_INPUT if set, else ./input). NOTE the raw monthly
files land under the ``file_tag`` dir (e.g. era5/BR), which for the big boxes
(US, BR) is NOT the config's ``[era5] path``: that points at the *_daily dir
produced afterwards by ``scripts/era5/combine.py``. Small boxes (NZ, CL, AR)
need no combine step and their config path is the raw dir directly.

The script is resumable: completed months are skipped (chunks are formed only
from the months still missing), partial downloads land in a .part file and are
renamed only on success. Expect the CDS queue, not bandwidth, to dominate
wall-clock time; hence the batching. The extended European box measured about
11 minutes per request, of which roughly 40 seconds was transfer.

``--workers N`` submits N requests at once through a thread pool, which
overlaps those queue waits. It defaults to 1, so nothing changes for an
existing caller, and is capped at 6. Four is the recommended value, for
reasons that are worth stating because none of them is obvious:

- **ECMWF documents no per-user concurrency limit.** The CDS documentation says
  only that limits exist, change with system load, and that requests which
  would exceed them are QUEUED rather than refused. The one concrete number
  anywhere ("most ECMWF services are limited to 20 concurrent requests",
  default 2) comes from a third-party R package, predates the current
  datastores backend, and is not relied on here.
- **The failure mode is therefore invisible.** Exceeding the limit does not
  raise; the quality-of-service scheduler queues and may deprioritise a heavy
  user, which cannot be measured from the client. A small pool takes most of
  the available overlap for almost none of that risk, which is why the cap is
  6 rather than 20.
- **A local reason for 4:** ``split_and_write`` opens each chunk with xarray,
  and an EU chunk is about 478 MB. This machine has 16 GB and has OOM-killed a
  test suite before.

**What four workers actually bought, once, on one box.** The extended European
download took 4.21 hours for 105 months at ``--workers 4``, with a median chunk
of 29 minutes against a single sequential sample of 11. That is roughly 1.5
times faster overall, not the 4 the worker count suggests: each request waited
longer while four were in flight.

**Do not cite 1.5 as a constant.** The sequential baseline is ONE request at a
different time of day, and CDS load varies on its own, so this is indicative
and nothing more. The only thing it suggests, and does not establish, is that
the throughput ceiling is per user rather than per request in flight, which
would mean raising the cap buys little. That is at least consistent with ECMWF
documenting no number while evidently applying one. Measure again before
relying on any of it.

**The client is not shared between workers.** ``requests.Session`` is not
thread-safe, so each worker builds its own client. That is enough on the path
this key takes: a key without a colon routes ``cdsapi.Client`` to
``ecmwf.datastores.legacy_client.LegacyClient``, whose ``session`` argument
defaults to None and builds a fresh Session per client. It would NOT be enough
on the older path: ``cdsapi.api.Client`` declares ``session=requests.Session()``
as a default argument, one instance evaluated at import and shared by every
client built without an explicit session. Anyone changing the key format needs
to know that, and to pass a session per worker if the old path comes back.

Submit-and-poll was considered and rejected.
``Client(wait_until_complete=False).retrieve`` does return an
``ecmwf.datastores.Remote`` with a request id, so all 36 requests could be
submitted at once and polled. It buys no real concurrency, since the server
queues them under the same limits, and it adds durable state: a request id per
chunk that must be persisted or the request is orphaned on a restart. The
resume rule here is "a month is done when its file exists", which needs no
state at all, and the thread pool keeps that property.
"""
import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vwf.harness.regions import load_region  # noqa: E402

DATASET = "reanalysis-era5-single-levels"
VARIABLES = [
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
GRID = [0.25, 0.25]
# Upper bound for --chunk-months. The practical ceiling is the CDS cost limit
# (3 accepted, 6 rejected for hourly all-day requests); this only guards the CLI
# so an over-large value fails fast here rather than as a 403 from the server.
MAX_CHUNK_MONTHS = 12
# Requests in flight at once. ECMWF documents no per-user concurrency limit and
# queues rather than refusing what exceeds it, so the ceiling here is a
# judgement about unmeasurable risk, not a published figure: see the docstring.
MAX_WORKERS = 6
RECOMMENDED_WORKERS = 4
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "regions"


def region_spec(code: str):
    path = CONFIG_DIR / f"{code.lower()}.toml"
    if not path.is_file():
        sys.exit(f"no region config at {path}: is {code!r} a shipped region?")
    return load_region(path)


@dataclass(frozen=True)
class DownloadSpec:
    """What a download needs: a name, a box and a directory to write into.

    A region config supplies all three, but a box that serves several regions
    belongs to none of them, so the fields can be given directly instead.
    """

    code: str
    bbox: tuple[float, float, float, float]
    file_tag: str


def check_bbox(bbox) -> tuple[float, float, float, float]:
    """Validate a [W, E, S, N] box, as ``RegionSpec`` does for a config."""
    w, e, s, n = (float(v) for v in bbox)
    if w >= e or s >= n:
        sys.exit(f"bbox must be [W, E, S, N] with W < E and S < N, got {list(bbox)}")
    if not (-180 <= w and e <= 360 and -90 <= s and n <= 90):
        sys.exit(f"bbox is outside the globe: {list(bbox)}")
    return (w, e, s, n)


def resolve_spec(args) -> tuple[DownloadSpec, list[int]]:
    """The box, tag and years to fetch, from a region config or from the flags.

    With ``--region`` the config supplies the defaults and any flag given
    replaces that one field. Without it the box is bare, so ``--code``,
    ``--bbox`` and ``--years`` are all required: nothing can supply them.
    """
    if args.region is None:
        missing = [n for n, v in
                   (("--code", args.code), ("--bbox", args.bbox), ("--years", args.years))
                   if not v]
        if missing:
            sys.exit(f"without --region these are required: {', '.join(missing)}")
        code = args.code
        return DownloadSpec(code, check_bbox(args.bbox),
                            args.file_tag or code.upper()), sorted(args.years)

    spec = region_spec(args.region)
    years = sorted(args.years) if args.years else list(
        range(spec.train_years[0], spec.test_years[-1] + 1))
    return DownloadSpec(
        args.code or spec.code,
        check_bbox(args.bbox) if args.bbox else tuple(spec.bbox),
        args.file_tag or spec.file_tag,
    ), years


def cds_area(bbox) -> list[float]:
    """Config bbox [W, E, S, N] -> CDS area [N, W, S, E]."""
    w, e, s, n = bbox
    return [n, w, s, e]


def output_dir(spec) -> Path:
    root = Path(os.environ.get("PYVWF_INPUT", "input"))
    return root / "era5" / spec.file_tag


def chunk_request(spec, year: int, months: list[int]) -> dict:
    """One CDS request for several months of a single year (all days/times)."""
    return {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [str(year)],
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in range(1, 32)],  # CDS ignores invalid days
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": cds_area(spec.bbox),
        "grid": GRID,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def plan_chunks(todo, chunk_months: int):
    """Group missing (year, month, path) into per-year chunks of <= N months.

    ``todo`` is ordered by (year, month), so grouping by year and slicing keeps
    each chunk within one calendar year: the ``year`` field of a CDS request
    must stay a single value or the month list would fan out across years.
    """
    chunks = []
    for year, grp in groupby(todo, key=lambda t: t[0]):
        items = list(grp)
        for i in range(0, len(items), chunk_months):
            chunks.append((year, items[i:i + chunk_months]))
    return chunks


def split_and_write(part_path: Path, chunk) -> list[Path]:
    """Write the requested per-month files out of a downloaded chunk.

    A one-month chunk is just renamed (byte-for-byte the old behaviour). A
    multi-month chunk is split on the time coordinate so each month lands in
    its own ``era5_<tag>_<YYYY>_<MM>.nc``: the layout every consumer expects.
    """
    if len(chunk) == 1:
        (_, _, path) = chunk[0]
        part_path.rename(path)
        return [path]

    import numpy as np
    import xarray as xr

    written = []
    with xr.open_dataset(part_path) as ds:
        tname = "valid_time" if "valid_time" in ds.coords else "time"
        month_of = ds[tname].dt.month.values
        for (_, m, path) in chunk:
            idx = np.nonzero(month_of == m)[0]
            if idx.size == 0:
                raise ValueError(f"chunk download is missing month {m:02d}")
            mpart = path.with_suffix(".nc.part")
            ds.isel({tname: idx}).to_netcdf(mpart)
            mpart.rename(path)
            written.append(path)
    return written


def thread_local_clients(factory):
    """A callable handing each thread its own client.

    ``requests.Session`` is not thread-safe, and a client owns one, so workers
    must not share a client. Built lazily, so a sequential run still builds
    exactly one.
    """
    local = threading.local()

    def client():
        if not hasattr(local, "client"):
            local.client = factory()
        return local.client

    return client


def fetch_chunk(clients, spec, out_dir: Path, tag: str, year: int, chunk) -> dict:
    """Fetch one chunk and split it into its months. Never raises.

    Returns what happened, for the caller to report: a worker that raised into
    a thread pool would lose the other chunks. The chunk's own ``.part`` file
    is named for its year and first month, so two workers cannot collide, and
    it is removed whether the request succeeds or fails.

    A failure part-way through the split leaves the months already renamed in
    place and the rest absent. That is resumable, not corrupt: the next run
    fetches only what is missing.
    """
    months = [m for (_, m, _) in chunk]
    span = (f"{year}-{months[0]:02d}" if len(months) == 1
            else f"{year}-{months[0]:02d}..{months[-1]:02d}")
    part = out_dir / f".era5_{tag}_{year}_{months[0]:02d}_chunk.nc.part"
    result = {"year": year, "months": months, "span": span,
              "written": [], "mb": 0.0, "seconds": 0.0, "error": None}
    # Printed on submission as well as on completion: a chunk waits minutes in
    # the CDS queue, and a run that prints nothing until the first one lands
    # looks hung. Lines carry their span, since workers interleave.
    print(f"  {span} submitted ({len(months)} month(s))", flush=True)
    t0 = time.time()
    try:
        clients().retrieve(DATASET, chunk_request(spec, year, months), str(part))
        written = split_and_write(part, chunk)
        result["written"] = written
        result["mb"] = sum(p.stat().st_size for p in written) / 1e6
    except Exception as exc:  # noqa: BLE001  (reported, not swallowed)
        result["error"] = str(exc)
    finally:
        part.unlink(missing_ok=True)
        result["seconds"] = time.time() - t0
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default=None,
                    help="Region code (e.g. cl, ar, nz). Omit to fetch a bare box, "
                         "which then needs --code, --bbox and --years")
    ap.add_argument("--years", type=int, nargs="+", default=None,
                    help="Override the year span (default: train[0]..test[-1])")
    ap.add_argument("--bbox", type=float, nargs=4, default=None,
                    metavar=("W", "E", "S", "N"),
                    help="Override the bounding box, in config order [W, E, S, N]")
    ap.add_argument("--code", default=None,
                    help="Override the region code. Files key on --file-tag, not on "
                         "this, since a code may carry a hyphen (AU-NEM writes "
                         "era5_au_*)")
    ap.add_argument("--file-tag", default=None,
                    help="Override the output directory under <input-root>/era5/")
    ap.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    ap.add_argument("--chunk-months", type=int, default=3,
                    help="Months per CDS request within a year (1-12, default 3, "
                         "the largest accepted by the CDS cost limit; 6+ is "
                         "rejected). 1 restores one-request-per-month.")
    ap.add_argument("--workers", type=int, default=1,
                    help=f"Requests in flight at once (1-{MAX_WORKERS}, default 1). "
                         "The CDS queue dominates wall-clock time, so a small pool "
                         "overlaps it; see the module docstring for why the cap is "
                         f"{MAX_WORKERS} and {RECOMMENDED_WORKERS} is recommended")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the request plan and exit without submitting")
    args = ap.parse_args()

    if not 1 <= args.chunk_months <= MAX_CHUNK_MONTHS:
        sys.exit(f"--chunk-months must be 1..{MAX_CHUNK_MONTHS} "
                 f"(a full year is the largest request under the CDS field cap)")
    if not 1 <= args.workers <= MAX_WORKERS:
        sys.exit(f"--workers must be 1..{MAX_WORKERS}; the limit is a judgement "
                 "about an undocumented CDS limit, not a published figure")

    spec, years = resolve_spec(args)
    # Filenames key on file_tag, not the region code: AU-NEM writes era5_au_*.
    tag = spec.file_tag.lower()
    out_dir = output_dir(spec)

    plan = [
        (y, m, out_dir / f"era5_{tag}_{y}_{m:02d}.nc")
        for y in years for m in sorted(args.months)
    ]
    todo = [(y, m, p) for y, m, p in plan if not p.is_file()]
    chunks = plan_chunks(todo, args.chunk_months)
    print(f"Region {spec.code}: bbox {spec.bbox} -> CDS area {cds_area(spec.bbox)}")
    print(f"Output directory: {out_dir}")
    print(f"{len(plan)} month(s) in plan, {len(plan) - len(todo)} already present, "
          f"{len(todo)} to fetch in {len(chunks)} request(s) "
          f"(up to {args.chunk_months} month(s) each).")

    if args.dry_run:
        for year, chunk in chunks:
            months = [m for (_, m, _) in chunk]
            names = ", ".join(p.name for (_, _, p) in chunk)
            print(f"  would request {year} months {months} -> {names}")
        return
    if not todo:
        return

    import cdsapi  # imported here so --dry-run works without it installed

    out_dir.mkdir(parents=True, exist_ok=True)
    clients = thread_local_clients(cdsapi.Client)

    done = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_chunk, clients, spec, out_dir, tag, year, chunk)
                   for year, chunk in chunks]
        # No cancellation on failure: one refused chunk must not lose the
        # others, and every month it did not write is simply fetched again by
        # the next run.
        for future in as_completed(futures):
            result = future.result()
            done += 1
            span = result["span"]
            if result["error"] is None:
                print(f"[{done}/{len(chunks)}] {span} done in {result['seconds']:.0f}s "
                      f"-> {len(result['written'])} file(s), {result['mb']:.0f} MB",
                      flush=True)
            else:
                failures.append((result["year"], result["months"], result["error"]))
                print(f"[{done}/{len(chunks)}] {span} FAILED: {result['error']}", flush=True)

    if failures:
        print(f"\n{len(failures)} request(s) failed; re-run to retry just those "
              f"(completed months are skipped):")
        for year, months, err in failures:
            print(f"  {year} {months}: {err[:100]}")
        sys.exit(1)
    print("\nAll requested months present.")


if __name__ == "__main__":
    main()
