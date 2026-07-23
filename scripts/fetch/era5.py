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
region-specific to hardcode here — a new region needs only its config file.
This replaced six near-identical ``fetch_era5_<code>.py`` scripts; each box's
rationale now lives in the comments of its region TOML (e.g. why Chile stops
at -44 and excludes Magallanes, why Argentina spans Patagonia + Pampas).

Requests are BATCHED across months to cut queue waits. One CDS request can
carry several months of a single year (``--chunk-months``, default 6, the
value confirmed to be accepted; the CDS ~120k-field per-request cap allows up
to 12 for these boxes — one full year in a single request). The multi-month
netcdf that comes back is then SPLIT into the same per-month files a
month-at-a-time run would have produced, so nothing downstream changes:

    dataset   reanalysis-era5-single-levels, hourly, all days/times
    variables 100m u/v (wind) + 10m u/v (needed for the roughness calc)
    area      from the config bbox, converted to CDS [N, W, S, E] order
    grid      0.25 x 0.25
    format    netcdf, unarchived

A single multi-YEAR request would exceed the per-request field cap and be
rejected, so chunks never cross a year boundary — the ``year`` field stays a
single year and only ``month`` carries the list. ``--chunk-months 1`` restores
the old one-request-per-month behaviour byte-for-byte (no split step).

Output goes to ``<input-root>/era5/<file_tag>/era5_<code>_<YYYY>_<MM>.nc``
(``<input-root>`` is $PYVWF_INPUT if set, else ./input). NOTE the raw monthly
files land under the ``file_tag`` dir (e.g. era5/BR), which for the big boxes
(US, BR) is NOT the config's ``[era5] path`` — that points at the *_daily dir
produced afterwards by ``scripts/era5/combine.py``. Small boxes (NZ, CL, AR)
need no combine step and their config path is the raw dir directly.

Requests run sequentially and the script is resumable: completed months are
skipped (chunks are formed only from the months still missing), partial
downloads land in a .part file and are renamed only on success. Expect the CDS
queue, not bandwidth, to dominate wall-clock time — hence the batching.
"""
import argparse
import os
import sys
import time
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
MAX_CHUNK_MONTHS = 12  # a full year stays under the CDS ~120k-field request cap
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "regions"


def region_spec(code: str):
    path = CONFIG_DIR / f"{code.lower()}.toml"
    if not path.is_file():
        sys.exit(f"no region config at {path} — is {code!r} a shipped region?")
    return load_region(path)


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
    each chunk within one calendar year — the ``year`` field of a CDS request
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
    its own ``era5_<tag>_<YYYY>_<MM>.nc`` — the layout every consumer expects.
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True, help="Region code (e.g. cl, ar, nz)")
    ap.add_argument("--years", type=int, nargs="+", default=None,
                    help="Override the year span (default: train[0]..test[-1])")
    ap.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    ap.add_argument("--chunk-months", type=int, default=6,
                    help="Months per CDS request within a year (1-12, default 6). "
                         "1 restores one-request-per-month.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the request plan and exit without submitting")
    args = ap.parse_args()

    if not 1 <= args.chunk_months <= MAX_CHUNK_MONTHS:
        sys.exit(f"--chunk-months must be 1..{MAX_CHUNK_MONTHS} "
                 f"(a full year is the largest request under the CDS field cap)")

    spec = region_spec(args.region)
    # Filenames key on file_tag, not the region code: AU-NEM writes era5_au_*.
    tag = spec.file_tag.lower()
    years = args.years or list(range(spec.train_years[0], spec.test_years[-1] + 1))
    out_dir = output_dir(spec)

    plan = [
        (y, m, out_dir / f"era5_{tag}_{y}_{m:02d}.nc")
        for y in sorted(years) for m in sorted(args.months)
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
    client = cdsapi.Client()

    failures = []
    for i, (year, chunk) in enumerate(chunks, 1):
        months = [m for (_, m, _) in chunk]
        span = (f"{year}-{months[0]:02d}"
                if len(months) == 1 else
                f"{year}-{months[0]:02d}..{months[-1]:02d}")
        part = out_dir / f".era5_{tag}_{year}_{months[0]:02d}_chunk.nc.part"
        print(f"[{i}/{len(chunks)}] {span} ({len(months)} month(s))", flush=True)
        t0 = time.time()
        try:
            client.retrieve(DATASET, chunk_request(spec, year, months), str(part))
            written = split_and_write(part, chunk)
            part.unlink(missing_ok=True)
            total_mb = sum(p.stat().st_size for p in written) / 1e6
            print(f"    done in {time.time() - t0:.0f}s "
                  f"-> {len(written)} file(s), {total_mb:.0f} MB", flush=True)
        except Exception as exc:
            part.unlink(missing_ok=True)
            failures.append((year, months, str(exc)))
            print(f"    FAILED: {exc}", flush=True)

    if failures:
        print(f"\n{len(failures)} request(s) failed; re-run to retry just those "
              f"(completed months are skipped):")
        for year, months, err in failures:
            print(f"  {year} {months}: {err[:100]}")
        sys.exit(1)
    print("\nAll requested months present.")


if __name__ == "__main__":
    main()
