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

What it requests (per month, one CDS request each — a single multi-year
request exceeds the per-request item cap and gets rejected):

    dataset   reanalysis-era5-single-levels, hourly, all days/times
    variables 100m u/v (wind) + 10m u/v (needed for the roughness calc)
    area      from the config bbox, converted to CDS [N, W, S, E] order
    grid      0.25 x 0.25
    format    netcdf, unarchived

Output goes to ``<input-root>/era5/<file_tag>/era5_<code>_<YYYY>_<MM>.nc``
(``<input-root>`` is $PYVWF_INPUT if set, else ./input). NOTE the raw monthly
files land under the ``file_tag`` dir (e.g. era5/BR), which for the big boxes
(US, BR) is NOT the config's ``[era5] path`` — that points at the *_daily dir
produced afterwards by ``scripts/era5/combine.py``. Small boxes (NZ, CL, AR)
need no combine step and their config path is the raw dir directly.

Requests run sequentially and the script is resumable: completed months are
skipped, partial downloads land in a .part file and are renamed only on
success. Expect the CDS queue, not bandwidth, to dominate wall-clock time.
"""
import argparse
import os
import sys
import time
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


def month_request(spec, year: int, month: int) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, 32)],  # CDS ignores invalid days
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": cds_area(spec.bbox),
        "grid": GRID,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True, help="Region code (e.g. cl, ar, nz)")
    ap.add_argument("--years", type=int, nargs="+", default=None,
                    help="Override the year span (default: train[0]..test[-1])")
    ap.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the request plan and exit without submitting")
    args = ap.parse_args()

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
    print(f"Region {spec.code}: bbox {spec.bbox} -> CDS area {cds_area(spec.bbox)}")
    print(f"Output directory: {out_dir}")
    print(f"{len(plan)} month(s) in plan, {len(plan) - len(todo)} already present, "
          f"{len(todo)} to fetch.")

    if args.dry_run:
        for y, m, p in todo:
            print(f"  would request {y}-{m:02d} -> {p.name}")
        return
    if not todo:
        return

    import cdsapi  # imported here so --dry-run works without it installed

    out_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    failures = []
    for i, (y, m, path) in enumerate(todo, 1):
        part = path.with_suffix(".nc.part")
        print(f"[{i}/{len(todo)}] {y}-{m:02d} -> {path.name}", flush=True)
        t0 = time.time()
        try:
            client.retrieve(DATASET, month_request(spec, y, m), str(part))
            part.rename(path)
            print(f"    done in {time.time() - t0:.0f}s "
                  f"({path.stat().st_size / 1e6:.0f} MB)", flush=True)
        except Exception as exc:
            part.unlink(missing_ok=True)
            failures.append((y, m, str(exc)))
            print(f"    FAILED: {exc}", flush=True)

    if failures:
        print(f"\n{len(failures)} month(s) failed; re-run to retry just those:")
        for y, m, err in failures:
            print(f"  {y}-{m:02d}: {err[:100]}")
        sys.exit(1)
    print("\nAll requested months present.")


if __name__ == "__main__":
    main()
