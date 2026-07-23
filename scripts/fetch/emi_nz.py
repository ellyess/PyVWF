#!/usr/bin/env python
"""Fetch the EMI (NZ Electricity Authority) open data for the NZ region.

USER-EXECUTED. Downloads are yours to run, like every other download on this
project. No registration or credentials are needed — EMI datasets are plain
public CSV downloads (each URL 302-redirects to an open Azure blob):

    python scripts/fetch/emi_nz.py             # 2019-2024 + plant register
    python scripts/fetch/emi_nz.py --dry-run   # list URLs, download nothing
    python scripts/fetch/emi_nz.py --years 2024

What it fetches (verified July 2026):

  Generation output by plant ("Generation_MD"), monthly CSVs, one row per
  plant per trading day, kWh per half-hourly trading period (TP1..TP50):
    https://www.emi.ea.govt.nz/Wholesale/Datasets/Generation/Generation_MD/
        <YYYYMM>_Generation_MD.csv
  Files run 199708 -> current-month-minus-one and are published around the
  11th-12th of the following month (~0.4-0.9 MB each).

  Dispatched-plant register (effective-dated nameplate/operator rows):
    https://www.emi.ea.govt.nz/Wholesale/Datasets/Generation/GenerationFleet/
        Existing/<YYYYMMDD>_DispatchedGenerationPlant.csv
  The filename is the publication date (~6-monthly); this script scrapes the
  directory listing for the newest one and also saves it under the stable
  name ``DispatchedGenerationPlant.csv``.

Output layout (consumed by scripts/process/emi_nz.py):
    <input-root>/emi_raw/<YYYYMM>_Generation_MD.csv
    <input-root>/emi_raw/DispatchedGenerationPlant.csv
where <input-root> is $PYVWF_INPUT if set, else ./input.

Downloads are resumable: existing files are skipped, partial downloads land
in a .part file and are renamed only on success. EMI notes Generation_MD will
eventually be superseded by a richer dataset — if a fetch 404s across the
board, check the dataset page.
"""
import argparse
import os
import re
import sys
import urllib.request
from pathlib import Path

BASE = "https://www.emi.ea.govt.nz/Wholesale/Datasets/Generation"
GEN_MD = BASE + "/Generation_MD/{yyyymm}_Generation_MD.csv"
REGISTER_DIR = BASE + "/GenerationFleet/Existing"
YEARS = (2019, 2020, 2021, 2022, 2023, 2024)


def output_dir() -> Path:
    root = Path(os.environ.get("PYVWF_INPUT", "input"))
    return root / "emi_raw"


def download(url: str, dest: Path) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "pyvwf-fetch"})
    with urllib.request.urlopen(req) as resp, open(part, "wb") as fh:
        fh.write(resp.read())
    part.rename(dest)


def latest_register_name() -> str:
    """Scrape the register directory listing for the newest filename."""
    req = urllib.request.Request(
        REGISTER_DIR, headers={"User-Agent": "pyvwf-fetch"}
    )
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    names = sorted(set(re.findall(
        r"(\d{8}_DispatchedGenerationPlant\.csv)", html
    )))
    if not names:
        raise RuntimeError(
            f"no *_DispatchedGenerationPlant.csv links found at {REGISTER_DIR}; "
            "the dataset may have moved — check the EMI page."
        )
    return names[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, nargs="+", default=list(YEARS),
                    help="Years of Generation_MD to fetch (default: 2019-2024)")
    ap.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    ap.add_argument("--skip-register", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the download plan and exit")
    args = ap.parse_args()

    out = output_dir()
    plan = [
        (GEN_MD.format(yyyymm=f"{y}{m:02d}"),
         out / f"{y}{m:02d}_Generation_MD.csv")
        for y in sorted(args.years) for m in sorted(args.months)
    ]
    todo = [(u, p) for u, p in plan if not p.is_file()]
    print(f"Output directory: {out}")
    print(f"{len(plan)} monthly file(s) in plan, {len(plan) - len(todo)} "
          f"already present, {len(todo)} to fetch.")

    if args.dry_run:
        for url, path in todo:
            print(f"  would fetch {url}")
        if not args.skip_register:
            print(f"  would fetch newest register from {REGISTER_DIR}")
        return

    out.mkdir(parents=True, exist_ok=True)
    failures = []
    for i, (url, path) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {path.name}", flush=True)
        try:
            download(url, path)
        except Exception as exc:  # noqa: BLE001 — report and continue
            failures.append((url, str(exc)))
            print(f"    FAILED: {exc}", flush=True)

    if not args.skip_register:
        try:
            name = latest_register_name()
            print(f"register: {name}")
            download(f"{REGISTER_DIR}/{name}", out / name)
            (out / name).replace(out / "DispatchedGenerationPlant.csv")
        except Exception as exc:  # noqa: BLE001
            failures.append((REGISTER_DIR, str(exc)))
            print(f"    register FAILED: {exc}", flush=True)

    if failures:
        print(f"\n{len(failures)} download(s) failed; re-run to retry:")
        for url, err in failures:
            print(f"  {url}: {err[:120]}")
        sys.exit(1)
    print("\nAll requested files present.")


if __name__ == "__main__":
    main()
