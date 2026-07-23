#!/usr/bin/env python
"""Fetch the Denmark wind-turbine data from the Danish Energy Agency (ens.dk).

USER-EXECUTED (though credential-free): downloads the two public Excel files
that back the Denmark region, so the manual ens.dk download becomes
reproducible like every other source.

    python scripts/fetch/dk.py            # both files -> input/observations/turbine/DK/
    python scripts/fetch/dk.py --dry-run  # show the plan, download nothing

What it fetches (the canonical "Stamdataregister for vindkraftanlæg" /
Master data register of wind turbines, from ens.dk's energy-sector data page
https://ens.dk/en/analyses-and-statistics/overview-energy-sector):

    anlaeg.xlsx                 master data register: one row per turbine
        (GSRN id, connection date, capacity kW, rotor diameter, hub height,
         X/Y coordinates, manufacturer/model), split across an
         "existing"/"decommissioned" sheet pair. Header on row 9.
    maanedsdata_2002_2020.xlsx  monthly production to grid: one sheet per
        year (``Månedsprod_<year>``, 2002-2020), kWh per turbine per month.
        Header on row 7. (ens.dk serves this under the stale filename
        ``maanedsdata_2002_2017.xlsx``, but the workbook carries all 19 years
        through 2020; it is saved here under the name the processor expects.)

Then reduce them to the adapter's CSVs:

    python scripts/process/dk.py          # -> dk_md.csv, dk_obs_2002_2020.csv

VINTAGE. These are stable ens.dk media URLs pointing at the register snapshot
taken ultimo January 2022: the exact files the Denmark validation was built
on, so this reproduces committed results bit-for-bit. The *monthly production*
history (2002-2020) is a fixed historical dataset. The *master data register*
is refreshed monthly under a NEW media id each time; to pull a newer master
snapshot, open the ens.dk page above, copy the current "Data on operating and
decommissioned wind turbines" link, and pass it with ``--anlaeg-url``. A newer
register only adds post-2022 turbines and does not change the 2002-2020
production, so the pinned default is the reproducible choice.

Downloads are resumable: existing files are skipped, partials land in a .part
file and are renamed only on success.
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

# Stable ens.dk media ids for the ultimo-2022 register snapshot (verified: the
# served content-disposition filenames and byte sizes match the committed
# Denmark inputs exactly).
ANLAEG_URL = "https://ens.dk/media/4945/download"        # -> anlaeg.xlsx
MAANEDSDATA_URL = "https://ens.dk/media/4948/download"   # -> maanedsdata (2002-2020)


def output_dir() -> Path:
    root = Path(os.environ.get("PYVWF_INPUT", "input"))
    return root / "observations/turbine" / "DK"


def download(url: str, dest: Path) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "pyvwf-fetch"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(part, "wb") as fh:
        fh.write(resp.read())
    part.rename(dest)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anlaeg-url", default=ANLAEG_URL,
                    help="Override the master-data-register URL (for a newer snapshot)")
    ap.add_argument("--maanedsdata-url", default=MAANEDSDATA_URL,
                    help="Override the monthly-production URL")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and exit")
    args = ap.parse_args()

    out = output_dir()
    plan = [
        (args.anlaeg_url, out / "anlaeg.xlsx"),
        (args.maanedsdata_url, out / "maanedsdata_2002_2020.xlsx"),
    ]
    todo = [(u, p) for u, p in plan if not p.is_file()]
    print(f"Output directory: {out}")
    print(f"{len(plan)} file(s), {len(plan) - len(todo)} already present, "
          f"{len(todo)} to fetch.")
    if args.dry_run:
        for u, p in todo:
            print(f"  would fetch {u} -> {p.name}")
        return
    if not todo:
        return

    out.mkdir(parents=True, exist_ok=True)
    failures = []
    for url, path in todo:
        print(f"  {path.name} <- {url}", flush=True)
        try:
            download(url, path)
            print(f"    {path.stat().st_size / 1e6:.1f} MB", flush=True)
        except Exception as exc:  # noqa: BLE001 (report and continue)
            failures.append((url, str(exc)))
            print(f"    FAILED: {exc}", flush=True)

    if failures:
        print(f"\n{len(failures)} download(s) failed; re-run to retry:")
        for url, err in failures:
            print(f"  {url}: {err[:120]}")
        sys.exit(1)
    print("\nBoth files present. Next: python scripts/process/dk.py")


if __name__ == "__main__":
    main()
