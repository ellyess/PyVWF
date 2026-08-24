#!/usr/bin/env python
"""Fetch the open UK wind-farm data (REPD) and guide the Ofgem ROC export.

USER-EXECUTED. The UK region has two observed-source components, with very
different reproducibility (see docs/runbooks/uk.md for the full story):

  METADATA: auto-downloadable. The Renewable Energy Planning Database (REPD,
  Open Government Licence) from gov.uk gives per-site location, capacity, and
  turbine count for every operational UK wind farm. This script downloads it.

  OBSERVATIONS: NOT auto-downloadable. Per-station monthly ROC issuance lives
  only in Ofgem's Renewable Electricity Register (rer.ofgem.gov.uk), behind a
  gated, form-driven export (the login-free historical bulk file is
  technology-aggregate only, no per-station data). This script CANNOT fetch it;
  it prints the manual steps, and scripts/process/uk.py processes whatever
  export you save.

    python scripts/fetch/uk.py            # download REPD, print the ROC steps
    python scripts/fetch/uk.py --dry-run  # show the plan only

REPD's direct file URL rotates every quarter (a new
assets.publishing.service.gov.uk/media/<hash>/REPD_publication_Qn_YYYY.csv),
so this scrapes the stable publication page for the current attachment rather
than hard-coding a URL that would go stale.
"""
import argparse
import os
import re
import sys
import urllib.request
from pathlib import Path

REPD_PUBLICATION = (
    "https://www.gov.uk/government/publications/"
    "renewable-energy-planning-database-quarterly-extract"
)
ROC_STEPS = """\
Ofgem ROC issuance (per-station monthly), MANUAL, then process:
  1. Go to the Renewable Electricity Register public reports:
     https://rer.ofgem.gov.uk/  (View Public Reports, no login), or email
     renewable.enquiry@ofgem.gov.uk for the current public-report access link.
  2. Export the "Certificates / ROCs issued" report for the Renewables
     Obligation scheme, filtered to wind, for your output-period window.
  3. Save it as input/ofgem_raw/roc_issuance.xlsx (or .csv).
  4. Run:  python scripts/process/uk.py observations \\
             --roc input/ofgem_raw/roc_issuance.xlsx
     (RER keeps only 7 years of issuance; older years may need the archived
      per-station reports.)
"""


def output_dir() -> Path:
    return Path(os.environ.get("PYVWF_INPUT", "input")) / "raw/repd"


def find_repd_csv() -> str:
    """Scrape the REPD publication page for the current CSV attachment URL."""
    req = urllib.request.Request(REPD_PUBLICATION,
                                 headers={"User-Agent": "pyvwf-fetch"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    urls = re.findall(
        r"https://assets\.publishing\.service\.gov\.uk/media/[a-z0-9]+/"
        r"REPD_publication_[^\"'\s]+\.csv", html)
    if not urls:
        sys.exit("could not find a REPD CSV link on the publication page; the "
                 f"page layout may have changed; check {REPD_PUBLICATION}")
    return sorted(set(urls))[-1]


def download(url: str, dest: Path) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "pyvwf-fetch"})
    with urllib.request.urlopen(req, timeout=180) as resp, open(part, "wb") as fh:
        fh.write(resp.read())
    part.rename(dest)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = output_dir()
    dest = out / "repd_wind.csv"
    print(f"REPD publication page: {REPD_PUBLICATION}")
    url = find_repd_csv()
    print(f"current REPD CSV: {url}")

    if args.dry_run:
        print(f"  would download -> {dest}")
    elif dest.is_file():
        print(f"  {dest} already present, skipping")
    else:
        out.mkdir(parents=True, exist_ok=True)
        download(url, dest)
        print(f"  {dest.stat().st_size / 1e6:.1f} MB -> {dest}")
        print("  next: python scripts/process/uk.py metadata")

    print("\n" + "=" * 70)
    print(ROC_STEPS)


if __name__ == "__main__":
    main()
