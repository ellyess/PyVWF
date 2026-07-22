#!/usr/bin/env python
"""Probe and fetch the Coordinador Eléctrico Nacional (Chile) SIP API.

USER-EXECUTED. Reads your API key from the environment so it never appears in
a command line, a settings file, or this repository:

    export CEN_API_KEY=<your SIP user_key>
    python scripts/fetch_cen_cl.py --probe          # decisive first check
    python scripts/fetch_cen_cl.py --plants         # write the wind plant list
    python scripts/fetch_cen_cl.py --years 2019 2024

Why a probe first (docs/RUNBOOK_CL.md): the July-2026 survey verified that
per-plant hourly generation exists, but not its history depth, field names, or
timezone convention. `--probe` answers all three in one run and prints what it
finds, so the adapter is written against observed data rather than assumption.

The SIP service (NOT the Operation service) carries the per-plant data:
    /centrales/v4/findByDate            plant registry (id_central, capacity,
                                        fecha_ent_oper, region/provincia/comuna)
    /generacion-real/v3/findByDate      per-plant generation, filterable by
                                        idCentral and tipoTecnologia, paginated
    /capacidad-instalada/v4/findByDate  installed-capacity history

Note the registry carries NO latitude/longitude — only region/province/commune.
Coordinates must come from the Global Wind Power Tracker (CC-BY-4.0, already in
input/); that join is the next step after this probe passes.

Raw responses land in <input-root>/cen_raw/ (input/ is git-ignored).
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://sip.api.coordinador.cl"
CENTRALES = "/centrales/v4/findByDate"
GEN_REAL = "/generacion-real/v3/findByDate"
CAPACIDAD = "/capacidad-instalada/v4/findByDate"

#: Values of tipo_tecnologia that denote wind in the CEN registry. Matched
#: case- and accent-insensitively because the field is free text.
WIND_TERMS = ("eolica", "eólica", "wind")


def api_key() -> str:
    key = os.environ.get("CEN_API_KEY", "").strip()
    if not key:
        sys.exit(
            "CEN_API_KEY is not set.\n"
            "  export CEN_API_KEY=<your SIP user_key from portal.api.coordinador.cl>\n"
            "The key is read from the environment on purpose: it must never be "
            "committed, logged, or passed on a command line."
        )
    return key


def get(path: str, params: dict, *, timeout: int = 120) -> dict:
    """One GET against the SIP API. The key is added here, once."""
    query = dict(params)
    query["user_key"] = api_key()
    url = f"{BASE}{path}?{urllib.parse.urlencode(query)}"
    req = urllib.request.Request(url, headers={"User-Agent": "pyvwf-fetch"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read()[:300].decode("utf-8", errors="replace")
        # Never echo the URL: it carries the key.
        sys.exit(f"HTTP {exc.code} on {path}: {body}")
    except urllib.error.URLError as exc:
        sys.exit(f"network error on {path}: {exc.reason}")


def rows(payload) -> list:
    """SIP wraps results in 'content' (paged) or 'data'; tolerate both."""
    if isinstance(payload, list):
        return payload
    for key in ("content", "data", "items"):
        if isinstance(payload.get(key), list):
            return payload[key]
    return []


def is_wind(record: dict) -> bool:
    blob = " ".join(
        str(record.get(k, "")) for k in ("tipo_tecnologia", "tipo_central",
                                         "tipo_conv_energia")
    ).lower()
    return any(term in blob for term in WIND_TERMS)


def probe() -> None:
    print("=" * 70)
    print("CEN SIP probe — plant registry")
    print("=" * 70)
    reg = rows(get(CENTRALES, {"page": 0, "limit": 2000}))
    print(f"registry records: {len(reg)}")
    if not reg:
        sys.exit("registry returned nothing — check the key is for the SIP plan.")
    print(f"registry fields: {sorted(reg[0].keys())}")

    wind = [r for r in reg if is_wind(r)]
    print(f"\nwind plants: {len(wind)}")
    for r in wind[:5]:
        print(f"  id_central={r.get('id_central')} "
              f"{str(r.get('central'))[:38]:38s} "
              f"cap={r.get('capac_max') or r.get('pot_max_bruta')} "
              f"oper={r.get('fecha_ent_oper')} "
              f"{r.get('region')}")
    has_coords = any(k.lower() in ("lat", "latitud", "latitude")
                     for k in reg[0])
    print(f"\ncoordinates present in registry: {has_coords}"
          f"{'' if has_coords else '  -> must join GWPT for lon/lat'}")

    if not wind:
        sys.exit("no wind plants matched — inspect tipo_tecnologia values above.")

    print("\n" + "=" * 70)
    print("CEN SIP probe — per-plant generation history depth")
    print("=" * 70)
    target = wind[0]
    tid = target.get("id_central")
    print(f"probing id_central={tid} ({target.get('central')})\n")
    for year in (2015, 2018, 2020, 2022, 2024, 2025):
        payload = get(GEN_REAL, {
            "startDate": f"{year}-06-01", "endDate": f"{year}-06-02",
            "idCentral": tid, "pageSize": 5,
        })
        got = rows(payload)
        total = payload.get("totalElements") if isinstance(payload, dict) else None
        print(f"  {year}: {len(got):3d} rows returned"
              f"{f', totalElements={total}' if total is not None else ''}")
        if got and year == 2024:
            print(f"\n  generation fields: {sorted(got[0].keys())}")
            print("  sample row:")
            print("   ", json.dumps(got[0], ensure_ascii=False)[:400])
        time.sleep(0.5)  # be polite to the gateway

    print("\n" + "=" * 70)
    print("Read the earliest year with rows > 0 as the per-plant history depth.")
    print("Check the timestamp field above for timezone convention (America/")
    print("Santiago has DST) before writing the adapter — see docs/RUNBOOK_CL.md.")
    print("=" * 70)


def write_plants(out_dir: Path) -> None:
    reg = rows(get(CENTRALES, {"page": 0, "limit": 2000}))
    wind = [r for r in reg if is_wind(r)]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cen_centrales_eolicas.json"
    path.write_text(json.dumps(wind, ensure_ascii=False, indent=1))
    print(f"{len(wind)} wind plants -> {path}")


def fetch_generation(out_dir: Path, y0: int, y1: int) -> None:
    reg = rows(get(CENTRALES, {"page": 0, "limit": 2000}))
    wind = [r for r in reg if is_wind(r)]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(wind)} wind plants; fetching {y0}-{y1} month by month")
    for year in range(y0, y1 + 1):
        for month in range(1, 13):
            dest = out_dir / f"cen_gen_{year}_{month:02d}.json"
            if dest.is_file():
                continue
            last = 28 if month == 2 else (30 if month in (4, 6, 9, 11) else 31)
            collected, page = [], 0
            while True:
                payload = get(GEN_REAL, {
                    "startDate": f"{year}-{month:02d}-01",
                    "endDate": f"{year}-{month:02d}-{last}",
                    "tipoTecnologia": "Eolica",
                    "page": page, "pageSize": 5000,
                })
                batch = rows(payload)
                collected.extend(batch)
                total_pages = payload.get("totalPages", 1) if isinstance(payload, dict) else 1
                page += 1
                if page >= (total_pages or 1) or not batch:
                    break
            if collected:
                dest.write_text(json.dumps(collected, ensure_ascii=False))
                print(f"  {year}-{month:02d}: {len(collected)} rows")
            else:
                print(f"  {year}-{month:02d}: empty (before fleet history?)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="store_true",
                    help="Registry + history-depth check; writes nothing")
    ap.add_argument("--plants", action="store_true",
                    help="Write the wind plant registry to cen_raw/")
    ap.add_argument("--years", type=int, nargs=2, metavar=("START", "END"),
                    help="Fetch per-plant generation for this inclusive window")
    args = ap.parse_args()

    out_dir = Path(os.environ.get("PYVWF_INPUT", "input")) / "cen_raw"
    if args.probe:
        probe()
    if args.plants:
        write_plants(out_dir)
    if args.years:
        fetch_generation(out_dir, *args.years)
    if not (args.probe or args.plants or args.years):
        ap.print_help()


if __name__ == "__main__":
    main()
