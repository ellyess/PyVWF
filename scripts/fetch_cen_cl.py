#!/usr/bin/env python
"""Probe and fetch the Coordinador Eléctrico Nacional (Chile) SIP API.

USER-EXECUTED. Reads your API key from the environment so it never appears in
a command line, a settings file, or this repository:

    export CEN_API_KEY=<your Información Pública (SIP) user_key>
    python scripts/fetch_cen_cl.py --probe          # verification, writes nothing
    python scripts/fetch_cen_cl.py --years 2015 2024

Verified against the live API on 2026-07-22 (docs/RUNBOOK_CL.md records the
full endpoint map). The facts that cost time to discover, so they are pinned
here rather than rediscovered:

- **Host is `sipub.api.coordinador.cl`.** The SIP OpenAPI document declares no
  `servers` block, so the host is not discoverable from the spec;
  `sip.api.coordinador.cl` does not resolve.
- **The service is SIP, not Operation.** Operation's `/reportes/v3/generation`
  returns aggregate GWh only.
- **`page`/`limit` on `/centrales/v4/findByDate` return HTTP 502.** Send
  `startDate`/`endDate` alone. That endpoint is near-useless anyway: it caps
  at 10 records whatever the window.
- **Responses wrap results in `data`** (the Infotécnica resources use
  `results` with `count`/`next`), not the `content` the schema suggests.
- **`tipoTecnologia` must be `Eólica`, accented.** `Eolica` silently returns
  an empty list rather than erroring — a trap that reads like "no data".
- **Plant metadata rides along with the generation rows** (`id_central`,
  `central`, `propietario`, `potencia_maxima`, `tipo_tecnologia`), so the
  fleet is derived from the generation stream. Neither `/centrales/v4` nor
  the Infotécnica registry exposes usable coordinates, so **lon/lat must be
  joined from the Global Wind Power Tracker** (CC-BY-4.0, already on disk).

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
from collections import Counter
from pathlib import Path

#: Resolved by DNS, not from the spec (see module docstring).
BASE = os.environ.get("CEN_API_BASE", "https://sipub.api.coordinador.cl")

GEN_REAL = "/generacion-real/v3/findByDate"
CENTRALES = "/centrales/v4/findByDate"
INFOTECNICA = "/api/v2/recursos/infotecnica/centrales/"

#: MUST carry the accent: the unaccented spelling returns an empty list.
WIND = "Eólica"


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


#: Seconds to pause between calls. The gateway returns HTTP 429 under a
#: sustained day-by-day pull (measured: ~59 consecutive calls was enough to
#: trip it), so the bulk fetch paces itself rather than sprinting and failing.
THROTTLE_S = float(os.environ.get("CEN_THROTTLE_S", "1.5"))

#: Backoff schedule for 429s, in seconds. Generous on purpose: a bulk fetch is
#: unattended, and losing a month to an impatient retry costs more than waiting.
BACKOFF_S = (30, 60, 120, 300, 600)


def get(path: str, params: dict, *, timeout: int = 300) -> dict:
    """One GET against the SIP API, with rate-limit backoff.

    The key is added here, once. On HTTP 429 the call sleeps and retries on
    the BACKOFF_S schedule; other errors are fatal, since they mean the
    request shape is wrong rather than the pace.
    """
    query = dict(params)
    query["user_key"] = api_key()
    url = f"{BASE}{path}?{urllib.parse.urlencode(query)}"

    for attempt, wait in enumerate((*BACKOFF_S, None)):
        req = urllib.request.Request(url, headers={"User-Agent": "pyvwf-fetch"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            body = exc.read()[:300].decode("utf-8", errors="replace")
            if exc.code == 429 and wait is not None:
                print(f"    rate limited; sleeping {wait}s "
                      f"(attempt {attempt + 1}/{len(BACKOFF_S)})", flush=True)
                time.sleep(wait)
                continue
            # Never echo the URL: it carries the key.
            sys.exit(
                f"HTTP {exc.code} on {path}: {body}\n"
                "(502 here usually means page/limit were sent — omit them; "
                "429 means the backoff schedule was exhausted, so re-run: "
                "completed months are skipped.)"
            )
        except urllib.error.URLError as exc:
            sys.exit(f"network error on {path}: {exc.reason}")
    raise AssertionError("unreachable")


def rows(payload) -> list:
    """SIP wraps results in 'data'; Infotécnica resources use 'results'."""
    if isinstance(payload, list):
        return payload
    for key in ("data", "results", "content", "items"):
        if isinstance(payload.get(key), list):
            return payload[key]
    return []


def wind_rows(records: list) -> list:
    """Wind rows only.

    Matched on the exact technology label, case-insensitively. A substring
    test on 'lica' is WRONG: it also matches 'Hidráulica'.
    """
    return [r for r in records
            if str(r.get("tipo_tecnologia", "")).strip().lower() == WIND.lower()]


def fetch_day(date: str, *, page_size: int = 20000) -> list:
    """Every wind generation row for one date.

    A `page` parameter makes this endpoint return HTTP 502 (as `page`/`limit`
    do on /centrales), so paging is not available: the whole day is requested
    in one call with a `pageSize` far above the fleet's daily row count
    (~65 plants x 24 h). If the response ever reports more than one page, that
    silently truncates, so it raises instead.
    """
    payload = get(GEN_REAL, {
        "startDate": date, "endDate": date,
        "tipoTecnologia": WIND, "pageSize": page_size,
    })
    total_pages = payload.get("totalPages") or 1
    if total_pages > 1:
        raise RuntimeError(
            f"{date}: response reports {total_pages} pages but this endpoint "
            "502s on `page` — raise page_size rather than truncating."
        )
    return rows(payload)


def probe() -> None:
    print("=" * 70)
    print(f"CEN SIP probe  (host {BASE})")
    print("=" * 70)

    day = fetch_day("2024-06-01")
    if not day:
        sys.exit("no wind rows for 2024-06-01 — is the key for the SIP plan?")
    plants = {r["id_central"] for r in day}
    print(f"2024-06-01: {len(day)} wind rows, {len(plants)} plants, "
          f"{len({r.get('hora') for r in day})} hours")
    print(f"generation fields: {sorted(day[0].keys())}")
    print("\nsample row:")
    print(" ", json.dumps(day[0], ensure_ascii=False)[:300])

    print("\n" + "-" * 70)
    print("history depth (one day per year)")
    print("-" * 70)
    for year in (2010, 2013, 2015, 2017, 2019, 2021, 2023, 2025):
        d = fetch_day(f"{year}-06-01")
        n = len({r["id_central"] for r in d}) if d else 0
        print(f"  {year}: {len(d):5d} rows  {n:3d} plants")
        time.sleep(0.4)

    print("\n" + "-" * 70)
    print("capacity and technology labels")
    print("-" * 70)
    caps = {r["central"]: r.get("potencia_maxima") for r in day}
    print(f"plants carrying potencia_maxima: "
          f"{sum(1 for v in caps.values() if v not in (None, '', '-'))}/{len(caps)}")
    print("subtipo_tecnologia:",
          Counter(r.get("subtipo_tecnologia") for r in day).most_common(5))
    print("\nlargest plants by potencia_maxima:")
    ranked = sorted(caps.items(),
                    key=lambda kv: float(kv[1] or 0), reverse=True)[:8]
    for name, cap in ranked:
        print(f"  {name[:44]:44s} {cap} MW")

    print("\n" + "=" * 70)
    print("Coordinates are NOT in the API: join the Global Wind Power Tracker.")
    print("fecha_hora is Chilean civil time (America/Santiago observes DST) and")
    print("hora is 1-24 where hora N starts at (N-1):00 — confirm before the")
    print("adapter bins to UTC. See docs/RUNBOOK_CL.md.")
    print("=" * 70)


def fetch_generation(out_dir: Path, y0: int, y1: int) -> None:
    """One JSON file per month of wind generation, resumable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    import calendar
    for year in range(y0, y1 + 1):
        for month in range(1, 13):
            dest = out_dir / f"cen_gen_{year}_{month:02d}.json"
            if dest.is_file():
                continue
            last = calendar.monthrange(year, month)[1]
            collected = []
            for day in range(1, last + 1):
                collected.extend(fetch_day(f"{year}-{month:02d}-{day:02d}"))
                time.sleep(THROTTLE_S)
            if collected:
                dest.write_text(json.dumps(collected, ensure_ascii=False))
                n = len({r["id_central"] for r in collected})
                print(f"  {year}-{month:02d}: {len(collected)} rows, {n} plants",
                      flush=True)
            else:
                print(f"  {year}-{month:02d}: empty (before fleet history?)",
                      flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="store_true",
                    help="Verification run; writes nothing")
    ap.add_argument("--years", type=int, nargs=2, metavar=("START", "END"),
                    help="Fetch per-plant wind generation for this window")
    args = ap.parse_args()

    out_dir = Path(os.environ.get("PYVWF_INPUT", "input")) / "cen_raw"
    if args.probe:
        probe()
    if args.years:
        fetch_generation(out_dir, *args.years)
    if not (args.probe or args.years):
        ap.print_help()


if __name__ == "__main__":
    main()
