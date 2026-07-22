#!/usr/bin/env python
"""Probe and fetch the EPİAŞ/EXIST Transparency Platform (Turkey).

USER-EXECUTED. Turkey is the only surveyed candidate with BOTH per-plant
hourly generation and a national turbine register (TÜREB) carrying turbine
models, so hub heights become derivable per plant — see
docs/RUNBOOK_TR.md and docs/findings/dataset_survey_2026-07.md.

WHERE TO PUT YOUR CREDENTIALS
-----------------------------
Two options; pick either. Both keep the password out of the shell history,
out of this repository, and out of any log.

1. A credentials file (easiest). Create `input/.epias_credentials`:

       {"username": "your@email", "password": "your-password"}

   `input/` is git-ignored in its entirety (.gitignore line 90), so the file
   cannot be committed by accident. The script reads it with no other setup.
   Optionally `chmod 600 input/.epias_credentials`.

2. Environment variables, if you prefer nothing on disk:

       export EPIAS_USERNAME='your@email'
       export EPIAS_PASSWORD='your-password'

Environment variables win when both are present.

    python scripts/fetch_epias_tr.py --probe      # verification, writes nothing
    python scripts/fetch_epias_tr.py --years 2021 2024

WHAT THE PROBE ANSWERS
----------------------
The one open question that decides whether Turkey is a full tier-1 region:
**per-plant history depth**. The platform launched in 2016 but per-plant
series start dates were never verified. `--probe` authenticates, pulls the
power-plant list, finds the wind plants, and walks one plant back through the
years, printing where the data begins.

Endpoint paths on Transparency 2.0 are POST-with-JSON-body and have moved
between versions, so the probe tries the documented candidates and reports
which one answers rather than assuming. Once a path is confirmed, pin it here.

The mature community wrapper `eptr2` (pip install eptr2) covers 110+ services
if you would rather not maintain raw paths; this script stays dependency-free
so it runs in the project environment as-is.

Turkey abolished daylight saving in 2016 and sits at a permanent UTC+3, so the
timestamp handling is a fixed offset — simpler than the NZ trading periods or
Chile's DST. Verify the API's own timestamp labels before trusting that.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CAS = "https://giris.epias.com.tr/cas/v1/tickets"
BASE = "https://seffaflik.epias.com.tr/electricity-service/v1"

#: Candidate paths tried by the probe, in order. Transparency 2.0 renamed
#: several of these between releases; the probe reports which answered.
PLANT_LIST_PATHS = (
    "/generation/data/powerplant-list",
    "/generation/data/injection-quantity-powerplant-list",
    "/generation/data/uevm-powerplant-list",
)
GENERATION_PATHS = (
    "/generation/data/injection-quantity",        # UEVM, settlement-grade
    "/generation/data/realtime-generation",       # operational
)

CRED_FILE = "input/.epias_credentials"


def credentials() -> tuple[str, str]:
    """Username and password, from the environment or the credentials file.

    Never printed, never placed on a command line, never written anywhere.
    """
    user = os.environ.get("EPIAS_USERNAME", "").strip()
    pwd = os.environ.get("EPIAS_PASSWORD", "")
    if user and pwd:
        return user, pwd

    path = Path(os.environ.get("PYVWF_INPUT", "input")) / ".epias_credentials"
    if not path.is_file():
        path = Path(CRED_FILE)
    if path.is_file():
        try:
            blob = json.loads(path.read_text())
            user = str(blob.get("username", "")).strip()
            pwd = str(blob.get("password", ""))
        except json.JSONDecodeError:
            # tolerate key=value lines
            blob = {}
            for line in path.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    blob[k.strip().lower()] = v.strip().strip("'\"")
            user, pwd = blob.get("username", ""), blob.get("password", "")
        if user and pwd:
            return user, pwd

    sys.exit(
        "No EPİAŞ credentials found.\n\n"
        f"  Either create {CRED_FILE} containing:\n"
        '      {"username": "your@email", "password": "your-password"}\n'
        "  (input/ is git-ignored, so it cannot be committed)\n\n"
        "  Or export them:\n"
        "      export EPIAS_USERNAME='your@email'\n"
        "      export EPIAS_PASSWORD='your-password'\n"
    )


def get_tgt() -> str:
    """Authenticate and return the ticket-granting ticket.

    The ticket is a bearer credential: it is returned to the caller and used
    in headers, but never logged or printed.
    """
    user, pwd = credentials()
    body = urllib.parse.urlencode({"username": user, "password": pwd}).encode()
    req = urllib.request.Request(
        CAS, data=body, method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Accept": "text/plain", "User-Agent": "pyvwf-fetch"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            tgt = resp.read().decode("utf-8", errors="replace").strip()
            location = resp.headers.get("Location", "")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            sys.exit(
                f"EPİAŞ authentication rejected (HTTP {exc.code}).\n"
                "Check the username/password, and that the account is "
                "activated and subscribed on seffaflik.epias.com.tr."
            )
        sys.exit(f"EPİAŞ auth failed: HTTP {exc.code}")
    except urllib.error.URLError as exc:
        sys.exit(f"network error contacting EPİAŞ: {exc.reason}")

    if not tgt and location:
        tgt = location.rstrip("/").rsplit("/", 1)[-1]
    if not tgt:
        sys.exit("EPİAŞ returned no ticket; the auth endpoint may have moved.")
    return tgt


def post(path: str, payload: dict, tgt: str, *, timeout: int = 180):
    """POST a JSON body to a Transparency endpoint. Returns (status, parsed)."""
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "TGT": tgt, "User-Agent": "pyvwf-fetch"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()[:200].decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return "ERR", str(exc.reason)[:120]


def items(payload) -> list:
    """Transparency wraps results in 'items' (sometimes 'body' > 'items')."""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("items"), list):
        return payload["items"]
    body = payload.get("body")
    if isinstance(body, dict) and isinstance(body.get("items"), list):
        return body["items"]
    for value in payload.values():
        if isinstance(value, list):
            return value
    return []


def day_window(date: str) -> dict:
    """Transparency wants ISO timestamps with Turkey's fixed +03:00 offset."""
    return {"startDate": f"{date}T00:00:00+03:00",
            "endDate": f"{date}T23:00:00+03:00"}


def probe() -> None:
    print("=" * 70)
    print("EPİAŞ Transparency probe")
    print("=" * 70)
    tgt = get_tgt()
    print("authentication: OK (ticket acquired)\n")

    plants, used_path = [], None
    for path in PLANT_LIST_PATHS:
        status, payload = post(path, day_window("2024-06-01"), tgt)
        found = items(payload) if status == 200 else []
        print(f"  {path:52s} {status} {len(found) if found else ''}")
        if found:
            plants, used_path = found, path
            break
    if not plants:
        sys.exit(
            "\nNo plant-list endpoint answered. The paths have moved; check\n"
            "https://seffaflik.epias.com.tr/electricity-service/technical/tr/index.html\n"
            "or use the eptr2 wrapper, then pin the working path in this script."
        )

    print(f"\nplant list via {used_path}: {len(plants)} plants")
    print(f"fields: {sorted(plants[0].keys())}")

    def is_wind(rec: dict) -> bool:
        blob = json.dumps(rec, ensure_ascii=False).lower()
        return "rüzgar" in blob or "ruzgar" in blob or "wind" in blob

    wind = [p for p in plants if is_wind(p)]
    print(f"wind plants detected: {len(wind)}")
    for p in wind[:5]:
        print("   ", json.dumps(p, ensure_ascii=False)[:150])
    sample = (wind or plants)[0]
    pid = sample.get("id") or sample.get("powerPlantId") or sample.get("plantId")
    print(f"\nprobing plant id={pid} ({sample.get('name', '?')})")

    gen_path = None
    for path in GENERATION_PATHS:
        payload = dict(day_window("2024-06-01"))
        payload["powerPlantId"] = pid
        status, resp = post(path, payload, tgt)
        found = items(resp) if status == 200 else []
        print(f"  {path:52s} {status} {len(found) if found else ''}")
        if found:
            gen_path = path
            print(f"\n  generation fields: {sorted(found[0].keys())}")
            print("  sample:", json.dumps(found[0], ensure_ascii=False)[:220])
            break
    if not gen_path:
        sys.exit("\nNo generation endpoint answered for that plant id.")

    print("\n" + "-" * 70)
    print("history depth (one day per year) — THE decisive question")
    print("-" * 70)
    for year in (2016, 2017, 2018, 2019, 2020, 2022, 2024):
        payload = dict(day_window(f"{year}-06-01"))
        payload["powerPlantId"] = pid
        status, resp = post(gen_path, payload, tgt)
        found = items(resp) if status == 200 else []
        print(f"  {year}: {len(found):3d} rows  {'' if status == 200 else status}")

    print("\n" + "=" * 70)
    print("The earliest year with rows is the per-plant history depth.")
    print("Record it in docs/findings/dataset_survey_2026-07.md, then pair the")
    print("plant list with the TÜREB register for turbine models -> hub heights.")
    print("=" * 70)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="store_true",
                    help="Authenticate and verify; writes nothing")
    ap.add_argument("--years", type=int, nargs=2, metavar=("START", "END"),
                    help="Fetch per-plant generation for this window")
    args = ap.parse_args()

    if args.probe:
        probe()
    elif args.years:
        sys.exit("Run --probe first: the endpoint paths and history depth must "
                 "be confirmed before a bulk fetch is worth starting.")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
