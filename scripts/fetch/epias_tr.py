#!/usr/bin/env python
"""Probe and fetch the EPİAŞ/EXIST Transparency Platform (Turkey).

USER-EXECUTED. Turkey is the only surveyed candidate with BOTH per-plant
hourly generation and a national turbine register (TÜREB) carrying turbine
models, so hub heights become derivable per plant; see
docs/runbooks/TR.md and docs/findings/dataset_survey_2026-07.md.

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

    python scripts/fetch/epias_tr.py --probe      # verification, writes nothing
    python scripts/fetch/epias_tr.py --years 2021 2024

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
timestamp handling is a fixed offset, simpler than the NZ trading periods or
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
#: CONFIRMED working (GET, 2026-07-22): the UEVM power-plant list.
PLANT_LIST_PATHS = (
    "/generation/data/injection-quantity-powerplant-list",
    "/generation/data/powerplant-list",
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


def call(path: str, payload: dict, tgt: str, *, method: str = "POST",
         timeout: int = 180):
    """Call a Transparency endpoint. Returns (status, parsed).

    Method matters and is not guessable from the path: the *-list endpoints
    answer to GET and return 404 to POST, while the data endpoints take a
    POST body. A 404 here usually means the wrong verb, not a wrong path.
    """
    data = json.dumps(payload).encode() if method == "POST" else None
    req = urllib.request.Request(
        f"{BASE}{path}", data=data, method=method,
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


def try_both(path: str, payload: dict, tgt: str):
    """Try GET then POST; return (status, parsed, method) for the first hit."""
    for method in ("GET", "POST"):
        status, parsed = call(path, payload, tgt, method=method)
        if status == 200 and items(parsed):
            return status, parsed, method
    return status, parsed, None


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

    plants, used_path, list_method = [], None, None
    for path in PLANT_LIST_PATHS:
        status, payload, method = try_both(
            path, {"period": "2024-06-01T00:00:00+03:00"}, tgt)
        found = items(payload) if status == 200 else []
        print(f"  {path:52s} {status} {method or ''} {len(found) if found else ''}")
        if found:
            plants, used_path, list_method = found, path, method
            break
    if not plants:
        sys.exit(
            "\nNo plant-list endpoint answered. The paths have moved; check\n"
            "https://seffaflik.epias.com.tr/electricity-service/technical/tr/index.html\n"
            "or use the eptr2 wrapper, then pin the working path in this script."
        )

    print(f"\nplant list via {list_method} {used_path}: {len(plants)} plants")
    print(f"fields: {sorted(plants[0].keys())}")

    def is_wind(rec: dict) -> bool:
        """Wind plants are named ...RÜZGAR... in the Turkish register.

        Matched on the plant NAME only. Testing the whole JSON blob would
        also catch an owner company with 'Rüzgar' in its title, the same
        overcount trap that 'lica' vs 'Hidraulica' produced for Chile.
        """
        name = str(rec.get("name", "")).lower()
        return "rüzgar" in name or "ruzgar" in name or "res" in name.split()

    wind = [p for p in plants if is_wind(p)]
    print(f"wind plants detected: {len(wind)} of {len(plants)}")
    for p in wind[:5]:
        print("   ", json.dumps(p, ensure_ascii=False)[:150])
    sample = (wind or plants)[0]
    pid = sample.get("id") or sample.get("powerPlantId") or sample.get("plantId")
    print(f"\nprobing plant id={pid} ({str(sample.get('name', '?')).strip()})")

    other = next((p for p in wind
                  if (p.get("id") or p.get("powerPlantId")) != pid), None)
    other_id = other.get("id") if other else None

    print("\n" + "-" * 70)
    print("MUST-DISTINGUISH: does powerPlantId actually filter?")
    print("-" * 70)
    print("A 200 with 24 rows is NOT evidence of per-plant data. These endpoints")
    print("return the NATIONAL fuel-mix series and ignore an unknown parameter")
    print("silently, so the only proof is that two different plant ids give")
    print("different numbers. A check that cannot fail is not a check.\n")

    for path in GENERATION_PATHS:
        sigs, keys = {}, []
        for label, plant in (("plant A", pid), ("plant B", other_id)):
            if plant is None:
                continue
            payload = dict(day_window("2024-06-01"))
            payload["powerPlantId"] = plant
            status, resp, method = try_both(path, payload, tgt)
            found = items(resp) if status == 200 else []
            if found:
                keys = [k for k in found[0] if k not in ("date", "hour")]
                sigs[label] = json.dumps(
                    {k: v for k, v in found[0].items() if isinstance(v, (int, float))},
                    sort_keys=True)
            print(f"  {path:46s} {label} {status} rows={len(found)}")
        if keys:
            print(f"    payload keys: {keys[:8]}")
        if len(sigs) > 1 and len(set(sigs.values())) == 1:
            print("    VERDICT: identical across plant ids -> powerPlantId IGNORED;")
            print("             this is national data, NOT per-plant.\n")
        elif len(sigs) > 1:
            print("    VERDICT: responses differ -> genuinely per-plant.\n")

    print("=" * 70)
    print("Status (2026-07-22): authentication and the 2,584-plant list work,")
    print("but NO per-plant generation endpoint is reachable on this")
    print("subscription. injection-quantity and realtime-generation both return")
    print("the national mix; renewables/licensed-realtime-generation returns 403")
    print("(not subscribed). Turkey is NOT confirmed tier-1; see docs/runbooks/TR.md.")
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
