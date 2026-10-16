#!/usr/bin/env python3
"""Build the per-region tensor caches the physics-informed model trains on.

Uses the licensed combined curve library by default, because that is what the
canonical runs the results are compared against were built on. Point
PYVWF_INPUT at ``input`` instead to run on the open library.

Each region/split cache records what its ERA5 reduction read, the roughness
treatment requested and applied, and where the fleet lies against the loaded
extent (``era5_record.json``). A ``run_manifest.json`` in ``--out`` records the
git state, curve library and the configs by sha256. The build refuses a dirty
tree unless ``--allow-dirty`` is given.

Run: PYVWF_INPUT=input/combined PYTHONPATH=src /opt/miniconda3/envs/pyvwf/bin/python \
         scripts/studies/physics-informed/build_cache.py --regions DK DE UK US BR
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import pandas as pd  # noqa: E402

from pyvwf.provenance import build_manifest, write_manifest  # noqa: E402
from pyvwf.harness.regions import load_region  # noqa: E402
from pyvwf.pinn.cache import build_cache, save_cache  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "configs" / "regions"
CACHE = ROOT / "output" / "pinn" / "cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=["DK", "DE", "UK", "US", "BR"])
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--out", default=str(CACHE))
    ap.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="CODE=PATH",
        help="region config to load for CODE; repeatable. Regions "
        "not named use configs/regions/<stem>.toml",
    )
    ap.add_argument(
        "--fleet-override",
        action="append",
        default=[],
        metavar="CODE=PATH",
        help="per-unit coordinates and hub heights for CODE, as a CSV of ID, "
        "lon, lat, height; repeatable. A unit the file does not name keeps "
        "what it had (issue #12, the MaStR match for DE).",
    )
    ap.add_argument(
        "--registration",
        default=None,
        help="the registration this build answers to, recorded in the manifest",
    )
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="build on a tree with uncommitted changes; the manifest records it",
    )
    args = ap.parse_args()

    if build_manifest()["git_dirty"] and not args.allow_dirty:
        raise SystemExit(
            "refusing to build on a dirty tree: the caches would be "
            "unattributable. Commit first, or pass --allow-dirty."
        )

    named = {}
    for item in args.config:
        code, sep, path = item.partition("=")
        if not sep or not path:
            raise SystemExit(f"--config expects CODE=PATH, got {item!r}")
        named[code] = Path(path)
    # Region codes may carry a hyphen (AU-NEM) where the config filename uses an
    # underscore (au_nem.toml).
    paths = {c: named.get(c, CONFIGS / f"{c.lower().replace('-', '_')}.toml") for c in args.regions}

    overrides = {}
    for item in args.fleet_override:
        code, sep, path = item.partition("=")
        if not sep or not path:
            raise SystemExit(f"--fleet-override expects CODE=PATH, got {item!r}")
        overrides[code] = Path(path)
    unknown = set(overrides) - set(args.regions)
    if unknown:
        raise SystemExit(
            f"--fleet-override names regions that are not being built: {sorted(unknown)}"
        )

    built, failed = {}, {}
    for code in args.regions:
        spec = load_region(paths[code])
        if spec.code != code:
            raise SystemExit(f"{paths[code]} is region {spec.code}, not {code}")
        for split in args.splits:
            t0 = time.time()
            print(f"[{code}/{split}] building...")
            try:
                override = pd.read_csv(overrides[code]) if code in overrides else None
                cache = build_cache(spec, split, fleet_override=override)
            except Exception as e:  # noqa: BLE001
                print(f"[{code}/{split}] FAILED: {type(e).__name__}: {e}")
                failed[f"{code}/{split}"] = f"{type(e).__name__}: {e}"
                continue
            d = save_cache(cache, args.out)
            built[f"{code}/{split}"] = {
                "dir": str(d),
                "units": len(cache.meta),
                "days": len(cache.dates),
                "fleet_override": cache.fleet_record.get("fleet_override"),
                **cache.era5_record,
            }
            print(f"[{code}/{split}] {cache}  -> {d}  ({time.time() - t0:.0f}s)")

    write_manifest(
        args.out,
        build_manifest(
            extra={
                "run_mode": "pinn-cache",
                "registration": args.registration,
                "argv": sys.argv[1:],
                "configs": {
                    c: {"path": str(p), "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                    for c, p in paths.items()
                },
                "fleet_overrides": {
                    c: {
                        "path": str(p),
                        "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest(),
                    }
                    for c, p in overrides.items()
                },
                "built": built,
                "failed": failed,
            }
        ),
    )
    if failed:
        raise SystemExit(f"{len(failed)} cache(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
