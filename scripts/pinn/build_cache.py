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

Run: PYVWF_INPUT=input/combined PYTHONPATH=src /opt/anaconda3/bin/python \
         scripts/pinn/build_cache.py --regions DK DE UK US BR
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vwf.provenance import build_manifest, write_manifest  # noqa: E402
from vwf.harness.regions import load_region  # noqa: E402
from vwf.pinn.cache import build_cache, save_cache  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
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

    built, failed = {}, {}
    for code in args.regions:
        spec = load_region(paths[code])
        if spec.code != code:
            raise SystemExit(f"{paths[code]} is region {spec.code}, not {code}")
        for split in args.splits:
            t0 = time.time()
            print(f"[{code}/{split}] building...")
            try:
                cache = build_cache(spec, split)
            except Exception as e:  # noqa: BLE001
                print(f"[{code}/{split}] FAILED: {type(e).__name__}: {e}")
                failed[f"{code}/{split}"] = f"{type(e).__name__}: {e}"
                continue
            d = save_cache(cache, args.out)
            built[f"{code}/{split}"] = {
                "dir": str(d),
                "units": len(cache.meta),
                "days": len(cache.dates),
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
                "built": built,
                "failed": failed,
            }
        ),
    )
    if failed:
        raise SystemExit(f"{len(failed)} cache(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
