#!/usr/bin/env python3
"""Build the per-region tensor caches the physics-informed model trains on.

Uses the licensed combined curve library by default, because that is what the
canonical runs the results are compared against were built on. Point
PYVWF_INPUT at ``input`` instead to run on the open library.

Run: PYVWF_INPUT=input/combined PYTHONPATH=src /opt/anaconda3/bin/python \
         scripts/pinn/build_cache.py --regions DK DE UK US BR
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
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
    args = ap.parse_args()

    for code in args.regions:
        spec = load_region(CONFIGS / f"{code.lower()}.toml")
        for split in args.splits:
            t0 = time.time()
            print(f"[{code}/{split}] building...")
            try:
                cache = build_cache(spec, split)
            except Exception as e:                        # noqa: BLE001
                print(f"[{code}/{split}] FAILED: {type(e).__name__}: {e}")
                continue
            d = save_cache(cache, args.out)
            print(f"[{code}/{split}] {cache}  -> {d}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
