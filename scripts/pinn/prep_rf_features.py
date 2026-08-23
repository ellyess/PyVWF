#!/usr/bin/env python3
"""Precompute the published SET_A terrain features the RF baseline needs.

The features depend only on position, so they are the same for every
leave-one-out fold. Deriving them means differentiating a 22-million-pixel
ETOPO window for the American bounding box, which is the largest memory spike
anywhere in this workstream. Doing it here, once, in a process that holds
nothing else, keeps that spike out of the long fitting run -- where an
out-of-memory kill would cost hours rather than a minute.

Run: PYTHONPATH=src /opt/anaconda3/bin/python scripts/pinn/prep_rf_features.py
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vwf.pinn.cache import load_cache  # noqa: E402
from analysis.ml_transfer_retest import (  # noqa: E402
    RUNS, build_centroids, terrain_features,
)

CACHE = ROOT / "output" / "pinn" / "cache"
OUT = ROOT / "output" / "pinn" / "rf_features"
REGIONS = ["DK", "DE", "UK", "US", "BR"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    centroids = build_centroids(RUNS)
    for x in ("lon", "lat"):
        centroids[f"{x}_norm"] = ((centroids[x] - centroids[x].min())
                                  / (centroids[x].max() - centroids[x].min()))
    lon_lo, lon_hi = centroids.lon.min(), centroids.lon.max()
    lat_lo, lat_hi = centroids.lat.min(), centroids.lat.max()
    centroids = terrain_features(centroids)
    centroids.to_csv(OUT / "centroids.csv", index=False)
    print(f"centroids: {centroids.shape} -> {OUT/'centroids.csv'}")
    del centroids
    gc.collect()

    for code in REGIONS:
        meta = load_cache(code, "test", CACHE).meta
        units = meta[["ID", "lon", "lat"]].copy()
        units["ID"] = units["ID"].astype(str)
        units["region"] = code
        # Normalised against the TRAINING centroids' span, so a holdout region
        # can land outside [0, 1]; that is what extrapolation looks like and it
        # is deliberately not clipped.
        units["lon_norm"] = (units["lon"] - lon_lo) / (lon_hi - lon_lo)
        units["lat_norm"] = (units["lat"] - lat_lo) / (lat_hi - lat_lo)
        units["abs_lat"] = units["lat"].abs()
        feats = terrain_features(units)
        feats.to_csv(OUT / f"units_{code}.csv", index=False)
        print(f"{code}: {feats.shape} -> {OUT/f'units_{code}.csv'}")
        del units, feats, meta
        gc.collect()


if __name__ == "__main__":
    main()
