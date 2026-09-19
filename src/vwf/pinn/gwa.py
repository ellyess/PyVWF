"""A wind-atlas speed-up ratio per unit, for a correction that learns nothing.

The closest physical model chain in the literature (Nayak et al. 2025, Applied
Energy 402, 126882) scales reanalysis wind at each plant by the ratio of the
Global Wind Atlas mean speed to the reanalysis mean speed. It needs no
observations, which makes it the fair comparison for a terrain correction
learned from generation: if an atlas applied blind does as well, the learned
term is not worth its data.

The ratio here is ``R = GWA / E``: ``GWA`` is the atlas's mean 100 m speed over
the cells within a radius of the unit, and ``E`` is the unit's mean ERA5 100 m
speed. It is clipped to the speed-up's own bounds, and a unit with no atlas
value within the radius gets 1. Nothing here imports torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EARTH_R_KM = 6371.0
KM_PER_DEG = np.pi * EARTH_R_KM / 180.0
#: The learned speed-up's bounds (``vwf.pinn.model.GAMMA_BOUNDS``, as ratios).
RATIO_BOUNDS = (0.67, 2.46)


def atlas_means(
    raster_path: str | Path, lon: np.ndarray, lat: np.ndarray, radius_km: float = 2.5
) -> np.ndarray:
    """Mean atlas value over the cells whose centres lie within ``radius_km``.

    Cells equal to the raster's nodata value, not finite, or not above zero are
    ignored. A point with no valid cell within the radius returns NaN. The
    raster must be in geographic coordinates (EPSG:4326).
    """
    import rasterio
    from rasterio.windows import Window

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    out = np.full(len(lon), np.nan)
    with rasterio.open(raster_path) as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise ValueError(f"{raster_path}: expected EPSG:4326, got {src.crs}")
        nodata = src.nodata
        dx, dy = abs(src.res[0]), abs(src.res[1])
        for k, (x, y) in enumerate(zip(lon, lat)):
            half_lat = radius_km / KM_PER_DEG
            half_lon = half_lat / max(np.cos(np.radians(y)), 1e-6)
            row0, col0 = src.index(x - half_lon, y + half_lat)
            row1, col1 = src.index(x + half_lon, y - half_lat)
            r0, r1 = max(min(row0, row1), 0), min(max(row0, row1) + 1, src.height)
            c0, c1 = max(min(col0, col1), 0), min(max(col0, col1) + 1, src.width)
            if r1 <= r0 or c1 <= c0:
                continue
            block = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(float)
            rows = np.arange(r0, r1)[:, None]
            cols = np.arange(c0, c1)[None, :]
            cx = src.transform.c + (cols + 0.5) * dx
            cy = src.transform.f - (rows + 0.5) * dy
            dist = KM_PER_DEG * np.hypot((cx - x) * np.cos(np.radians(y)), cy - y)
            valid = np.isfinite(block) & (block > 0) & (dist <= radius_km)
            if nodata is not None:
                valid &= block != nodata
            if valid.any():
                out[k] = float(block[valid].mean())
    return out


def gwa_ratio(
    ids,
    lon,
    lat,
    era5_mean,
    raster_path,
    radius_km: float = 2.5,
    bounds: tuple[float, float] = RATIO_BOUNDS,
) -> pd.DataFrame:
    """The clipped atlas-to-ERA5 ratio per unit, with how each value arose.

    Returns:
        One row per unit: ``ID``, ``gwa_mean``, ``era5_mean``, ``ratio_raw``,
        ``ratio``, ``clipped`` and ``neutral``.
    """
    gwa = atlas_means(raster_path, lon, lat, radius_km)
    era5 = np.asarray(era5_mean, dtype=float)
    raw = gwa / era5
    neutral = ~(np.isfinite(raw) & (raw > 0))
    ratio = np.where(neutral, 1.0, np.clip(raw, *bounds))
    clipped = ~neutral & ((raw < bounds[0]) | (raw > bounds[1]))
    return pd.DataFrame(
        {
            "ID": [str(i) for i in ids],
            "gwa_mean": gwa,
            "era5_mean": era5,
            "ratio_raw": raw,
            "ratio": ratio,
            "clipped": clipped,
            "neutral": neutral,
        }
    )
