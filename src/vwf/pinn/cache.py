"""Assemble the per-region tensors the physics-informed model trains on.

One cache per region and split. It holds everything the forward model needs and
nothing it does not: the daily ERA5 fields at each turbine's own location, the
static physiography of that location, the fleet metadata, the power curves, and
the observed monthly capacity factors.

The fleet, the observations and the curve assignment come from exactly the paths
``vwf.data.train_set`` and ``vwf.data.val_set`` use, so the model is fitted to
the same data the incumbent affine correction is fitted to and any difference in
result is a difference of method. What the cache adds is the within-day wind
spread and the shear exponent, which the daily-mean pipeline discards.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vwf.config import PyVWFPaths
from vwf.data import clean_obs_data, load_power_curves, prep_country
from vwf.harness.driver import resolve_source
from vwf.harness.regions import RegionSpec
from vwf.pinn.era5_stats import daily_stats_at_points
from vwf.pinn.terrain import terrain_descriptors

OBS_COLS = [f"obs_{m}" for m in range(1, 13)]


@dataclass
class RegionCache:
    """Everything one region's forward model needs, aligned on a turbine axis."""

    code: str
    split: str
    dates: pd.DatetimeIndex          # (T,)
    meta: pd.DataFrame               # (N,) turbine metadata + terrain features
    obs: pd.DataFrame                # long [ID, year, month, obs]
    w_mean: np.ndarray               # (T, N) daily mean 100 m wind, m/s
    w_std: np.ndarray                # (T, N) within-day wind spread, m/s
    z0: np.ndarray                   # (T, N) roughness as the incumbent sees it
    shear: np.ndarray                # (T, N) 10-100 m power-law exponent
    curve_speeds: np.ndarray         # (S,) power-curve speed grid, m/s
    curve_cf: np.ndarray             # (M, S) capacity factor per model
    curve_names: list[str]           # (M,) model names, index-aligned to curve_cf
    turbine_curve: np.ndarray        # (N,) index into curve_cf for each turbine

    def __repr__(self) -> str:       # pragma: no cover - convenience only
        return (f"RegionCache({self.code}/{self.split}: {len(self.meta)} units, "
                f"{len(self.dates)} days, {len(self.obs)} obs rows)")


def _hourly_dir(spec: RegionSpec) -> Path:
    """The hourly ERA5 directory, even when the config points at a daily one.

    US and BR configs point at pre-aggregated ``*_daily`` directories to skip the
    averaging cost. The within-day spread cannot be recovered from those, so the
    raw hourly directory beside them is used instead.
    """
    configured = PyVWFPaths.INPUT_ROOT / spec.era5_path
    if configured.name.endswith("_daily"):
        raw = configured.with_name(configured.name[: -len("_daily")])
        if raw.is_dir() and any(raw.glob("*.nc")):
            return raw
    return configured


def _observations(spec: RegionSpec, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fleet metadata and long-format monthly observations for one split."""
    is_train = split == "train"
    year_test = None if is_train else int(spec.test_years[0])
    source = resolve_source(spec, "train" if is_train else "test")
    obs_data, turb_info = prep_country(
        spec.code, year_test, obs_level="turbine", source=source
    )
    obs = clean_obs_data(obs_data, spec.code, is_train)

    if is_train:
        # train_set keeps only units observed across the whole window, so the
        # fitted factors are not tilted by units that appear part-way through.
        span = int(obs.year.max() - obs.year.min()) + 1
        obs = obs[obs.groupby("ID").ID.transform("count") == span].reset_index(drop=True)
    else:
        obs = obs.copy()
        obs["year"] = int(spec.test_years[0])

    obs = obs[["ID", "year", *OBS_COLS]]
    obs.columns = ["ID", "year", *[str(m) for m in range(1, 13)]]
    obs = obs.melt(id_vars=["ID", "year"], var_name="month", value_name="obs")
    obs["month"] = obs["month"].astype(int)
    obs["year"] = obs["year"].astype(int)
    obs["ID"] = obs["ID"].astype(str)

    turb_info["ID"] = turb_info["ID"].astype(str)
    keep = set(obs["ID"]) & set(turb_info["ID"])
    obs = obs[obs["ID"].isin(keep)].reset_index(drop=True)
    turb_info = turb_info[turb_info["ID"].isin(keep)].reset_index(drop=True)
    return turb_info, obs


def _curve_matrix(turb_info: pd.DataFrame) -> tuple[np.ndarray, np.ndarray,
                                                    list[str], np.ndarray]:
    """Dense capacity-factor curves for the models this fleet actually uses."""
    table = load_power_curves()
    speeds = table["data$speed"].to_numpy(dtype="float64")
    available = [c for c in table.columns if c != "data$speed"]
    if not available:
        raise ValueError("power-curve table has no model columns")

    wanted = list(dict.fromkeys(turb_info["model"].astype(str)))
    default = available[0]
    names, missing = [], []
    for m in wanted:
        if m in available:
            names.append(m)
        else:
            missing.append(m)
    if missing:
        # Mirror vwf.wind's warned fallback rather than failing the whole region.
        print(f"  [curves] {len(missing)} model(s) absent from the library, "
              f"falling back to {default!r}: {missing[:5]}"
              f"{' ...' if len(missing) > 5 else ''}")
        if default not in names:
            names.append(default)

    cf = np.stack([table[m].to_numpy(dtype="float64") for m in names])
    index = {m: i for i, m in enumerate(names)}
    # Resolved once: a dict.get default argument is evaluated on every call, so
    # index[default] there would raise whenever the fallback was not needed.
    fallback = index.get(default, 0)
    turbine_curve = np.array(
        [index.get(str(m), fallback) for m in turb_info["model"]], dtype="int64"
    )
    return speeds, cf, names, turbine_curve


def build_cache(spec: RegionSpec, split: str = "train") -> RegionCache:
    """Build one region/split cache from the same inputs the incumbent uses."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    turb_info, obs = _observations(spec, split)
    years = (list(range(int(spec.train_years[0]), int(spec.train_years[-1]) + 1))
             if split == "train" else [int(spec.test_years[0])])
    print(f"  {spec.code}/{split}: {len(turb_info)} units, years {years}")

    lon = turb_info["lon"].to_numpy(dtype=float)
    lat = turb_info["lat"].to_numpy(dtype=float)

    dates, w_mean, w_std, z0, shear = daily_stats_at_points(
        _hourly_dir(spec), spec.bbox, lon, lat, years
    )
    terr = terrain_descriptors(
        lon, lat, PyVWFPaths.INPUT_ROOT / "reference" / "terrain" / "etopo_global.nc"
    )
    meta = pd.concat([turb_info.reset_index(drop=True), terr], axis=1)

    speeds, cf, names, turbine_curve = _curve_matrix(turb_info)
    return RegionCache(
        code=spec.code, split=split, dates=dates, meta=meta, obs=obs,
        w_mean=w_mean, w_std=w_std, z0=z0, shear=shear,
        curve_speeds=speeds, curve_cf=cf, curve_names=names,
        turbine_curve=turbine_curve,
    )


def save_cache(cache: RegionCache, root: str | Path) -> Path:
    """Persist a cache under ``root/<CODE>_<split>/``."""
    d = Path(root) / f"{cache.code}_{cache.split}"
    d.mkdir(parents=True, exist_ok=True)
    cache.meta.to_csv(d / "meta.csv", index=False)
    cache.obs.to_csv(d / "obs.csv", index=False)
    np.savez_compressed(
        d / "fields.npz",
        dates=cache.dates.values.astype("datetime64[ns]").astype("int64"),
        w_mean=cache.w_mean, w_std=cache.w_std, z0=cache.z0, shear=cache.shear,
        curve_speeds=cache.curve_speeds, curve_cf=cache.curve_cf,
        curve_names=np.array(cache.curve_names, dtype=object),
        turbine_curve=cache.turbine_curve,
    )
    return d


def load_cache(code: str, split: str, root: str | Path) -> RegionCache:
    """Load a cache written by :func:`save_cache`."""
    d = Path(root) / f"{code}_{split}"
    z = np.load(d / "fields.npz", allow_pickle=True)
    return RegionCache(
        code=code, split=split,
        dates=pd.DatetimeIndex(z["dates"].astype("datetime64[ns]")),
        meta=pd.read_csv(d / "meta.csv", dtype={"ID": str}),
        obs=pd.read_csv(d / "obs.csv", dtype={"ID": str}),
        w_mean=z["w_mean"], w_std=z["w_std"], z0=z["z0"], shear=z["shear"],
        curve_speeds=z["curve_speeds"], curve_cf=z["curve_cf"],
        curve_names=list(z["curve_names"]), turbine_curve=z["turbine_curve"],
    )
