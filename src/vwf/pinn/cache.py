"""Assemble the per-region tensors the physics-informed model trains on.

One cache per region and split. It holds everything the forward model needs and
nothing it does not: the daily ERA5 fields at each turbine's own location, the
static physiography of that location, the fleet metadata, the power curves, and
the observed monthly capacity factors.

Two observation tiers are supported. A turbine-level cache holds one monthly
capacity factor per unit. A country-level cache holds the grid points of a
national fleet as its units and one national monthly capacity factor, with the
capacity of every point in every year of the split, so a prediction can be
aggregated to the national series the observation describes.

The fleet, the observations and the curve assignment come from exactly the paths
``vwf.data.train_set`` and ``vwf.data.val_set`` use, so the model is fitted to
the same data the incumbent affine correction is fitted to and any difference in
result is a difference of method. What the cache adds is the within-day wind
spread and the shear exponent, which the daily-mean pipeline discards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from vwf.config import PyVWFPaths
from vwf.data import clean_obs_data, country_cf_to_monthly, load_power_curves, prep_country
from vwf.harness.driver import resolve_source
from vwf.harness.regions import RegionSpec
from vwf.sources.entsoe_files import EntsoeFileSource
from vwf.pinn.era5_stats import daily_stats_at_points
from vwf.pinn.terrain import terrain_descriptors
from vwf.wind import loaded_extent_coverage

OBS_COLS = [f"obs_{m}" for m in range(1, 13)]
ERA5_RECORD_NAME = "era5_record.json"
FLEET_RECORD_NAME = "fleet_record.json"
#: The ID a country-level cache gives its one national observation series.
NATIONAL_ID = "__national__"
#: Country grid files store capacity in MW; turbine metadata stores kW.
MW_TO_KW = 1000.0


@dataclass
class RegionCache:
    """Everything one region's forward model needs, aligned on a turbine axis."""

    code: str
    split: str
    dates: pd.DatetimeIndex  # (T,)
    meta: pd.DataFrame  # (N,) turbine metadata + terrain features
    obs: pd.DataFrame  # long [ID, year, month, obs]
    w_mean: np.ndarray  # (T, N) daily mean 100 m wind, m/s
    w_std: np.ndarray  # (T, N) within-day wind spread, m/s
    z0: np.ndarray  # (T, N) roughness as the incumbent sees it
    shear: np.ndarray  # (T, N) 10-100 m power-law exponent
    curve_speeds: np.ndarray  # (S,) power-curve speed grid, m/s
    curve_cf: np.ndarray  # (M, S) capacity factor per model
    curve_names: list[str]  # (M,) model names, index-aligned to curve_cf
    turbine_curve: np.ndarray  # (N,) index into curve_cf for each turbine
    # What the ERA5 reduction read and applied, and where the fleet lies against
    # the loaded extent. Empty for caches written before it was recorded.
    era5_record: dict[str, Any] = field(default_factory=dict)
    # "turbine" or "country". Caches written before the country tier existed
    # load as turbine-level, which is what they are.
    level: str = "turbine"
    # Country level only: the years of the split, and each grid point's
    # capacity in kW in each of them, (Y, N) aligned to ``meta``.
    capacity_years: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype="int64"))
    capacity_by_year: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    # What the fleet and observations were read from, and curve coverage.
    fleet_record: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"RegionCache({self.code}/{self.split}: {len(self.meta)} units, "
            f"{len(self.dates)} days, {len(self.obs)} obs rows)"
        )


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
    obs_data, turb_info = prep_country(spec.code, year_test, obs_level="turbine", source=source)
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


def _country_observations(
    spec: RegionSpec, split: str
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    """Grid points, the national monthly series, and per-year capacities.

    The grid points are the fleet the harness simulates for the split: the
    train-end year for training and the test year for testing, both through
    ``EntsoeFileSource``. The observation is the energy-weighted national
    monthly capacity factor (``vwf.data.country_cf_to_monthly``), restricted
    to the split's years.

    Every year of the split also gets its own capacities, so a training month
    is aggregated with the fleet of its own year rather than the train-end
    snapshot. Each year's file is resolved by the exact name the source builds
    for that year, and a resolution to any other file is refused: the static
    fallback grid is a uniform lattice for SE and NO, not a fleet. The points
    must be the same in every year, which is how the per-year grids are built.

    Returns:
        ``(grid, obs, capacity_years, capacity_by_year, record)``. ``grid``
        carries capacity in kW, from the split's fleet year. ``obs`` has the
        national series under :data:`NATIONAL_ID`.
    """
    is_train = split == "train"
    source = resolve_source(spec, "train" if is_train else "test")
    if not isinstance(source, EntsoeFileSource):
        raise ValueError(
            f"{spec.code}: country-level caches support the national "
            f"'entsoe-country' source only, got {type(source).__name__}"
        )
    grid = prepare_grid(source.load_metadata())
    obs_raw = source.load_observations()
    monthly = country_cf_to_monthly(obs_raw)

    years = (
        list(range(int(spec.train_years[0]), int(spec.train_years[-1]) + 1))
        if is_train
        else [int(spec.test_years[0])]
    )
    monthly = monthly[monthly["year"].isin(years)].dropna(subset=["obs"])
    obs = monthly.assign(ID=NATIONAL_ID)[["ID", "year", "month", "obs"]].reset_index(drop=True)

    ids = grid["ID"].astype(str).to_numpy()
    capacity = np.zeros((len(years), len(ids)))
    grid_files = {}
    for k, year in enumerate(years):
        year_source = EntsoeFileSource(spec.code, "test", spec.train_years, year)
        path = year_source._grid_points_path()
        expected = f"{spec.code.lower()}_grid_points_{year}.csv"
        if path.name != expected:
            raise FileNotFoundError(
                f"{spec.code}: no per-year grid for {year}; the source resolved "
                f"{path.name}, not {expected}. A static grid is not a fleet for "
                "that year and is refused."
            )
        year_grid = prepare_grid(year_source.load_metadata()).set_index("ID")
        if set(year_grid.index) != set(ids):
            raise ValueError(
                f"{spec.code}: {path.name} holds different grid points from the "
                "split's fleet file, so its capacities cannot be aligned"
            )
        year_grid = year_grid.loc[list(ids)]
        moved = ~(
            np.isclose(year_grid["lon"].to_numpy(dtype=float), grid["lon"].to_numpy(dtype=float))
            & np.isclose(year_grid["lat"].to_numpy(dtype=float), grid["lat"].to_numpy(dtype=float))
        )
        if moved.any():
            raise ValueError(
                f"{spec.code}: {int(moved.sum())} grid point(s) in {path.name} "
                "sit at different coordinates from the split's fleet file"
            )
        capacity[k] = year_grid["capacity"].to_numpy(dtype=float)
        grid_files[str(year)] = str(path)

    record = {
        "fleet_file": str(source._grid_points_path()),
        "fleet_year": int(source.fleet_year),
        "grid_files_by_year": grid_files,
        "observation_file": str(source._obs_path()),
        "observation_target": "energy-weighted national monthly capacity factor",
        "observation_months": int(len(obs)),
        "grid_points": int(len(ids)),
        "capacity_mw_by_year": {
            str(y): float(capacity[k].sum() / MW_TO_KW) for k, y in enumerate(years)
        },
        "points_with_capacity_by_year": {
            str(y): int((capacity[k] > 0).sum()) for k, y in enumerate(years)
        },
    }
    return grid, obs, np.array(years, dtype="int64"), capacity, record


def prepare_grid(grid: pd.DataFrame) -> pd.DataFrame:
    """Coerce a country grid-point file and put its capacity in kW.

    Rows missing a position, a height, a capacity or a model are dropped, as
    ``vwf.data.prepare_country_fleet`` drops them. The conversion from MW puts
    both tiers' capacities in one unit, which the capacity-density features
    require.
    """
    grid = grid.copy()
    grid["ID"] = grid["ID"].astype(str)
    for column in ("capacity", "height", "lon", "lat"):
        grid[column] = pd.to_numeric(grid[column], errors="coerce")
    grid = grid.dropna(subset=["capacity", "height", "lon", "lat", "model"]).reset_index(drop=True)
    grid["capacity"] = grid["capacity"] * MW_TO_KW
    return grid


def _curve_matrix(
    turb_info: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, list[str]]:
    """Dense capacity-factor curves for the models this fleet actually uses.

    Also returns the model keys absent from ``power_curves.csv``, whose units
    are simulated on the table's first column.
    """
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
        print(
            f"  [curves] {len(missing)} model(s) absent from the library, "
            f"falling back to {default!r}: {missing[:5]}"
            f"{' ...' if len(missing) > 5 else ''}"
        )
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
    return speeds, cf, names, turbine_curve, missing


def build_cache(spec: RegionSpec, split: str = "train") -> RegionCache:
    """Build one region/split cache from the same inputs the incumbent uses."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    years = (
        list(range(int(spec.train_years[0]), int(spec.train_years[-1]) + 1))
        if split == "train"
        else [int(spec.test_years[0])]
    )
    if spec.obs_level == "country":
        turb_info, obs, capacity_years, capacity_by_year, fleet_record = _country_observations(
            spec, split
        )
    elif spec.obs_level == "turbine":
        turb_info, obs = _observations(spec, split)
        capacity_years = np.zeros(0, dtype="int64")
        capacity_by_year = np.zeros((0, 0))
        fleet_record = {}
    else:
        raise ValueError(f"{spec.code}: unsupported obs_level {spec.obs_level!r}")
    print(f"  {spec.code}/{split}: {len(turb_info)} units ({spec.obs_level}), years {years}")

    lon = turb_info["lon"].to_numpy(dtype=float)
    lat = turb_info["lat"].to_numpy(dtype=float)

    dates, w_mean, w_std, z0, shear, reduction = daily_stats_at_points(
        _hourly_dir(spec), spec.bbox, lon, lat, years, roughness=spec.roughness
    )
    era5_record = era5_record_for(reduction, turb_info, spec)
    terr = terrain_descriptors(
        lon, lat, PyVWFPaths.INPUT_ROOT / "reference" / "terrain" / "etopo_global.nc"
    )
    meta = pd.concat([turb_info.reset_index(drop=True), terr], axis=1)

    speeds, cf, names, turbine_curve, missing = _curve_matrix(turb_info)
    is_missing = turb_info["model"].astype(str).isin(missing).to_numpy()
    total = float(turb_info["capacity"].sum())
    fleet_record = {
        "level": spec.obs_level,
        **fleet_record,
        "curves": {
            "missing_model_keys": list(missing),
            "units_substituted": int(is_missing.sum()),
            "capacity_share_substituted": (
                float(turb_info.loc[is_missing, "capacity"].sum()) / total if total > 0 else 0.0
            ),
        },
    }
    return RegionCache(
        code=spec.code,
        split=split,
        dates=dates,
        meta=meta,
        obs=obs,
        w_mean=w_mean,
        w_std=w_std,
        z0=z0,
        shear=shear,
        curve_speeds=speeds,
        curve_cf=cf,
        curve_names=names,
        turbine_curve=turbine_curve,
        era5_record=era5_record,
        level=spec.obs_level,
        capacity_years=capacity_years,
        capacity_by_year=capacity_by_year,
        fleet_record=fleet_record,
    )


def era5_record_for(
    reduction: dict[str, Any], turb_info: pd.DataFrame, spec: RegionSpec
) -> dict[str, Any]:
    """The run record's ERA5 entries, in the harness manifest's own format.

    ``era5_roughness`` and ``era5_extent`` carry the same keys as a harness
    manifest, so a physics-informed run and a scorecard row can be compared
    field by field. One difference is stated in the record itself: the
    harness extrapolates winds to units outside the loaded extent when a
    region opts in, and this path never does. Those units get no wind, and
    :func:`vwf.pinn.train.RegionTensors.from_cache` drops them.
    """
    lon_min, lon_max, lat_min, lat_max = reduction["loaded_extent"]
    grid = xr.Dataset(coords={"lon": [lon_min, lon_max], "lat": [lat_min, lat_max]})
    coverage = loaded_extent_coverage(grid, turb_info)
    return {
        "era5_dir": reduction["era5_dir"],
        "n_files_read": reduction["n_files_read"],
        "era5_roughness": reduction["roughness"],
        "era5_extent": {
            **coverage,
            "requested_bbox": list(spec.bbox),
            "allow_extrapolation": bool(spec.allow_extrapolation),
            "meaning": (
                "units outside the loaded ERA5 extent are not simulated by vwf.pinn: "
                "they are dropped, never extrapolated, whatever allow_extrapolation says"
            ),
        },
    }


def save_cache(cache: RegionCache, root: str | Path) -> Path:
    """Persist a cache under ``root/<CODE>_<split>/``."""
    d = Path(root) / f"{cache.code}_{cache.split}"
    d.mkdir(parents=True, exist_ok=True)
    cache.meta.to_csv(d / "meta.csv", index=False)
    cache.obs.to_csv(d / "obs.csv", index=False)
    np.savez_compressed(
        d / "fields.npz",
        dates=cache.dates.values.astype("datetime64[ns]").astype("int64"),
        w_mean=cache.w_mean,
        w_std=cache.w_std,
        z0=cache.z0,
        shear=cache.shear,
        curve_speeds=cache.curve_speeds,
        curve_cf=cache.curve_cf,
        curve_names=np.array(cache.curve_names, dtype=object),
        turbine_curve=cache.turbine_curve,
        level=np.array(cache.level),
        capacity_years=cache.capacity_years,
        capacity_by_year=cache.capacity_by_year,
    )
    for name, record in (
        (ERA5_RECORD_NAME, cache.era5_record),
        (FLEET_RECORD_NAME, cache.fleet_record),
    ):
        with open(d / name, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
            fh.write("\n")
    return d


def load_cache(code: str, split: str, root: str | Path) -> RegionCache:
    """Load a cache written by :func:`save_cache`."""
    d = Path(root) / f"{code}_{split}"
    z = np.load(d / "fields.npz", allow_pickle=True)
    records = {}
    for name in (ERA5_RECORD_NAME, FLEET_RECORD_NAME):
        path = d / name
        records[name] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    # Caches written before the country tier carry none of these three arrays.
    level = str(z["level"]) if "level" in z.files else "turbine"
    capacity_years = (
        z["capacity_years"] if "capacity_years" in z.files else np.zeros(0, dtype="int64")
    )
    capacity_by_year = z["capacity_by_year"] if "capacity_by_year" in z.files else np.zeros((0, 0))
    return RegionCache(
        code=code,
        split=split,
        dates=pd.DatetimeIndex(z["dates"].astype("datetime64[ns]")),
        meta=pd.read_csv(d / "meta.csv", dtype={"ID": str}),
        obs=pd.read_csv(d / "obs.csv", dtype={"ID": str}),
        w_mean=z["w_mean"],
        w_std=z["w_std"],
        z0=z["z0"],
        shear=z["shear"],
        curve_speeds=z["curve_speeds"],
        curve_cf=z["curve_cf"],
        curve_names=list(z["curve_names"]),
        turbine_curve=z["turbine_curve"],
        era5_record=records[ERA5_RECORD_NAME],
        level=level,
        capacity_years=capacity_years,
        capacity_by_year=capacity_by_year,
        fleet_record=records[FLEET_RECORD_NAME],
    )
