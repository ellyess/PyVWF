"""Wind interpolation and simulation utilities for PyVWF.

Performance optimizations:
- Caches power curve Akima interpolators
- Optional turbine aggregation for massive speedups
- Vectorized operations where possible
"""

from __future__ import annotations

import warnings
import weakref
from typing import Any

import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator

from vwf.time_utils import add_time_resolution_columns
from vwf.utils import ensure_numeric

#: Dataset attribute carrying whether a run may extrapolate winds beyond the
#: loaded ERA5 extent. ``prep_era5`` sets it; ``interpolate_wind`` reads it, so
#: the permission travels with the data through every path that simulates.
EXTRAPOLATION_ATTR = "pyvwf_allow_extrapolation"


class ExtrapolationError(ValueError):
    """Units lie outside the loaded ERA5 extent and extrapolation is not allowed."""


def loaded_extent_coverage(reanalysis, turb_info) -> dict[str, Any]:
    """Where a fleet's units lie relative to the loaded ERA5 extent.

    The loaded extent is the lon/lat range of the grid a run actually loaded,
    after the bbox slice. A unit inside it has its winds interpolated between
    grid cells; a unit outside it would have them extrapolated linearly from
    the edge of the grid, which produces winds from data that does not exist
    (the European download stops at 42N, and Spanish grid points five degrees
    south of it were simulated at speeds down to -57.7 m/s).

    Inside the loaded extent is a statement about position only. It does not
    verify the data in the surrounding cells: a unit can sit inside the extent
    over cells whose values are unusable; :func:`off_curve_record` counts
    those.

    Args:
        reanalysis: Dataset with ``lon`` and ``lat`` coordinates.
        turb_info: Units with ``ID``, ``lon``, ``lat`` and ``capacity``.

    Returns:
        The loaded extent, the number and capacity share of units outside it,
        how far outside the furthest one lies (degrees), and up to ten of their
        IDs.
    """
    lon = np.asarray(reanalysis["lon"].values, dtype=float)
    lat = np.asarray(reanalysis["lat"].values, dtype=float)
    lon_min, lon_max, lat_min, lat_max = lon.min(), lon.max(), lat.min(), lat.max()
    x = np.asarray(turb_info["lon"], dtype=float)
    y = np.asarray(turb_info["lat"], dtype=float)
    capacity = np.asarray(turb_info["capacity"], dtype=float)
    beyond = np.maximum.reduce(
        [lon_min - x, x - lon_max, lat_min - y, y - lat_max, np.zeros_like(x)]
    )
    outside = beyond > 1e-9
    total = float(np.nansum(capacity))
    return {
        "loaded_extent": [float(lon_min), float(lon_max), float(lat_min), float(lat_max)],
        "units": int(len(x)),
        "units_outside_loaded_extent": int(outside.sum()),
        "capacity_share_outside_loaded_extent": (
            float(np.nansum(capacity[outside])) / total if total > 0 else 0.0
        ),
        "max_degrees_outside_loaded_extent": float(beyond.max()) if len(x) else 0.0,
        "ids_outside": [str(i) for i in np.asarray(turb_info["ID"])[outside][:10]],
    }


def off_curve_record(
    ws: pd.DataFrame, cf: pd.DataFrame, capacity: pd.Series, power_curves: pd.DataFrame
) -> dict[str, Any]:
    """How many simulated values the power curves could not convert, and why.

    A speed below the power curve table's first speed (0 m/s) or above its last
    (40 m/s) has no value on the curve. The Akima interpolator returns NaN
    there, not zero output, and a missing speed (for example an undefined
    roughness in the input) also gives NaN. A monthly or national mean then
    skips the value. So a unit-month can be scored on only some of its steps,
    and the steps it loses are the ones the simulation could not handle. Off
    the curve, the values are left missing: counting them as zero output is a
    separate question about the physics.

    Args:
        ws: Wide frame of simulated speeds, a ``time`` column plus one column
            per unit, as :func:`simulate_wind` returns.
        cf: The matching wide frame of capacity factors.
        capacity: Capacity by unit ID (string index), for the weights.
        power_curves: The power curve table the speeds were converted on.

    Returns:
        Capacity-weighted shares of unit-steps below the curve, above it, and
        with no speed, and the number of unit-months with every step missing
        and with some but not all.
    """
    cols = [c for c in ws.columns if c != "time"]
    speeds = ws[cols].to_numpy(dtype=float)
    grid = power_curves["data$speed"].to_numpy(dtype=float)
    weights = capacity.reindex([str(c) for c in cols]).to_numpy(dtype=float)
    weights = np.broadcast_to(np.nan_to_num(weights)[None, :], speeds.shape)
    total = float(weights.sum())

    def share(mask: np.ndarray) -> float:
        return float(weights[mask].sum()) / total if total > 0 else 0.0

    with np.errstate(invalid="ignore"):
        below = speeds < grid.min()
        above = speeds > grid.max()
    months = pd.to_datetime(cf["time"]).dt.to_period("M").to_numpy()
    missing = cf[cols].isna().groupby(months).mean().to_numpy()
    return {
        "off_curve_below_share": share(below),
        "off_curve_above_share": share(above),
        "no_speed_share": share(np.isnan(speeds)),
        "unit_months_wholly_missing": int((missing == 1.0).sum()),
        "unit_months_partly_missing": int(((missing > 0) & (missing < 1.0)).sum()),
    }


def fit_diagnostics(
    reanalysis,
    clus_info: pd.DataFrame,
    factors: pd.DataFrame,
    time_res: str,
    power_curves: pd.DataFrame,
    seasons=None,
    years: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Where a fitted correction sends the speeds it was fitted on.

    ``fit_quality`` bounds the scalar and checks that each offset converged. It
    never asks what the pair does to the speeds it is applied to. An affine
    pair sends every speed below ``-offset / scalar`` to a negative corrected
    speed, which has no value on the power curve and drops out of both the fit's
    objective and the score. In the Spanish country row, clusters 0 and 3 cross
    zero at 11.7 and 8.9 m/s, and more than half their training days fell below
    it. This applies each cluster's fitted factors to its own training winds and
    records how much of them lands off the curve.

    Args:
        reanalysis: The training reanalysis the factors were fitted on.
        clus_info: The fitted units, with ``cluster``, ``capacity`` and the
            columns :func:`interpolate_wind` needs.
        factors: The fitted factors table (``cluster``, the slice column,
            ``scalar``, ``offset``).
        time_res: The time slice the factors were fitted at.
        power_curves: The power curve table, for its speed range.
        seasons: Season definitions, as for :func:`correct_wind_speed`.
        years: Inclusive ``(first, last)`` training years to keep; the loaded
            reanalysis may hold more.

    Returns:
        One row per cluster, slice value and year: the scalar and offset, the
        zero-crossing speed (``-offset / scalar`` where the offset is negative,
        else NaN), the unit-steps, and the capacity-weighted steps in total,
        below 0 m/s and above the curve. The weighted counts are kept so that
        shares aggregate exactly.
    """
    ws = interpolate_wind(reanalysis, clus_info).transpose("time", "turbine")
    times = pd.DatetimeIndex(ws["time"].values)
    keep = np.ones(len(times), dtype=bool)
    if years is not None:
        keep = (times.year >= years[0]) & (times.year <= years[1])
    speeds = np.asarray(ws.values, dtype=float)[keep]
    times = times[keep]
    slices = add_time_resolution_columns(pd.DataFrame({"month": times.month}), seasons)[
        time_res
    ].to_numpy()
    grid = power_curves["data$speed"].to_numpy(dtype=float)
    top, bottom = float(grid.max()), float(grid.min())
    units = pd.Index(np.asarray(ws["turbine"].values).astype(str))
    info = clus_info.assign(ID=clus_info["ID"].astype(str)).set_index("ID").loc[units]
    cluster = info["cluster"].to_numpy()
    capacity = info["capacity"].to_numpy(dtype=float)
    rows = []
    for cl, value, scalar, offset in zip(
        factors["cluster"],
        factors[time_res],
        factors["scalar"].astype(float),
        factors["offset"].astype(float),
    ):
        cols = cluster == cl
        rows_t = slices == value
        if not cols.any() or not rows_t.any():
            continue
        corrected = speeds[np.ix_(rows_t, cols)] * scalar + offset
        weight = np.broadcast_to(capacity[cols][None, :], corrected.shape)
        valid = ~np.isnan(corrected)
        for year in sorted(set(times[rows_t].year)):
            in_year = (times[rows_t].year == year)[:, None] & valid
            rows.append(
                {
                    "cluster": cl,
                    time_res: value,
                    "year": int(year),
                    "scalar": scalar,
                    "offset": offset,
                    "zero_crossing_speed": (
                        -offset / scalar if offset < 0 and scalar > 0 else float("nan")
                    ),
                    "unit_steps": int(in_year.sum()),
                    "weight_steps": float(weight[in_year].sum()),
                    "weight_below_zero": float(weight[in_year & (corrected < bottom)].sum()),
                    "weight_above_curve": float(weight[in_year & (corrected > top)].sum()),
                }
            )
    return pd.DataFrame(rows)


# Global cache for power curve interpolators (cleared on module reload).
# Keyed by id() of the power-curve table, the cheapest key for a lookup made on
# every objective evaluation. id() is only unique while the table is alive, so
# each entry holds a weak reference to its table and is dropped when the table
# is garbage-collected: a later table that reuses the id cannot receive stale
# curves, and the cache does not grow for the life of the process. A table
# edited in place keeps its id and entry; build a new table instead.
_power_curve_cache: dict[int, dict[str, Any]] = {}


def default_curve_key(power_curves: pd.DataFrame) -> str | None:
    """The model whose curve stands in for any model the table lacks.

    It is the table's first model column, and its identity decides results:
    every unit whose model is missing from the table is simulated on this
    curve (see ``_CurveByModel``), and ``vwf.curves._default_power_curve`` assigns
    it to a country grid with no ``model`` column. For the bundled library it
    is a 100 kW distributed-wind machine, which is what every country-level run
    on that library simulated with. ``vwf.provenance.curve_resolution``
    calls this too, so the resolution log cannot disagree with the simulation,
    and ``tests/test_curve_resolution.py`` pins its value for the bundled
    library so that reordering the table cannot change results unnoticed.
    """
    cols = [c for c in power_curves.columns if c != "data$speed"]
    return cols[0] if cols else None


class _CurveByModel(dict):
    """Model name -> Akima power-curve interpolator, with a warned fallback.

    Looking up a model that has no column in the loaded power-curve table returns
    the default model's curve instead of raising ``KeyError``, after emitting a
    warning that names both the missing model and the fallback used. This keeps a
    run going when a configured or requested model is absent from the table (for
    example a proprietary model name against the shipped open library) without ever *silently*
    substituting: the warning makes the fallback explicit, so a real run cannot
    quietly use the wrong curve.

    When the requested model IS present, lookup behaves exactly like a plain
    ``dict`` (``__missing__`` is never called), so runs against a table that
    contains the model are unaffected.
    """

    def __init__(self, *args, default_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._default_model = default_model
        self._warned = set()

    def __missing__(self, model):
        default = self._default_model
        if default is None or default not in self:
            # Last resort: the first available model column, if any.
            default = next(iter(self)) if len(self) else None
        if default is None:
            # Genuinely empty table: nothing to fall back to.
            raise KeyError(model)
        if model not in self._warned:
            self._warned.add(model)
            warnings.warn(
                f"Power curve for model {model!r} not found in the loaded "
                f"power-curve table; falling back to {default!r}. Add the "
                f"model's column to power_curves.csv to avoid this fallback.",
                stacklevel=2,
            )
        return self[default]


def power_curve_arrays(power_curves):
    """The speeds and per-model curve arrays the simulation interpolates on.

    Returns ``(speeds, curve_by_model)``, from the same cache the simulation
    uses, so a study reads the curves exactly as they are applied.
    """
    return _get_power_curve_cache(power_curves)


def _get_power_curve_cache(powerCurveFile):
    """Return cached power curve arrays for a given power curve table."""
    cache_key = id(powerCurveFile)
    cached = _power_curve_cache.get(cache_key)
    if cached is not None and cached["table"]() is powerCurveFile:
        return cached["x"], cached["curve_by_model"]

    x = powerCurveFile["data$speed"].to_numpy()
    model_cols = [m for m in powerCurveFile.columns if m != "data$speed"]
    curve_by_model = _CurveByModel(
        {m: Akima1DInterpolator(x, powerCurveFile[m].to_numpy()) for m in model_cols},
        default_model=default_curve_key(powerCurveFile),
    )
    _power_curve_cache[cache_key] = {
        "table": weakref.ref(powerCurveFile),
        "x": x,
        "curve_by_model": curve_by_model,
    }
    weakref.finalize(powerCurveFile, _power_curve_cache.pop, cache_key, None)
    return x, curve_by_model


def aggregate_turbines_to_grid(turb_info: pd.DataFrame, reanalysis) -> pd.DataFrame:
    """Collapse turbines onto nearest reanalysis grid cell.

    This reduces interpolation cost by aggregating turbines to grid cells and
    height bins, summing capacity within each group.

    Args:
        turb_info: Turbine metadata with ``lat``, ``lon``, ``height``, and ``capacity``.
        reanalysis: Reanalysis dataset with ``lat`` and ``lon`` coordinates.

    Returns:
        DataFrame with columns ``ID``, ``lat``, ``lon``, ``height``, ``capacity``, and ``model``.

    Raises:
        ValueError: If no valid turbines remain after cleaning.
    """
    ti = turb_info.copy()

    # OPTIMIZATION: Use vwf.utils helper (2-3x faster)
    ti = ensure_numeric(ti, ["lat", "lon", "height", "capacity"])
    ti["ID"] = ti["ID"].astype(str)

    # Drop unusable rows
    ti = ti.dropna(subset=["lat", "lon", "height", "capacity"]).reset_index(drop=True)
    if ti.empty:
        raise ValueError("aggregate_turbines_to_grid: turb_info has no valid rows after cleaning.")

    # Reanalysis grid coordinates
    grid_lats = np.asarray(reanalysis["lat"].values)
    grid_lons = np.asarray(reanalysis["lon"].values)

    # Nearest gridpoint index for each turbine
    lat_idx = np.abs(ti["lat"].to_numpy()[:, None] - grid_lats[None, :]).argmin(axis=1)
    lon_idx = np.abs(ti["lon"].to_numpy()[:, None] - grid_lons[None, :]).argmin(axis=1)

    ti["lat_cell"] = grid_lats[lat_idx]
    ti["lon_cell"] = grid_lons[lon_idx]

    # Optional: bin heights to reduce unique heights further (change bin if you want)
    ti["height_bin"] = (ti["height"] / 10.0).round().astype(int) * 10.0

    # Model: for country-level, one model is usually enough, but keep per-model if present
    if "model" not in ti.columns:
        ti["model"] = None

    g = ti.groupby(["lat_cell", "lon_cell", "height_bin", "model"], dropna=False, as_index=False)

    out = g.agg(
        capacity=("capacity", "sum"),
        lat=("lat_cell", "first"),
        lon=("lon_cell", "first"),
        height=("height_bin", "first"),
    )

    # Create stable IDs
    out["ID"] = (
        out["lat"].astype(str)
        + "_"
        + out["lon"].astype(str)
        + "_"
        + out["height"].astype(str)
        + "_"
        + out["model"].astype(str)
    )

    return out[["ID", "lat", "lon", "height", "capacity", "model"]]


def simulate_country_cf(
    reanalysis,
    turb_info,
    powerCurveFile,
    bc_factors=None,
    time_res=None,
    *,
    resample="ME",
):
    """Simulate country-level capacity factors from reanalysis data.

    Args:
        reanalysis: Reanalysis dataset with wind fields.
        turb_info: Turbine metadata with locations and capacities.
        powerCurveFile: Power curve table.
        bc_factors: Optional bias correction factors.
        time_res: Time resolution used for corrections.
        resample: Pandas resample string (e.g., "ME") or None to skip resampling.

    Returns:
        Series with simulated capacity factor values.
    """
    # >>> ADD THIS (massive speed-up) <<<
    turb_info = aggregate_turbines_to_grid(turb_info, reanalysis)

    sim_ws = interpolate_wind(reanalysis, turb_info)

    if bc_factors is not None:
        if time_res is None:
            raise ValueError("time_res must be provided when bc_factors is provided.")
        sim_ws = correct_wind_speed(sim_ws, time_res, bc_factors, turb_info)

    x, curve_by_model = _get_power_curve_cache(powerCurveFile)

    def speed_to_cf_fast(da):
        model = da.model[0].item()
        akima = curve_by_model[model]
        vals = np.clip(akima(da.data), 0.0, 1.0)
        return xr.DataArray(vals, coords=da.coords, dims=da.dims)

    sim_cf = sim_ws.groupby("model").map(speed_to_cf_fast)

    w = sim_cf["capacity"]
    country_cf = sim_cf.weighted(w).mean("turbine")

    if resample is not None:
        country_cf = country_cf.resample(time=resample).mean()

    return country_cf.to_series()


def interpolate_wind(reanalysis, turb_info, *, allow_extrapolation: bool | None = None):
    """Interpolate reanalysis wind speeds to turbine locations.

    Refuses by default when any unit lies outside the loaded ERA5 extent:
    ``xarray``'s interpolation is called with ``fill_value=None``, which would
    otherwise extrapolate linearly past the grid without a warning. Passing
    this check means the units lie inside the loaded extent. It does not verify
    the data in those cells (see :func:`loaded_extent_coverage`).

    Args:
        reanalysis: Reanalysis dataset with wind fields.
        turb_info: Turbine metadata with lon/lat/height.
        allow_extrapolation: Permit units outside the loaded extent. None (the
            default) reads the permission ``prep_era5`` attached to the
            dataset, which is False unless a run opted in.

    Returns:
        DataArray of interpolated wind speeds.

    Raises:
        ExtrapolationError: If a unit lies outside the loaded extent and
            extrapolation is not allowed.
    """
    if allow_extrapolation is None:
        allow_extrapolation = bool(reanalysis.attrs.get(EXTRAPOLATION_ATTR, False))
    coverage = loaded_extent_coverage(reanalysis, turb_info)
    if coverage["units_outside_loaded_extent"] and not allow_extrapolation:
        lon_min, lon_max, lat_min, lat_max = coverage["loaded_extent"]
        raise ExtrapolationError(
            f"{coverage['units_outside_loaded_extent']} of {coverage['units']} units "
            f"({coverage['capacity_share_outside_loaded_extent']:.1%} of capacity) lie "
            f"outside the loaded ERA5 extent (lon {lon_min} to {lon_max}, lat {lat_min} "
            f"to {lat_max}), up to {coverage['max_degrees_outside_loaded_extent']:.2f} "
            f"degrees beyond it; for example {coverage['ids_outside'][:5]}. Their winds "
            "would be extrapolated from the edge of the grid, not interpolated. Download "
            "ERA5 that covers them, or opt in with [era5] allow_extrapolation = true "
            "(allow_extrapolation=True outside the harness), which records the share "
            "and marks any scorecard result. Passing this check means only that units "
            "lie inside the loaded extent; it does not verify the data in those cells."
        )
    reanalysis = reanalysis.assign_coords(height=("height", turb_info["height"].unique()))

    EPS = 1e-6  # meters
    z0 = reanalysis["roughness"].clip(min=EPS)

    # Avoid denom = log(100/z0) ~ 0 when z0 ~ 100m (unphysical but can exist via bad values)
    denom = np.log(100.0 / z0)
    denom = denom.where(np.abs(denom) > 1e-12)

    numer = np.log(reanalysis["height"] / z0)

    ws = reanalysis["wnd100m"] * (numer / denom)

    # Coerce pandas columns to plain numpy arrays before building xarray
    # coordinates. Under pandas >= 3.0 string columns are backed by
    # ArrowStringArray, which xarray cannot use as an indexable coordinate
    # (it breaks groupby("model") and label-based indexing on the turbine dim).
    ids = np.asarray(turb_info["ID"], dtype=object)
    lat = xr.DataArray(
        np.asarray(turb_info["lat"], dtype=float), dims="turbine", coords={"turbine": ids}
    )
    lon = xr.DataArray(
        np.asarray(turb_info["lon"], dtype=float), dims="turbine", coords={"turbine": ids}
    )
    height = xr.DataArray(
        np.asarray(turb_info["height"], dtype=float), dims="turbine", coords={"turbine": ids}
    )

    # print(f"Interpolating wind speeds for {len(turb_info)} turbines (this may take a few minutes)...")
    sim_ws = ws.interp(lon=lon, lat=lat, height=height, kwargs={"fill_value": None})

    sim_ws = sim_ws.assign_coords(
        {
            "model": ("turbine", np.asarray(turb_info["model"], dtype=object)),
            "capacity": ("turbine", np.asarray(turb_info["capacity"], dtype=float)),
        }
    )
    return sim_ws


def simulate_wind(reanalysis, turb_info, powerCurveFile, *args, aggregate=False, seasons=None):
    """Simulate wind speeds and capacity factors for turbines (OPTIMIZED).

    Performance improvements:
    - Uses np.interp instead of Akima (20-100x faster)
    - Optional turbine aggregation (10-100x speedup for large datasets)
    - Pre-computes power curves once

    Args:
        reanalysis: Reanalysis dataset with wind fields.
        turb_info: Turbine metadata with lon/lat/height.
        powerCurveFile: Power curve table.
        *args: Optional (bc_factors, time_res) for correction.
        aggregate: If True, aggregate turbines to grid cells first (huge speedup).
        seasons: Optional season-name → month-list mapping forwarded to
            correct_wind_speed, for regions whose seasons differ from the
            Northern-Hemisphere defaults. Default None keeps the hardcoded
            NH season assignment.

    Returns:
        Tuple of (wind speed DataFrame, capacity factor DataFrame).

    Examples:
        >>> # Standard usage
        >>> sim_ws, sim_cf = simulate_wind(reanalysis, turb_info, power_curves)

        >>> # Fast mode for large turbine counts
        >>> sim_ws, sim_cf = simulate_wind(reanalysis, turb_info, power_curves, aggregate=True)
    """
    # OPTIMIZATION 1: Aggregate turbines to grid (10-100x speedup)
    if aggregate:
        original_count = len(turb_info)
        turb_info = aggregate_turbines_to_grid(turb_info, reanalysis)
        print(f"Aggregated {original_count} turbines to {len(turb_info)} grid points")

    sim_ws = interpolate_wind(reanalysis, turb_info)
    print("Interpolated wind speeds to turbine locations")

    if len(args) >= 1:
        bc_factors = args[0]
        time_res = args[1]
        sim_ws = correct_wind_speed(sim_ws, time_res, bc_factors, turb_info, seasons=seasons)

    # Pre-compute power curves once (not repeatedly)
    x, curve_by_model = _get_power_curve_cache(powerCurveFile)

    # Use Akima interpolation for power curve
    def speed_to_cf_fast(da):
        """Convert wind speed to capacity factor using Akima interpolation."""
        model = da.model[0].item()
        akima = curve_by_model[model]
        vals = akima(da.data)
        # vals = np.clip(akima(da.data), 0.0, 1.0)
        return xr.DataArray(vals, coords=da.coords, dims=da.dims)

    sim_cf = sim_ws.groupby("model").map(speed_to_cf_fast)

    return sim_ws.to_pandas().reset_index(), sim_cf.to_pandas().reset_index()


def correct_wind_speed(ds, time_res, bc_factors, turb_info, seasons=None):
    """Apply bias correction factors to wind speeds.

    Args:
        ds: Wind speed DataArray.
        time_res: Temporal resolution key used in corrections.
        bc_factors: Bias correction factors DataFrame.
        turb_info: Turbine metadata with cluster assignments.
        seasons: Optional season-name → month-list mapping, for regions
            whose seasons differ from the Northern-Hemisphere defaults.
            Default None keeps the hardcoded NH season assignment.

    Returns:
        DataArray of corrected wind speeds.
    """
    # robust cluster handling
    if "cluster" in turb_info.columns:
        clusters = turb_info["cluster"].to_numpy()
    else:
        clusters = np.zeros(len(turb_info), dtype=int)

    ds = ds.assign_coords({"cluster": ("turbine", clusters)})

    df = ds.to_dataframe("unc_ws").reset_index()
    df["year"] = pd.DatetimeIndex(df["time"]).year
    df["month"] = pd.DatetimeIndex(df["time"]).month

    df = add_time_resolution_columns(df, seasons)

    df = df.merge(bc_factors, on=["cluster", time_res], how="left").set_index(["time", "turbine"])

    # to_xarray() builds each axis from the sorted index levels, so the turbine
    # axis comes back in ID order rather than turb_info's order. The model and
    # capacity coordinates below are attached by position, so restore the input
    # order first; otherwise a fleet whose IDs are not already sorted gets other
    # units' power curves and capacities in the corrected simulation.
    ds2 = (
        df[["scalar", "offset", "unc_ws"]]
        .to_xarray()
        .reindex(time=ds["time"].values, turbine=ds["turbine"].values)
    )
    ds2 = ds2.assign(cor_ws=(ds2["unc_ws"] * ds2["scalar"]) + ds2["offset"])

    # model coord for downstream mapping (coerce to numpy so the coordinate is
    # not a pandas ArrowStringArray, which breaks groupby("model") on pandas>=3)
    ds2 = ds2.assign_coords({"model": ("turbine", np.asarray(turb_info["model"], dtype=object))})
    ds2 = ds2.assign_coords(
        {"capacity": ("turbine", np.asarray(turb_info["capacity"], dtype=float))}
    )

    return ds2.cor_ws


def prepare_offset_arrays(unc_ws, powerCurveFile):
    """Pre-extract numpy arrays from xarray for fast offset optimization.

    Call this once before the iterative offset search to avoid repeated
    xarray overhead.

    Args:
        unc_ws: Interpolated wind speed DataArray (time x turbine).
        powerCurveFile: Power curve lookup table.

    Returns:
        dict with keys: ws_data, model_groups, capacities, total_weighted.
    """
    ws_data = unc_ws.values  # (n_time, n_turbine)
    models = unc_ws.model.values
    capacities = unc_ws.capacity.values.astype(float)

    _, curve_by_model = _get_power_curve_cache(powerCurveFile)

    # Pre-group turbine indices by model
    unique_models = np.unique(models)
    model_groups = []
    for m in unique_models:
        mask = models == m
        model_groups.append((curve_by_model[m], mask))

    return {
        "ws_data": ws_data,
        "model_groups": model_groups,
        "capacities": capacities,
    }


def fast_simulate_cf(arrays, scalar, offset):
    """Compute capacity-weighted mean CF using pure numpy.

    Args:
        arrays: Dict from prepare_offset_arrays.
        scalar: Multiplicative correction.
        offset: Additive correction.

    Returns:
        float: Capacity-weighted mean capacity factor.
    """
    ws = arrays["ws_data"]
    cor_ws = ws * scalar + offset

    # Allocate CF array same shape as wind speeds
    cf = np.empty_like(cor_ws)

    for akima, mask in arrays["model_groups"]:
        cf[:, mask] = akima(cor_ws[:, mask])

    # Capacity-weighted mean across all turbines and timesteps
    capacities = arrays["capacities"]
    weighted_cf = np.nanmean(cf, axis=0)  # mean over time per turbine

    # Mask out turbines with all-NaN CF (matches xarray weighted().mean() NaN-skipping)
    valid = ~np.isnan(weighted_cf)
    if not valid.any():
        return np.nan
    valid_cap = capacities[valid]
    total_cap = valid_cap.sum()
    if total_cap == 0:
        return np.nan
    return np.dot(weighted_cf[valid], valid_cap) / total_cap


def train_simulate_wind_from_ws(unc_ws, powerCurveFile, scalar=1, offset=0):
    """Simulate a mean capacity factor from pre-interpolated wind speeds.

    The fast inner step of the offset optimisation: applies the affine
    correction ``ws * scalar + offset``, converts to capacity factor through
    each turbine's power curve, and reduces to one capacity-weighted mean.
    Unlike :func:`train_simulate_wind` it skips hub-height extrapolation and
    spatial interpolation, so an optimiser can call it repeatedly on wind
    speeds prepared once.

    Args:
        unc_ws: Uncorrected hub-height wind speeds as an xarray DataArray
            with a ``turbine`` dimension carrying ``model`` and ``capacity``
            coordinates (the shape produced by :func:`interpolate_wind`).
        powerCurveFile: Power curve table (``data$speed`` plus one column per
            model).
        scalar: Multiplicative wind-speed correction.
        offset: Additive wind-speed correction, in m/s.

    Returns:
        The capacity-weighted mean capacity factor, as a scalar.
    """
    cor_ws = (unc_ws * scalar) + offset
    x, curve_by_model = _get_power_curve_cache(powerCurveFile)

    def speed_to_cf_fast(data):
        """Convert wind speed to capacity factor using Akima interpolation."""
        model = data.model[0].item()
        akima = curve_by_model[model]
        vals = akima(data.data)
        return xr.DataArray(vals, coords=data.coords, dims=data.dims)

    cor_cf = cor_ws.groupby("model").map(speed_to_cf_fast)
    avg_cf = cor_cf.weighted(cor_cf["capacity"]).mean()
    return avg_cf.data


def train_simulate_wind(reanalysis, turb_info, powerCurveFile, scalar=1, offset=0):
    """Simulate a mean capacity factor for training.

    Args:
        reanalysis: Wind parameters on a grid.
        turb_info: Turbine metadata including height and coordinates.
        powerCurveFile: Power curve data for turbine models.
        scalar: Multiplicative correction factor.
        offset: Additive correction factor.

    Returns:
        float: Weighted average of simulated capacity factor.
    """
    unc_ws = interpolate_wind(reanalysis, turb_info)
    return train_simulate_wind_from_ws(unc_ws, powerCurveFile, scalar, offset)
