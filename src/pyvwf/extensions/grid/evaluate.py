"""Score a gridded correction where the chapter scores it: at observations.

Ported on 2026-09-13 from
``development:scripts/pyvwf_to_grid/evaluate_grid_corrections.py``. This is the
half of the chapter's validation that turns a correction surface into a number
a reviewer recognises: extract the correction at each unit's location, apply it
to wind speed, convert through the power curve, and compare with observations.

Both registered studies state their gates in capacity-factor error at
observation locations, which is why they wait for this rather than substituting
a surface-level proxy: a proxy could not contradict the chapter's own offshore
failure case, which is stated in those terms
(``docs/findings/method-offshore-pool-prereg.md``,
``docs/findings/method-grid-nl-holdout-prereg.md``).

**Neutral fills are counted and returned.** A unit outside the grid, or inside
a masked region, receives scalar 1 and offset 0, which is the correction
declining to answer. The original substituted those silently, so a run in which
most units got no correction at all scored as an ordinary result. The share is
reported beside every metric, because for a holdout it is the first thing to
look at: it is the difference between a correction that failed and a correction
that was never applied.

**Failures raise.** The original wrapped its turbine-level metric in a bare
``except Exception`` that printed and returned NaN, so a broken run produced a
row of missing values in a results table rather than stopping.

One prose and code difference is reproduced rather than resolved: the chapter
says corrected wind speeds are "clipped to physically admissible bounds before
conversion", and the code clips the resulting capacity factor to 0 and 1
instead, leaving the speed unclipped. That is recorded in the maintainer's
manuscript note, kept untracked, at
``https://github.com/ellyess/PyVWF/blob/ce20ead0716ffcbe2ad132d616f14c288cbc679d/docs/design/manuscript-chapters-45.md``
(T0, row 3); the behaviour here is the code's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from pyvwf.metrics import calculate_error, weighted_mean
from pyvwf.wind import _get_power_curve_cache

#: What a unit gets where the surface declines to answer.
NEUTRAL_SCALAR, NEUTRAL_OFFSET = 1.0, 0.0


def _axes(grid: xr.Dataset) -> tuple[str, str]:
    """The surface's longitude and latitude axis names, either convention."""
    for lon, lat in (("x", "y"), ("lon", "lat")):
        if lon in grid.dims and lat in grid.dims:
            return lon, lat
    raise KeyError(f"surface has dims {list(grid.dims)}; expected x and y, or lon and lat")


def corrections_at(grid: xr.Dataset, units: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The correction each unit receives, and how many received none.

    Bilinear from the surface to each unit's position. A unit off the grid or
    in a masked region takes the neutral values.

    Args:
        grid: a correction surface with ``scalar`` and ``offset``.
        units: fleet or grid points with ``ID``, ``lon`` and ``lat``.

    Returns:
        The per-unit corrections, and a summary carrying ``n_units``,
        ``n_neutral`` and ``neutral_share``. **The summary is not optional
        output**: a holdout that silently neutralised most of its units would
        otherwise score as an ordinary result.
    """
    for column in ("ID", "lon", "lat"):
        if column not in units.columns:
            raise ValueError(f"units are missing {column!r}")
    lon_name, lat_name = _axes(grid)
    at = {
        lon_name: xr.DataArray(units["lon"].to_numpy(float), dims="points"),
        lat_name: xr.DataArray(units["lat"].to_numpy(float), dims="points"),
    }
    # coords= rather than **at: xarray accepts both, and the keyword form has
    # mypy resolve the axis names against interp's own parameters.
    scalar = np.asarray(grid["scalar"].interp(coords=at, method="linear").values)
    offset = np.asarray(grid["offset"].interp(coords=at, method="linear").values)

    missing = np.isnan(scalar) | np.isnan(offset)
    neutral = missing | ((scalar == NEUTRAL_SCALAR) & (offset == NEUTRAL_OFFSET))
    frame = pd.DataFrame(
        {
            "ID": units["ID"].astype(str).to_numpy(),
            "scalar": np.where(missing, NEUTRAL_SCALAR, scalar),
            "offset": np.where(missing, NEUTRAL_OFFSET, offset),
            "neutral": neutral,
        }
    )
    summary = {
        "n_units": int(len(frame)),
        "n_neutral": int(neutral.sum()),
        "neutral_share": float(neutral.mean()) if len(frame) else float("nan"),
        "n_off_grid": int(missing.sum()),
    }
    return frame, summary


def corrected_capacity_factors(
    uncorrected_speed: xr.DataArray, corrections: pd.DataFrame, power_curves: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Apply the correction at wind-speed level and convert through the curves.

    Args:
        uncorrected_speed: speeds by time and turbine, carrying a ``model``
            coordinate, as ``pyvwf.wind.interpolate_wind`` returns.
        corrections: from :func:`corrections_at`.
        power_curves: the curve table.

    Returns:
        Capacity factors, wide, one column per unit and a ``time`` column, and
        a summary carrying ``n_off_curve`` and ``off_curve_share``. **A
        corrected speed past the ends of the curve table has no capacity factor
        and comes back missing, not clipped**, and a missing value then drops
        out of the metrics without trace. The chapter's text says speeds are
        "clipped to physically admissible bounds before conversion" and the
        code clips the capacity factor instead, leaving the speed unclipped;
        the behaviour here is the code's and the count is what makes it
        visible.

    Raises:
        ValueError: if a unit in the speeds has no correction. The original
            raised a bare KeyError from inside a list comprehension, which said
            nothing about which side was short.
    """
    ids = [str(i) for i in uncorrected_speed.turbine.values]
    lookup = corrections.drop_duplicates("ID").set_index("ID")
    absent = [i for i in ids if i not in lookup.index]
    if absent:
        raise ValueError(
            f"{len(absent)} of {len(ids)} units in the wind speeds have no correction, "
            f"for example {absent[:5]}. The two sides key on different units."
        )

    coords = {"turbine": uncorrected_speed.turbine.values}
    scalars = xr.DataArray(lookup.loc[ids, "scalar"].to_numpy(float), dims="turbine", coords=coords)
    offsets = xr.DataArray(lookup.loc[ids, "offset"].to_numpy(float), dims="turbine", coords=coords)
    corrected = uncorrected_speed * scalars + offsets

    _, curve_by_model = _get_power_curve_cache(power_curves)

    def to_cf(block: xr.DataArray) -> xr.DataArray:
        curve = curve_by_model[block.model[0].item()]
        return xr.DataArray(
            np.clip(curve(block.data), 0.0, 1.0), coords=block.coords, dims=block.dims
        )

    frame = corrected.groupby("model").map(to_cf).to_pandas()
    assert isinstance(frame, pd.DataFrame)  # time by turbine, so never a Series
    wide = frame.reset_index()
    wide.columns = [str(c) for c in wide.columns]
    values = wide.drop(columns=["time"]).to_numpy(dtype=float)
    summary = {
        "n_off_curve": int(np.isnan(values).sum()),
        "off_curve_share": float(np.isnan(values).mean()) if values.size else float("nan"),
    }
    return wide, summary


def turbine_skill(simulated: pd.DataFrame, observed: pd.DataFrame, units: pd.DataFrame) -> dict:
    """Capacity-weighted per-unit error, by the pipeline's own definition.

    Delegates to :func:`pyvwf.metrics.calculate_error`, so a gridded correction
    is scored the way every other correction in this project is scored. The
    original wrapped the same call in a bare ``except Exception`` that returned
    missing values; a failure here raises.
    """
    rmse, mae, mbe = calculate_error("total", simulated.copy(), observed.copy(), units.copy())
    return {"mae": float(mae), "rmse": float(rmse), "bias": float(mbe)}


def country_skill(simulated: pd.DataFrame, observed: pd.DataFrame, units: pd.DataFrame) -> dict:
    """Capacity-weighted national monthly error.

    Grid points are aggregated to one national series by capacity, monthly, and
    compared with the national observation. A point with no simulated value is
    left out of both the numerator and the weight for that step, so the
    aggregate is over the points that reported.

    Raises:
        ValueError: if no grid point carries a capacity, or if no month pairs,
            either of which the original returned as missing values.
    """
    sim, obs = simulated.copy(), observed.copy()
    sim["time"] = pd.to_datetime(sim["time"])
    obs["time"] = pd.to_datetime(obs["time"])

    capacity = units.assign(ID=units["ID"].astype(str)).set_index("ID")["capacity"]
    columns = [c for c in sim.columns if c != "time" and str(c) in capacity.index]
    if not columns:
        raise ValueError(
            "no simulated grid point matches a unit with a capacity; the two sides key "
            "on different identifiers"
        )

    weights = capacity[[str(c) for c in columns]].to_numpy(float)
    values = sim[columns].to_numpy(float)
    sim["cf_sim"] = weighted_mean(values, weights, axis=1)

    monthly = sim.assign(ym=sim["time"].dt.to_period("M")).groupby("ym")["cf_sim"].mean()
    observed_monthly = obs.assign(ym=obs["time"].dt.to_period("M")).groupby("ym")["obs"].mean()
    paired = pd.concat([monthly, observed_monthly.rename("cf_obs")], axis=1).dropna()
    if paired.empty:
        raise ValueError("no month has both a simulated and an observed value")

    difference = paired["cf_sim"] - paired["cf_obs"]
    return {
        "mae": float(difference.abs().mean()),
        "rmse": float(np.sqrt((difference**2).mean())),
        "bias": float(difference.mean()),
        "n_months": int(len(paired)),
    }


def skill(
    simulated: pd.DataFrame, observed: pd.DataFrame, units: pd.DataFrame, obs_level: str
) -> dict:
    """Whichever skill the observation level calls for."""
    if obs_level == "turbine":
        return turbine_skill(simulated, observed, units)
    if obs_level == "country":
        return country_skill(simulated, observed, units)
    raise ValueError(f"unknown obs_level {obs_level!r}; use 'turbine' or 'country'")
