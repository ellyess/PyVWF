"""
wind module.

Summary
-------
Interpolation and simulation of wind speeds and power at turbine locations.

Data conventions
----------------
Expected dimensions follow xarray conventions (e.g., time × lat × lon) unless stated otherwise.
Time coordinates are assumed to be UTC unless explicitly converted by the caller.

Units
-----
Wind speed: [m s^-1]; Hub height: [m]; Power: [MW]; Energy: [MWh]; Capacity factor: [-] (unless stated otherwise).

Assumptions
-----------
- ERA5/reanalysis fields are treated as representative at the chosen spatial/temporal resolution.
- Wake effects, curtailment, availability losses are not modelled unless explicitly implemented in this module.

References
----------
Add dataset and methodological references relevant to this module.
"""
import xarray as xr
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
from scipy.interpolate import Akima1DInterpolator

import xarray as xr
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate

# --- local helper to avoid circular import with vwf.data ---
def _add_time_res_local(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df["month"] == 1, ["bimonth", "season"]] = ["1/6", "winter"]
    df.loc[df["month"] == 2, ["bimonth", "season"]] = ["1/6", "winter"]
    df.loc[df["month"] == 3, ["bimonth", "season"]] = ["2/6", "spring"]
    df.loc[df["month"] == 4, ["bimonth", "season"]] = ["2/6", "spring"]
    df.loc[df["month"] == 5, ["bimonth", "season"]] = ["3/6", "spring"]
    df.loc[df["month"] == 6, ["bimonth", "season"]] = ["3/6", "summer"]
    df.loc[df["month"] == 7, ["bimonth", "season"]] = ["4/6", "summer"]
    df.loc[df["month"] == 8, ["bimonth", "season"]] = ["4/6", "summer"]
    df.loc[df["month"] == 9, ["bimonth", "season"]] = ["5/6", "autumn"]
    df.loc[df["month"] == 10, ["bimonth", "season"]] = ["5/6", "autumn"]
    df.loc[df["month"] == 11, ["bimonth", "season"]] = ["6/6", "autumn"]
    df.loc[df["month"] == 12, ["bimonth", "season"]] = ["6/6", "winter"]
    df["fixed"] = "1/1"
    return df

def aggregate_turbines_to_grid(turb_info: pd.DataFrame, reanalysis) -> pd.DataFrame:
    """
    Collapse turbines onto nearest reanalysis grid cell to massively reduce interpolation cost.

    Returns a turb_info-like dataframe with one row per (lat_cell, lon_cell, height_bin/model)
    and capacity summed. Keeps required cols: ID, lat, lon, height, capacity, model.
    """
    ti = turb_info.copy()

    # Ensure numeric
    for c in ["lat", "lon", "height", "capacity"]:
        ti[c] = pd.to_numeric(ti[c], errors="coerce")
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
    # >>> ADD THIS (massive speed-up) <<<
    turb_info = aggregate_turbines_to_grid(turb_info, reanalysis)

    sim_ws = interpolate_wind(reanalysis, turb_info)

    if bc_factors is not None:
        if time_res is None:
            raise ValueError("time_res must be provided when bc_factors is provided.")
        sim_ws = correct_wind_speed(sim_ws, time_res, bc_factors, turb_info)

    x = powerCurveFile["data$speed"].to_numpy()
    curve_by_model = {
        m: powerCurveFile[m].to_numpy()
        for m in powerCurveFile.columns
        if m != "data$speed"
    }

    def speed_to_cf_fast(da):
        model = da.model[0].item()
        y = curve_by_model[model]
        vals = np.interp(da.data, x, y, left=0.0, right=1.0)
        return xr.DataArray(vals, coords=da.coords, dims=da.dims)

    sim_cf = sim_ws.groupby("model").map(speed_to_cf_fast)

    w = sim_cf["capacity"]
    country_cf = sim_cf.weighted(w).mean("turbine")

    if resample is not None:
        country_cf = country_cf.resample(time=resample).mean()

    return country_cf.to_series()


def interpolate_wind(reanalysis, turb_info):
    reanalysis = reanalysis.assign_coords(height=("height", turb_info["height"].unique()))

    EPS = 1e-6  # meters
    z0 = reanalysis["roughness"].clip(min=EPS)

    # Avoid denom = log(100/z0) ~ 0 when z0 ~ 100m (unphysical but can exist via bad values)
    denom = np.log(100.0 / z0)
    denom = denom.where(np.abs(denom) > 1e-12)

    numer = np.log(reanalysis["height"] / z0)

    ws = reanalysis["wnd100m"] * (numer / denom)

    lat = xr.DataArray(turb_info["lat"], dims="turbine", coords={"turbine": turb_info["ID"]})
    lon = xr.DataArray(turb_info["lon"], dims="turbine", coords={"turbine": turb_info["ID"]})
    height = xr.DataArray(turb_info["height"], dims="turbine", coords={"turbine": turb_info["ID"]})

    sim_ws = ws.interp(lon=lon, lat=lat, height=height, kwargs={"fill_value": None})

    sim_ws = sim_ws.assign_coords(
        {
            "model": ("turbine", turb_info["model"]),
            "capacity": ("turbine", turb_info["capacity"]),
        }
    )
    return sim_ws


def simulate_wind(reanalysis, turb_info, powerCurveFile, *args):
    sim_ws = interpolate_wind(reanalysis, turb_info)
    print("Interpolated wind speeds to turbine locations")

    if len(args) >= 1:
        bc_factors = args[0]
        time_res = args[1]
        sim_ws = correct_wind_speed(sim_ws, time_res, bc_factors, turb_info)

    def speed_to_power(data):
        x = powerCurveFile["data$speed"]
        y = powerCurveFile[data.model[0].data]
        f = interpolate.Akima1DInterpolator(x, y)
        return f(data)

    sim_cf = sim_ws.groupby("model").map(speed_to_power)

    return sim_ws.to_pandas().reset_index(), sim_cf.to_pandas().reset_index()


def correct_wind_speed(ds, time_res, bc_factors, turb_info):
    # robust cluster handling
    if "cluster" in turb_info.columns:
        clusters = turb_info["cluster"].to_numpy()
    else:
        clusters = np.zeros(len(turb_info), dtype=int)

    ds = ds.assign_coords({"cluster": ("turbine", clusters)})

    df = ds.to_dataframe("unc_ws").reset_index()
    df["year"] = pd.DatetimeIndex(df["time"]).year
    df["month"] = pd.DatetimeIndex(df["time"]).month

    df = _add_time_res_local(df)

    df = df.merge(bc_factors, on=["cluster", time_res], how="left").set_index(["time", "turbine"])

    ds2 = df[["scalar", "offset", "unc_ws"]].to_xarray()
    ds2 = ds2.assign(cor_ws=(ds2["unc_ws"] * ds2["scalar"]) + ds2["offset"])

    # model coord for downstream mapping
    ds2 = ds2.assign_coords({"model": ("turbine", turb_info["model"])})
    ds2 = ds2.assign_coords({"capacity": ("turbine", turb_info["capacity"])})

    return ds2.cor_ws

def train_simulate_wind(reanalysis, turb_info, powerCurveFile, scalar=1, offset=0):
    """
    Simulate average capacity factor of desired resolution for training.

        Args:
            reanalysis (xarray.Dataset): wind parameters on a grid
            turb_info (pandas.DataFrame): turbine metadata including height and coordinates
            powerCurveFile (pandas.DataFrame): capacity factor at increasing wind speeds for different models
            scalar (float): multiplicative correction factor
            offset (float): additive correction factor

        Returns:
            float: weighted average of simulated CF
    """ 
    unc_ws = interpolate_wind(reanalysis, turb_info)
    cor_ws = (unc_ws * scalar) + offset
    
    def speed_to_power(data):
        """
        Speed to power.

            Args:
                data (pandas.DataFrame): data containing simulated wind speeds.

            Assumptions:
                - Datetime handling is assumed to be UTC unless stated otherwise.
                - Units are assumed to be consistent with SI conventions unless stated otherwise.
                - Power curves are assumed to be static and representative (no wake, curtailment, or availability losses unless explicitly modelled).
                - Capacity factor is assumed to be bounded in [0, 1].
        """
        x = powerCurveFile['data$speed']
        y = powerCurveFile[data.model[0].data]
        f = interpolate.Akima1DInterpolator(x, y)
        return f(data)
    
    cor_cf = cor_ws.groupby('model').map(speed_to_power)
    avg_cf = cor_cf.weighted(cor_cf['capacity']).mean()
    return avg_cf.data
