"""
data module.

Summary
-------
Preprocessing of the data required in the model.

Data conventions
----------------
Tabular inputs are assumed to be tidy (one observation per row) unless stated otherwise.
Datetime columns are assumed to be timezone-naive UTC unless specified.

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
import numpy as np
import pandas as pd
import geopandas as gpd
import difflib

import utm
from calendar import monthrange

import vwf.wind as wind
from vwf.datasets.era5 import prep_era5
# from vwf.datasets.era5 import prep_era5_daily_cached
from vwf.clustering import cluster_turbines
import vwf.correction as correction

from vwf.wind import simulate_country_cf


# -----------------------------------------------------------------------------
# New helpers for country-level observations
# -----------------------------------------------------------------------------
def _hours_in_month(year: int, month: int) -> int:
    return monthrange(int(year), int(month))[1] * 24

def _default_power_curve(power_curves: pd.DataFrame) -> str:
    """
    Pick a safe default turbine model from power_curves.csv.
    """
    cols = [c for c in power_curves.columns if c != "data$speed"]
    if not cols:
        raise ValueError("power_curves has no turbine model columns.")
    return cols[0]


def country_gen_to_cf(
    obs_country_gen: pd.DataFrame,
    turb_info: pd.DataFrame,
    *,
    output_col: str = "output_kwh",
    capacity_unit: str = "kW",  # "kW" or "MW"
) -> pd.DataFrame:
    """
    Convert country-level monthly generation to CF using total capacity from turb_info.

    obs_country_gen must have columns: ['year','month', output_col]
    turb_info must have column: ['capacity'] in capacity_unit.

    Returns tidy df: ['year','month','obs'] where obs is CF [-].
    """
    if "capacity" not in turb_info.columns:
        raise ValueError("turb_info must contain a 'capacity' column.")

    cap = pd.to_numeric(turb_info["capacity"], errors="coerce")

    if capacity_unit.lower() == "mw":
        cap_kw = float(cap.sum() * 1000.0)
    else:
        cap_kw = float(cap.sum())

    if not np.isfinite(cap_kw) or cap_kw <= 0:
        raise ValueError(
            "Total capacity from turb_info is not valid for converting country gen to CF. "
            f"(sum capacity={cap.sum()}, unit={capacity_unit})"
        )

    df = obs_country_gen.copy()
    if not {"year", "month", output_col}.issubset(df.columns):
        raise ValueError(f"country_gen_to_cf expects columns ['year','month','{output_col}'].")

    df["hours"] = df.apply(lambda r: _hours_in_month(r["year"], r["month"]), axis=1)
    df["obs"] = pd.to_numeric(df[output_col], errors="coerce") / (cap_kw * df["hours"].astype(float))
    return df[["year", "month", "obs"]]


def sim_turbines_to_country_cf(sim_cf_long: pd.DataFrame, turb_info: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate long-form turbine simulations to a country-level series.

    sim_cf_long must have columns: ['year','month','ID','sim'] where sim is CF [-]
    turb_info must have columns: ['ID','capacity'] (kW)
    Returns tidy df: ['year','month','sim'] where sim is capacity-weighted mean CF.
    """
    df = sim_cf_long.copy()
    df["ID"] = df["ID"].astype(str)
    caps = turb_info[["ID", "capacity"]].copy()
    caps["ID"] = caps["ID"].astype(str)
    df = df.merge(caps, on="ID", how="left")

    df = df.dropna(subset=["capacity", "sim"])
    if df.empty:
        raise ValueError("No valid rows to aggregate in sim_turbines_to_country_cf.")

    sim_country = (
        df.groupby(["year", "month"], as_index=False)
          .apply(lambda g: pd.Series({"sim": (g["sim"] * g["capacity"]).sum() / g["capacity"].sum()}))
          .reset_index(drop=True)
    )
    return sim_country


# -----------------------------------------------------------------------------
# Existing functions (mostly unchanged)
# -----------------------------------------------------------------------------
def clean_obs_data(df, country, train=False):
    """
    Preprocess the turbines/farms found in the observed data.

    Prepares the observational data (obs_cf and turb_info) for the desired country
    used in the model. Converts observed power generation into observed CF and ensures
    that all turbines in the data are acceptable.

        Args:
            country (str): country code e.g. Denmark "DK"

        Returns:
            df (pandas.DataFrame): cleaned observational dataframe
    """
    # cf can't be greater than 100%
    df["cf_max"] = df[df.columns[df.columns.str.startswith("obs")]].max(axis=1)
    df = df.drop(df[df["cf_max"] > 1].index)
    df = df.drop("cf_max", axis=1)

    # case exists for Denmark solely, develop a method to consider the weight of missing data
    # remove any turbines that have cf of 0 at any point
    if (train) & (country == "DK"):
        df["cf_min"] = df[df.columns[df.columns.str.startswith("obs")]].min(axis=1)
        df = df.drop(df[df["cf_min"] <= 0.01].index)
        df = df.drop("cf_min", axis=1)

    # turn 0 into nan to not be considered in groupby functions
    df = df.replace(0, np.nan)

    # turbine should atleast have a cf of atleast 1%
    df["cf_mean"] = df[df.columns[df.columns.str.startswith("obs")]].mean(axis=1)
    df = df.drop(df[df["cf_mean"] <= 0.01].index)
    df = df.drop(["cf_mean"], axis=1)

    return df


def load_power_curves():
    """
    Load power curves.
    """
    file_loc = "input/power_curves.csv"
    df = pd.read_csv(file_loc)
    return df


def train_set(
    country,
    calc_z0,
    mode="all",
    year_test=None,
    add_nan=None,
    interp_nan=None,
    fix_turb=None,
    *,
    obs_level: str = "turbine",
):
    obs_data, turb_info = prep_country(country, year_test, obs_level=obs_level)

    if mode != "all":
        turb_info = turb_info[turb_info["type"] == mode].copy()

    if fix_turb is not None:
        turb_info["model"] = fix_turb

    # prep era5 + curves once
    reanalysis = prep_era5(country, True, calc_z0)
    # BBOX = {
    #     "UK": (-11, 3, 49, 61),
    #     "NL": (  2.5, 7.5, 50.5, 53.8),
    #     "BE": (  2.2, 6.6, 49.3, 51.8),
    #     "DE": (  5.0, 16.0, 47.0, 55.2),
    #     "DK": (  7.5, 13.5, 54.0, 58.2),
    #     "NO": (  4.0, 31.5, 57.5, 71.5),
    #     "FR": ( -6.0, 10.5, 41.0, 51.5),
    #     }
    # reanalysis = prep_era5_daily_cached(country, train=True, calc_z0=calc_z0, bbox=BBOX[country])
    power_curves = load_power_curves()

    # -------------------------
    # FAST country-level branch
    # -------------------------
    if obs_level == "country":
        obs_country = country_gen_to_cf(obs_data, turb_info, output_col="output_kwh")

        # -------------------------------------------------
        # Ensure a valid power-curve model exists
        # -------------------------------------------------
        if "model" not in turb_info.columns or turb_info["model"].isna().all():
            default_model = (
                fix_turb
                if fix_turb is not None
                else _default_power_curve(power_curves)
            )
            turb_info = turb_info.copy()
            turb_info["model"] = default_model

        # -------------------------------------------------
        # Ensure numeric fields required by interpolate_wind
        # -------------------------------------------------
        turb_info["capacity"] = pd.to_numeric(turb_info["capacity"], errors="coerce")
        turb_info["height"] = pd.to_numeric(turb_info["height"], errors="coerce")
        turb_info["lon"] = pd.to_numeric(turb_info["lon"], errors="coerce")
        turb_info["lat"] = pd.to_numeric(turb_info["lat"], errors="coerce")

        turb_info = turb_info.dropna(
            subset=["capacity", "height", "lon", "lat", "model"]
        ).reset_index(drop=True)

        sim_country_cf = simulate_country_cf(
            reanalysis,
            turb_info,
            power_curves,
            resample="ME",
        )
        
        sim_country = sim_country_cf.reset_index()

        # robustly detect the datetime column (index column)
        time_col = None
        for c in sim_country.columns:
            if pd.api.types.is_datetime64_any_dtype(sim_country[c]):
                time_col = c
                break

        # if not found, assume first column is the former index
        if time_col is None:
            time_col = sim_country.columns[0]
            sim_country[time_col] = pd.to_datetime(sim_country[time_col], errors="coerce")

        # name it consistently
        sim_country = sim_country.rename(columns={time_col: "time"})
        sim_country = sim_country.rename(columns={sim_country.columns[1]: "sim"})  # second col is values

        sim_country["year"] = sim_country["time"].dt.year.astype(int)
        sim_country["month"] = sim_country["time"].dt.month.astype(int)
        sim_country = sim_country[["year", "month", "sim"]]

        gen_cf = sim_country.merge(obs_country, on=["year", "month"], how="inner")
        gen_cf = add_time_res(gen_cf)

        return gen_cf.reset_index(drop=True), turb_info, reanalysis, power_curves

    # ---------------------------------
    # Existing turbine-level branch
    # ---------------------------------
    # simulate turbines (only for turbine obs)
    sim_ws, sim_cf = wind.simulate_wind(reanalysis, turb_info, power_curves)

    sim_cf = sim_cf.groupby(pd.Grouper(key="time", freq="ME")).mean().reset_index()
    sim_cf = sim_cf.melt(id_vars=["time"], var_name="ID", value_name="sim")
    sim_cf = add_times(sim_cf)
    sim_cf = add_time_res(sim_cf)
    sim_cf["ID"] = sim_cf["ID"].astype(str)

    obs_cf = obs_data
    obs_cf = clean_obs_data(obs_cf, country, True)

    year_star = obs_cf.year.min()
    year_end = obs_cf.year.max()

    obs_cf = obs_cf[obs_cf.groupby("ID").ID.transform("count") == ((year_end - year_star) + 1)].reset_index(drop=True)

    obs_cf = obs_cf[
        ["ID", "year",
         "obs_1","obs_2","obs_3","obs_4","obs_5","obs_6",
         "obs_7","obs_8","obs_9","obs_10","obs_11","obs_12"]
    ]
    obs_cf.columns = ["ID","year","1","2","3","4","5","6","7","8","9","10","11","12"]

    obs_cf = obs_cf.loc[obs_cf["ID"].isin(turb_info["ID"])].reset_index(drop=True)

    obs_cf = obs_cf.melt(id_vars=["ID", "year"], var_name="month", value_name="obs")
    obs_cf["month"] = obs_cf["month"].astype(int)
    obs_cf["year"] = obs_cf["year"].astype(int)

    if add_nan is not None:
        obs_cf["obs"] = obs_cf["obs"].sample(frac=(1 - add_nan), random_state=42)

    if interp_nan is not None:
        obs_cf = interp_nans(obs_cf, interp_nan)

    turb_info = turb_info.loc[turb_info["ID"].isin(obs_cf["ID"])].reset_index(drop=True)

    obs_cf["ID"] = obs_cf["ID"].astype(str)

    gen_cf = pd.merge(sim_cf, obs_cf, on=["ID", "month", "year"], how="left")
    gen_cf = gen_cf.drop(["time"], axis=1).reset_index(drop=True)

    return gen_cf, turb_info, reanalysis, power_curves


def val_set(country, calc_z0, mode="all", year_test=None, fix_turb=None, *, obs_level: str = "turbine"):
    """
    Validation set preparation.

    For obs_level="turbine": unchanged (returns obs_cf wide->long time-indexed).
    For obs_level="country": returns tidy monthly country CF series in a time-indexed frame.
    """
    obs_data, turb_info = prep_country(country, year_test, obs_level=obs_level)

    if mode != "all":
        turb_info = turb_info[turb_info["type"] == mode].copy()

    if fix_turb is not None:
        turb_info["model"] = fix_turb

    # preping era5 for val
    reanalysis = prep_era5(country, False, calc_z0)
    power_curves = load_power_curves()

    if obs_level == "country":
        # obs_data: ['year','month','output_kwh'] for year_test only (because year_test passed)
        obs_country = country_gen_to_cf(obs_data, turb_info, output_col="output_kwh", capacity_unit="kW")
        # build a time index similar to turbine val output
        obs_country["time"] = pd.to_datetime(dict(year=obs_country["year"], month=obs_country["month"], day=1))
        obs_country = obs_country.sort_values("time")[["time", "obs"]]
        return obs_country, turb_info, reanalysis, power_curves

    # turbine-level path
    obs_cf = obs_data
    obs_cf = clean_obs_data(obs_cf, country, False)

    # formatting for testing
    dates = np.arange(str(year_test) + "-01", str(year_test + 1) + "-01", dtype="datetime64[M]")
    cols = dates.tolist()
    obs_cf = obs_cf.drop("year", axis=1)
    obs_cf.columns = ["ID"] + cols
    obs_cf = obs_cf.loc[obs_cf["ID"].isin(turb_info["ID"])].reset_index(drop=True)
    turb_info = turb_info.loc[turb_info["ID"].isin(obs_cf["ID"])].reset_index(drop=True)
    obs_cf = obs_cf.set_index("ID").transpose().rename_axis("time").reset_index()

    return obs_cf, turb_info, reanalysis, power_curves


def cluster_train_set(gen_cf, time_res, num_clu, turb_info, *, obs_level: str = "turbine"):
    """
    Applying desired resolution to train set and calculating scalar.

    For obs_level="country", you only have one observation series, so you cannot
    fit per-cluster scalars. We return a single cluster=0 and attach all turbines to it.
    """
    if obs_level == "country":
        # time average to desired temporal resolution
        df = gen_cf.groupby(["year", time_res], as_index=False)[["obs", "sim"]].mean()

        # compute scalar (and keep an offset column for compatibility)
        df["cluster"] = 0
        df["scalar"] = df["obs"] / df["sim"]
        df["offset"] = 0.0

        # keep same column naming convention as downstream expects
        # (your format_bc_factors expects columns: year, time_res, cluster, scalar, offset after dropping obs/sim)
        df = df[["year", time_res, "cluster", "obs", "sim", "scalar", "offset"]]

        clus_info = turb_info.copy()
        clus_info["cluster"] = 0

        return df, clus_info

    # turbine-level existing behavior
    gen_cf = gen_cf.groupby(["year", time_res, "ID"], as_index=False)[["obs", "sim"]].mean()

    clus_info = cluster_turbines(num_clu, turb_info, True)
    gen_cf = pd.merge(
        gen_cf,
        clus_info[["ID", "cluster", "lon", "lat", "capacity", "height", "model"]],
        on="ID",
        how="left",
    )

    train_bias_df = correction.calculate_scalar(gen_cf, time_res)

    return train_bias_df, clus_info


def interp_nans(df, limit):
    """
    Interpolate nans in observed data.
    """
    df = (
        df.sort_values(["ID", "year"])
        .groupby(["ID"])
        .apply(
            lambda group: group.interpolate(
                method="linear",
                limit=limit,
                limit_direction="both",
            )
        )
        .reset_index(drop=True)
        .sort_values(["year", "month"])
    )
    return df


def add_models(df):
    """
    Assign model names to input turbines.
    """
    models = pd.read_csv("input/models.csv")
    models["model"] = models["model"].astype(pd.StringDtype())
    models["manufacturer"] = models["manufacturer"].str.lower()

    print("Total observed turbines/farms before conditions: ", len(df))

    df["capacity"] = df["capacity"].astype(float)
    df["diameter"] = df["diameter"].astype(float)
    df["height"] = df["height"].astype(float)

    # removing turbines that are unrealistic
    df = df.drop(df[df["height"] < 1].index).reset_index(drop=True)

    df["p_density"] = (df["capacity"] * 1000) / (np.pi * (df["diameter"] / 2) ** 2)
    df["capacity"] = df["capacity"].astype(int)
    df["ID"] = df["ID"].astype(str)

    # manufacturer must be plain strings for difflib (pd.NA breaks it)
    if "manufacturer" in df.columns:
        df["manufacturer"] = df["manufacturer"].astype("string")
        df["manufacturer"] = df["manufacturer"].str.lower()
        df["manufacturer"] = df["manufacturer"].fillna("")  # <- critical
    else:
        df["manufacturer"] = ""
        
    # merging by manufacturer and power density if available
    if "manufacturer" in df:
        # df["manufacturer"] = df["manufacturer"].astype(pd.StringDtype())
        # df["manufacturer"] = df["manufacturer"].str.lower()

        df = df.merge(
            models.assign(
                match=models["manufacturer"].apply(
                    lambda x: difflib.get_close_matches(x, df["manufacturer"], cutoff=0.3, n=100)
                )
            )
            .explode("match")
            .drop_duplicates(),
            left_on=["manufacturer"],
            right_on=["match"],
            how="outer",
        )
        df = df.dropna(subset=["ID"])

        df = (
            df.assign(closest=np.abs(df["p_density_x"] - df["p_density_y"]))
            .sort_values("closest")
            .drop_duplicates(subset=["ID"], keep="first")
        )

        df["model"] = df["model"].where(df["closest"] < 1, np.nan)

        df = df.drop(
            ["diameter_y", "p_density_y", "offshore", "manufacturer_y", "capacity_y", "match", "closest", "manufacturer_x"],
            axis=1,
        )

    if "type" in df.columns:
        df.columns = ["ID", "capacity", "diameter", "height", "lon", "lat", "type", "p_density", "model"]
    else:
        df.columns = ["ID", "capacity", "diameter", "height", "lon", "lat", "p_density", "model"]
        df["type"] = "onshore"

    df = df[["ID", "type", "capacity", "diameter", "height", "lon", "lat", "model", "p_density"]]

    df = df.sort_values("p_density").reset_index(drop=True)
    models = models.sort_values("p_density")
    df.loc[df["model"].isna(), "model"] = pd.merge_asof(df, models, on="p_density", direction="nearest", tolerance=100)["model_y"]

    df = df.dropna(subset=["model"])
    df = df.sort_values("ID").reset_index(drop=True)

    return df


def format_bc_factors(train_bias_df, time_res):
    """
    Mean of all the bias corrections factors calculated for every cluster and time period.
    If the scalar is NA then set to unbiascorrected.
    """
    train_bias_df = train_bias_df.drop(["obs", "sim"], axis=1)
    train_bias_df["scalar"] = train_bias_df["scalar"].replace(0, np.nan)
    train_bias_df.columns = ["year", time_res, "cluster", "scalar", "offset"]

    bc_factors = train_bias_df.groupby(["cluster", time_res], as_index=False).agg({"scalar": "mean", "offset": "mean"})
    bc_factors.loc[bc_factors["scalar"].isna(), "offset"] = 0
    bc_factors.loc[bc_factors["scalar"].isna(), "scalar"] = 1

    return bc_factors


def add_times(df):
    """
    Add columns to identify year and month.
    """
    df["year"] = pd.DatetimeIndex(df["time"]).year
    df["month"] = pd.DatetimeIndex(df["time"]).month
    df.insert(1, "year", df.pop("year"))
    df.insert(2, "month", df.pop("month"))
    df["month"] = df["month"].astype(int)
    df["year"] = df["year"].astype(int)
    return df


def add_time_res(df):
    """
    Add columns to identify time resolutions.
    """
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


def prep_country(country, year_test=None, *, obs_level: str = "turbine"):
    """
    Country specific preprocessing of observational data.

    obs_level:
        - "turbine": returns turbine-level monthly obs (wide: obs_1..obs_12) + turb_info
        - "country": returns country-level monthly generation tidy: ['year','month','output_kwh'] + turb_info

    NOTE: For FR/NL country data we support obs_level="country" without pseudo-splitting.
    """
    train = year_test is None

    # -------------------------
    # Denmark
    # -------------------------
    if country == "DK":
        dk_md = pd.read_csv("input/country-data/DK/observations/DK_md.csv")
        columns = [
            "Turbine identifier (GSRN)",
            "Manufacture",
            "Capacity (kW)",
            "Rotor diameter (m)",
            "Hub height (m)",
            "X (east) coordinate\nUTM 32 Euref89",
            "Y (north) coordinate\nUTM 32 Euref89",
            "Type of location",
        ]
        dk_md = dk_md[columns]
        dk_md.columns = ["ID", "manufacturer", "capacity", "diameter", "height", "x_east_32", "y_north_32", "type"]

        dk_md["type"] = dk_md["type"].str.lower()
        dk_md.loc[dk_md["type"] == "land", "type"] = "onshore"
        dk_md.loc[dk_md["type"] == "hav", "type"] = "offshore"

        dk_md["x_east_32"] = pd.to_numeric(dk_md["x_east_32"], errors="coerce")
        dk_md["y_north_32"] = pd.to_numeric(dk_md["y_north_32"], errors="coerce")
        dk_md = dk_md.dropna(subset=["capacity", "diameter", "x_east_32", "y_north_32"]).reset_index(drop=True)

        def rule(row):
            lat, lon = utm.to_latlon(row["x_east_32"], row["y_north_32"], 32, "W")
            return pd.Series({"lat": lat, "lon": lon})

        dk_md = dk_md.merge(dk_md.apply(rule, axis=1), left_index=True, right_index=True)

        dk_md = dk_md[["ID", "manufacturer", "capacity", "diameter", "height", "lon", "lat", "type"]]
        dk_md["manufacturer"] = dk_md["manufacturer"].str.split(" ").str[0]

        turb_info = add_models(dk_md)

        if train:
            year_star, year_end = 2015, 2019
        else:
            year_star, year_end = year_test, year_test

        appended_data = []
        for i in range(year_star, year_end + 1):
            data = pd.read_excel(f"input/country-data/DK/observations/Denmark_{i}.xlsx")
            data = data.iloc[3:, np.r_[0:1, 3:15]]
            data.columns = ["ID", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
            data["ID"] = data["ID"].astype(str)
            data = data.reset_index(drop=True)
            data["year"] = i
            appended_data.append(data[:-1])

        obs_gen = pd.concat(appended_data).reset_index(drop=True)
        obs_gen = obs_gen.fillna(0)

    # -------------------------
    # Germany
    # -------------------------
    elif country == "DE":
        de_geo = pd.read_csv("input/country-data/DE/observations/geolocate.germany.csv")
        de_md = pd.read_csv("input/country-data/DE/observations/DE_md.csv")

        de_md = de_md[["V1", "Manufacturer", "kW", "Rotor..m.", "Tower..m."]]
        de_md.columns = ["ID", "manufacturer", "capacity", "diameter", "height"]
        de_md["postcode"] = de_md["ID"].astype(str).str[:5].astype(int)

        de_md = pd.merge(de_md, de_geo[["postcode", "lon", "lat"]], on="postcode", how="left")
        de_md = de_md.drop(["postcode"], axis=1)
        de_md = de_md.dropna(subset=["capacity", "diameter", "lon", "lat"]).reset_index(drop=True)

        turb_info = add_models(de_md)

        if train:
            year_star, year_end = 2015, 2018
        else:
            year_star, year_end = year_test, year_test

        de_data = pd.read_csv("input/country-data/DE/observations/DE_data.csv")
        de_data = (
            de_data.loc[(de_data["Year"] >= year_star) & (de_data["Year"] <= year_end)]
            .drop(["Downtime"], axis=1)
            .reset_index(drop=True)
        )
        de_data.columns = ["ID", "year", "month", "output"]
        de_data = de_data.dropna(subset=["ID", "year", "month"])
        obs_gen = de_data.pivot(index=["ID", "year"], columns="month", values="output").reset_index()
        obs_gen = obs_gen.fillna(0)

    # -------------------------
    # UK
    # -------------------------
    elif country == "UK":
        uk_md = pd.read_csv("input/country-data/UK/observations/uk_md.csv")
        turb_info = add_models(uk_md)

        if train:
            year_star, year_end = 2015, 2018
        else:
            year_star, year_end = year_test, year_test

        obs_gen = pd.read_csv("input/country-data/UK/observations/ukobs.csv")

    # -------------------------
    # France
    # -------------------------
    elif country == "FR":
        fr_md = gpd.read_file("input/country-data/fr/fr_turb_info.csv")
        fr_md = fr_md.loc[fr_md["statut_parc"] == "Autorisé"].reset_index(drop=True)
        fr_md = fr_md.loc[
            :,
            [
                "id_aerogenerateur",
                "puissance_mw",
                "diametre_rotor",
                "hauteur_mat_nacelle",
                "constructeur",
                "x_aerogenerateur",
                "y_aerogenerateur",
                "epsg",
            ],
        ]
        fr_md.columns = ["ID", "capacity", "diameter", "height", "manufacturer", "x", "y", "epsg"]

        fr_md = gpd.GeoDataFrame(
            fr_md[["ID", "capacity", "diameter", "height", "manufacturer"]],
            geometry=gpd.points_from_xy(fr_md.x, fr_md.y, crs=fr_md.epsg.iloc[0]),
        ).to_crs(epsg=4326)

        fr_md["capacity"] = fr_md["capacity"].astype(float) * 1e3  # MW -> kW
        fr_md["lon"] = fr_md.geometry.x
        fr_md["lat"] = fr_md.geometry.y
        fr_md = fr_md.drop(columns="geometry")
        
        # numeric coercion
        for c in ["capacity", "diameter", "height", "lon", "lat"]:
            if c in fr_md.columns:
                fr_md[c] = pd.to_numeric(fr_md[c], errors="coerce")
        if obs_level == "country":
            # country-level path: do NOT call add_models()
            turb_info = fr_md.dropna(subset=["capacity", "height", "lon", "lat"]).reset_index(drop=True)

            # ensure required cols exist
            if "type" not in turb_info.columns:
                turb_info["type"] = "onshore"

            # (optional) standardise capacity units here if needed
            turb_info["capacity"] = turb_info["capacity"] * 1000.0  # MW->kW if your file is MW

        else:
            # turbine-level path: require model assignment
            fr_md = fr_md.dropna(subset=["capacity", "diameter"]).reset_index(drop=True)
            turb_info = add_models(fr_md)

        if train:
            year_star, year_end = 2015, 2018
        else:
            year_star, year_end = year_test, year_test

        ns_data = pd.read_csv("input/country-data/northsea_country_generation.csv")
        ns_data = ns_data.loc[:, ["Standard international energy product classification (SIEC)", "TIME_PERIOD", "OBS_VALUE", "geo"]]
        ns_data.columns = ["carrier", "date", "output", "country"]

        ns_data = ns_data.loc[(ns_data["country"] == country) & (ns_data["carrier"] == "Wind")].reset_index(drop=True)
        ns_data["date"] = pd.to_datetime(ns_data["date"])
        ns_data["output"] = pd.to_numeric(ns_data["output"])
        ns_data["output"] = ns_data["output"] * 1e6  # GWh -> kWh
        ns_data["year"] = ns_data["date"].dt.year.astype(int)
        ns_data["month"] = ns_data["date"].dt.month.astype(int)
        ns_data = ns_data.drop(columns=["date"])
        ns_data = ns_data.fillna(0).groupby(["year", "month"])["output"].sum().reset_index()
        ns_data = ns_data.loc[(ns_data["year"] >= year_star) & (ns_data["year"] <= year_end)].reset_index(drop=True)

        if obs_level == "country":
            return ns_data.rename(columns={"output": "output_kwh"}), turb_info

        # legacy pseudo split (kept for backwards-compat, but not recommended)
        turb_info["ratio"] = turb_info["capacity"] / turb_info["capacity"].sum()
        ns_data = ns_data.merge(turb_info[["ID", "ratio"]], how="cross")
        ns_data["output"] = ns_data["output"] * ns_data["ratio"]
        ns_data = ns_data.dropna(subset=["ID", "year", "month"])
        obs_gen = ns_data.pivot(index=["ID", "year"], columns="month", values="output").reset_index()

    # -------------------------
    # Netherlands
    # -------------------------
    elif country == "NL":
        nl_md = gpd.read_file("input/country-data/NL/nl_md.json").to_crs(epsg=4326)
        nl_md["lon"] = nl_md.geometry.x
        nl_md["lat"] = nl_md.geometry.y
        nl_md = nl_md.drop(columns=["geometry", "x", "y", "prov_naam", "gem_naam", "naam"])
        nl_md["ondergrond"] = nl_md["ondergrond"].replace({"land": "onshore", "zee": "offshore"})
        nl_md["land"] = nl_md["land"].replace({"België": "BE", "Duitsland": "DE", "Nederland": "NL"})
        nl_md.columns = ["ID", "diameter", "height", "capacity", "country", "manufacturer", "type", "lon", "lat"]
        nl_md = nl_md.loc[nl_md["country"] == "NL"].reset_index(drop=True).drop(columns=["country"])
        nl_md = nl_md[["ID", "capacity", "diameter", "height", "manufacturer", "lon", "lat", "type"]]
        nl_md["manufacturer"] = nl_md["manufacturer"].str.split(" ").str[0].str.strip("123-.,").astype(str)

        # numeric coercion
        for c in ["capacity", "diameter", "height", "lon", "lat"]:
            if c in nl_md.columns:
                nl_md[c] = pd.to_numeric(nl_md[c], errors="coerce")

        if obs_level == "country":
            # country-level path: do NOT call add_models()
            turb_info = nl_md.dropna(subset=["capacity", "height", "lon", "lat"]).reset_index(drop=True)

            # ensure required cols exist
            if "type" not in turb_info.columns:
                turb_info["type"] = "onshore"

            # (optional) standardise capacity units here if needed
            turb_info["capacity"] = turb_info["capacity"] * 1000.0  # MW->kW if your file is MW

        else:
            # turbine-level path: require model assignment
            nl_md = nl_md.dropna(subset=["capacity", "diameter"]).reset_index(drop=True)
            turb_info = add_models(nl_md)

        if train:
            year_star, year_end = 2015, 2018
        else:
            year_star, year_end = year_test, year_test

        ns_data = pd.read_csv("input/country-data/northsea_country_generation.csv")
        ns_data = ns_data.loc[:, ["Standard international energy product classification (SIEC)", "TIME_PERIOD", "OBS_VALUE", "geo"]]
        ns_data.columns = ["carrier", "date", "output", "country"]

        ns_data = ns_data.loc[(ns_data["country"] == country) & (ns_data["carrier"] == "Wind")].reset_index(drop=True)
        ns_data["date"] = pd.to_datetime(ns_data["date"])
        ns_data["output"] = pd.to_numeric(ns_data["output"])
        ns_data["output"] = ns_data["output"] * 1e6  # GWh -> kWh
        ns_data["year"] = ns_data["date"].dt.year.astype(int)
        ns_data["month"] = ns_data["date"].dt.month.astype(int)
        ns_data = ns_data.drop(columns=["date"])
        ns_data = ns_data.fillna(0).groupby(["year", "month"])["output"].sum().reset_index()
        ns_data = ns_data.loc[(ns_data["year"] >= year_star) & (ns_data["year"] <= year_end)].reset_index(drop=True)

        if obs_level == "country":
            return ns_data.rename(columns={"output": "output_kwh"}), turb_info

        # legacy pseudo split (kept for backwards-compat, but not recommended)
        turb_info["ratio"] = turb_info["capacity"] / turb_info["capacity"].sum()
        ns_data = ns_data.merge(turb_info[["ID", "ratio"]], how="cross")
        ns_data["output"] = ns_data["output"] * ns_data["ratio"]
        ns_data = ns_data.dropna(subset=["ID", "year", "month"])
        obs_gen = ns_data.pivot(index=["ID", "year"], columns="month", values="output").reset_index()

    # -------------------------
    # Belgium
    # -------------------------
    elif country == "BE":
        be_md = pd.read_csv("input/country-data/BE/be_md.csv")

        # numeric coercion
        for c in ["capacity", "diameter", "height", "lon", "lat"]:
            if c in be_md.columns:
                be_md[c] = pd.to_numeric(be_md[c], errors="coerce")

        if obs_level == "country":
            # country-level path: do NOT call add_models()
            turb_info = be_md.dropna(subset=["capacity", "height", "lon", "lat"]).reset_index(drop=True)

            # ensure required cols exist
            if "type" not in turb_info.columns:
                turb_info["type"] = "onshore"

            # (optional) standardise capacity units here if needed
            turb_info["capacity"] = turb_info["capacity"] * 1000.0  # MW->kW if your file is MW

        else:
            # turbine-level path: require model assignment
            be_md = be_md.dropna(subset=["capacity", "diameter"]).reset_index(drop=True)
            turb_info = add_models(be_md)

        if train:
            year_star, year_end = 2015, 2018
        else:
            year_star, year_end = year_test, year_test

        ns_data = pd.read_csv("input/country-data/northsea_country_generation.csv")
        ns_data = ns_data.loc[:, ["Standard international energy product classification (SIEC)", "TIME_PERIOD", "OBS_VALUE", "geo"]]
        ns_data.columns = ["carrier", "date", "output", "country"]

        ns_data = ns_data.loc[(ns_data["country"] == country) & (ns_data["carrier"] == "Wind")].reset_index(drop=True)
        ns_data["date"] = pd.to_datetime(ns_data["date"])
        ns_data["output"] = pd.to_numeric(ns_data["output"])
        ns_data["output"] = ns_data["output"] * 1e6  # GWh -> kWh
        ns_data["year"] = ns_data["date"].dt.year.astype(int)
        ns_data["month"] = ns_data["date"].dt.month.astype(int)
        ns_data = ns_data.drop(columns=["date"])
        ns_data = ns_data.fillna(0).groupby(["year", "month"])["output"].sum().reset_index()
        ns_data = ns_data.loc[(ns_data["year"] >= year_star) & (ns_data["year"] <= year_end)].reset_index(drop=True)

        if obs_level == "country":
            return ns_data.rename(columns={"output": "output_kwh"}), turb_info

        # legacy pseudo split (kept for backwards-compat, but not recommended)
        turb_info["ratio"] = turb_info["capacity"] / turb_info["capacity"].sum()
        ns_data = ns_data.merge(turb_info[["ID", "ratio"]], how="cross")
        ns_data["output"] = ns_data["output"] * ns_data["ratio"]
        ns_data = ns_data.dropna(subset=["ID", "year", "month"])
        obs_gen = ns_data.pivot(index=["ID", "year"], columns="month", values="output").reset_index()
    
    # -------------------------
    # Norway
    # -------------------------
    elif country == "NO":
        no_md = pd.read_csv("input/country-data/NO/no_md.csv")

        # numeric coercion
        for c in ["capacity", "diameter", "height", "lon", "lat"]:
            if c in no_md.columns:
                no_md[c] = pd.to_numeric(no_md[c], errors="coerce")

        if obs_level == "country":
            # country-level path: do NOT call add_models()
            turb_info = no_md.dropna(subset=["capacity", "height", "lon", "lat"]).reset_index(drop=True)

            # ensure required cols exist
            if "type" not in turb_info.columns:
                turb_info["type"] = "onshore"

            # (optional) standardise capacity units here if needed
            turb_info["capacity"] = turb_info["capacity"] * 1000.0  # MW->kW if your file is MW

        else:
            # turbine-level path: require model assignment
            be_md = be_md.dropna(subset=["capacity", "diameter"]).reset_index(drop=True)
            turb_info = add_models(be_md)

        if train:
            year_star, year_end = 2015, 2018
        else:
            year_star, year_end = year_test, year_test

        ns_data = pd.read_csv("input/country-data/northsea_country_generation.csv")
        ns_data = ns_data.loc[:, ["Standard international energy product classification (SIEC)", "TIME_PERIOD", "OBS_VALUE", "geo"]]
        ns_data.columns = ["carrier", "date", "output", "country"]

        ns_data = ns_data.loc[(ns_data["country"] == country) & (ns_data["carrier"] == "Wind")].reset_index(drop=True)
        ns_data["date"] = pd.to_datetime(ns_data["date"])
        ns_data["output"] = pd.to_numeric(ns_data["output"])
        ns_data["output"] = ns_data["output"] * 1e6  # GWh -> kWh
        ns_data["year"] = ns_data["date"].dt.year.astype(int)
        ns_data["month"] = ns_data["date"].dt.month.astype(int)
        ns_data = ns_data.drop(columns=["date"])
        ns_data = ns_data.fillna(0).groupby(["year", "month"])["output"].sum().reset_index()
        ns_data = ns_data.loc[(ns_data["year"] >= year_star) & (ns_data["year"] <= year_end)].reset_index(drop=True)

        if obs_level == "country":
            return ns_data.rename(columns={"output": "output_kwh"}), turb_info

        # legacy pseudo split (kept for backwards-compat, but not recommended)
        turb_info["ratio"] = turb_info["capacity"] / turb_info["capacity"].sum()
        ns_data = ns_data.merge(turb_info[["ID", "ratio"]], how="cross")
        ns_data["output"] = ns_data["output"] * ns_data["ratio"]
        ns_data = ns_data.dropna(subset=["ID", "year", "month"])
        obs_gen = ns_data.pivot(index=["ID", "year"], columns="month", values="output").reset_index()
        
    else:
        raise ValueError("Country not recognised, please choose from: DK, DE, UK, FR, NL, BE.")

    # -------------------------------------------------------------------------
    # Common processing for turbine-level obs: power -> CF and return obs_cf wide
    # -------------------------------------------------------------------------
    if obs_level == "country":
        raise RuntimeError("Internal error: country obs_level should have returned earlier in FR/NL branches.")

    obs_gen.columns = [f"obs_{i}" if i not in ["ID", "year"] else f"{i}" for i in obs_gen.columns]
    obs_gen = obs_gen.merge(turb_info[["ID", "capacity"]], how="left", on=["ID"])
    obs_gen = obs_gen.dropna().reset_index(drop=True)

    def daysDuringMonth(yy, m):
        result = []
        [result.append(monthrange(y, m)[1]) for y in yy]
        return result

    for i in range(1, 13):
        obs_gen[f"obs_{i}"] = obs_gen[f"obs_{i}"] / (((daysDuringMonth(obs_gen.year, i)) * obs_gen["capacity"]) * 24)

    obs_gen = obs_gen.drop(["capacity"], axis=1)
    return obs_gen, turb_info