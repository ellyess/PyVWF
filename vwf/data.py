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

from pathlib import Path

COUNTRY_DIR = Path("input/country-data")

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _year_range(train: bool, year_test: int | None, default_train=(2015, 2018)) -> tuple[int, int]:
    if train:
        return int(default_train[0]), int(default_train[1])
    if year_test is None:
        raise ValueError("year_test must be provided when train=False")
    return int(year_test), int(year_test)


def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _standardise_turb_info_minimal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure columns exist and types are correct for wind.interpolate_wind().
    Needs at least: ID, capacity, height, lon, lat. Adds default 'type' if missing.
    """
    df = df.copy()

    if "ID" not in df.columns:
        raise ValueError("turbine metadata must contain 'ID'")
    df["ID"] = df["ID"].astype(str)

    if "type" not in df.columns:
        df["type"] = "onshore"

    df = _ensure_numeric(df, ["capacity", "diameter", "height", "lon", "lat"])

    # enforce physical hub heights
    df = df[df["height"].notna()]
    df = df[df["height"] > 1.0]   # or >0, but >1 m avoids pathological cases

    if df.empty:
        raise ValueError("No turbines with valid hub height (>1 m) after standardisation")

    return df.reset_index(drop=True)

def load_fr_turbine_metadata_standard(
    path: str | Path,
    *,
    only_authorised: bool = True,
    default_type: str = "onshore",
) -> pd.DataFrame:
    """
    Load France turbine metadata into the standard PyVWF turb_info format:
      ID, capacity (kW), diameter (m), height (m), lon, lat, manufacturer, type

    Handles multiple EPSG values by converting per-group to EPSG:4326.
    """
    path = Path(path)
    df = pd.read_csv(path)

    if only_authorised and "statut_parc" in df.columns:
        df = df.loc[df["statut_parc"] == "Autorisé"].copy()

    # Select + rename
    keep = {
        "id_aerogenerateur": "ID",
        "puissance_mw": "capacity_mw",
        "diametre_rotor": "diameter",
        "hauteur_mat_nacelle": "height",
        "constructeur": "manufacturer",
        "x_aerogenerateur": "x",
        "y_aerogenerateur": "y",
        "epsg": "epsg",
    }
    missing = [k for k in keep if k not in df.columns]
    if missing:
        raise ValueError(f"FR metadata missing columns: {missing}")

    df = df[list(keep.keys())].rename(columns=keep)

    # Numeric coercion
    for c in ["capacity_mw", "diameter", "height", "x", "y", "epsg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ID"] = df["ID"].astype(str)
    df["manufacturer"] = df["manufacturer"].astype("string")

    # Build geometry + reproject (supports mixed EPSG)
    df = df.dropna(subset=["x", "y", "epsg"]).copy()

    parts = []
    for epsg, g in df.groupby("epsg"):
        if not np.isfinite(epsg):
            continue
        gdf = gpd.GeoDataFrame(
            g,
            geometry=gpd.points_from_xy(g["x"], g["y"]),
            crs=f"EPSG:{int(epsg)}",
        ).to_crs(4326)
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
        parts.append(gdf.drop(columns=["geometry"]))

    if not parts:
        raise ValueError("FR metadata: could not build any valid geometries (x/y/epsg).")

    out = pd.concat(parts, ignore_index=True)

    # Convert MW -> kW
    out["capacity"] = out["capacity_mw"] * 1000.0

    out["type"] = default_type
    out = out.drop(columns=["capacity_mw", "x", "y", "epsg"])

    # Final sanity filter: needs these for interpolate_wind()
    out = out.dropna(subset=["capacity", "height", "lon", "lat"]).reset_index(drop=True)

    return out[["ID", "type", "capacity", "diameter", "height", "lon", "lat", "manufacturer"]]
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

    if cap_kw < 10_000:  # entire country < 10 MW? probably wrong units
        raise ValueError("Total capacity looks too small; check units (kW vs MW).")
    
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

# -----------------------------------------------------------------------------
# Country generation (generic, extendable)
# -----------------------------------------------------------------------------
def load_country_generation_monthly_kwh(country: str, year_start: int, year_end: int) -> pd.DataFrame:
    """
    Return tidy monthly country generation:
        columns: ['year','month','output_kwh']

    Currently tries your cached northsea_country_generation.csv (GWh -> kWh).
    Add more providers here later (e.g. OPSD cached monthly).
    """
    country = country.upper()

    ns_path = COUNTRY_DIR / "northsea_country_generation.csv"
    if ns_path.exists():
        ns = pd.read_csv(ns_path)
        ns = ns.loc[:, ["Standard international energy product classification (SIEC)", "TIME_PERIOD", "OBS_VALUE", "geo"]]
        ns.columns = ["carrier", "date", "output", "country"]

        ns = ns.loc[(ns["country"] == country) & (ns["carrier"] == "Wind")].copy()
        if not ns.empty:
            ns["date"] = pd.to_datetime(ns["date"], errors="coerce")
            ns["output"] = pd.to_numeric(ns["output"], errors="coerce") * 1e6  # GWh -> kWh
            ns["year"] = ns["date"].dt.year.astype(int)
            ns["month"] = ns["date"].dt.month.astype(int)

            out = (
                ns.drop(columns=["date", "carrier", "country"])
                .groupby(["year", "month"], as_index=False)["output"]
                .sum()
                .rename(columns={"output": "output_kwh"})
            )
            out = out.loc[(out["year"] >= year_start) & (out["year"] <= year_end)].reset_index(drop=True)
            return out

    raise ValueError(
        f"No country-level generation data available for {country} in known sources "
        f"(tried: {ns_path})."
    )


# -----------------------------------------------------------------------------
# Turbine metadata loaders (country-specific; keep small + focused)
# -----------------------------------------------------------------------------
def load_turbine_metadata(country: str) -> pd.DataFrame:
    """
    Return raw turbine metadata with (ideally):
      ID, capacity, diameter, height, manufacturer (optional), lon, lat, type (optional)
    Capacity convention should be kW (recommended). If a source is MW, convert to kW here.
    """
    country = country.upper()

    if country == "DK":
        dk_md = pd.read_csv(COUNTRY_DIR / "DK/observations/DK_md.csv")
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

        dk_md["type"] = dk_md["type"].str.lower().replace({"land": "onshore", "hav": "offshore"})
        dk_md = _ensure_numeric(dk_md, ["x_east_32", "y_north_32", "capacity", "diameter", "height"])
        dk_md = dk_md.dropna(subset=["capacity", "diameter", "x_east_32", "y_north_32"]).reset_index(drop=True)

        def rule(row):
            lat, lon = utm.to_latlon(row["x_east_32"], row["y_north_32"], 32, "W")
            return pd.Series({"lat": lat, "lon": lon})

        dk_md = dk_md.merge(dk_md.apply(rule, axis=1), left_index=True, right_index=True)
        dk_md = dk_md[["ID", "manufacturer", "capacity", "diameter", "height", "lon", "lat", "type"]]
        dk_md["manufacturer"] = dk_md["manufacturer"].astype(str).str.split(" ").str[0]
        return _standardise_turb_info_minimal(dk_md)

    if country == "DE":
        de_geo = pd.read_csv(COUNTRY_DIR / "DE/observations/geolocate.germany.csv")
        de_md = pd.read_csv(COUNTRY_DIR / "DE/observations/DE_md.csv")

        de_md = de_md[["V1", "Manufacturer", "kW", "Rotor..m.", "Tower..m."]]
        de_md.columns = ["ID", "manufacturer", "capacity", "diameter", "height"]
        de_md["postcode"] = de_md["ID"].astype(str).str[:5].astype(int)

        de_md = pd.merge(de_md, de_geo[["postcode", "lon", "lat"]], on="postcode", how="left").drop(columns=["postcode"])
        de_md = de_md.dropna(subset=["capacity", "diameter", "lon", "lat"]).reset_index(drop=True)
        de_md["type"] = "onshore"
        return _standardise_turb_info_minimal(de_md)

    if country == "UK":
        uk_md = pd.read_csv(COUNTRY_DIR / "UK/observations/uk_md.csv")
        return _standardise_turb_info_minimal(uk_md)

    if country == "FR":
        return _standardise_turb_info_minimal(
            load_fr_turbine_metadata_standard(COUNTRY_DIR / "fr/fr_turb_info.csv")
        )

    if country == "NL":
        nl_md = gpd.read_file(COUNTRY_DIR / "NL/nl_md.json").to_crs(epsg=4326)
        nl_md["lon"] = nl_md.geometry.x
        nl_md["lat"] = nl_md.geometry.y
        nl_md = nl_md.drop(columns=["geometry", "x", "y", "prov_naam", "gem_naam", "naam"])
        nl_md["ondergrond"] = nl_md["ondergrond"].replace({"land": "onshore", "zee": "offshore"})
        nl_md["land"] = nl_md["land"].replace({"België": "BE", "Duitsland": "DE", "Nederland": "NL"})
        nl_md.columns = ["ID", "diameter", "height", "capacity", "country", "manufacturer", "type", "lon", "lat"]
        nl_md = nl_md.loc[nl_md["country"] == "NL"].reset_index(drop=True).drop(columns=["country"])
        nl_md = nl_md[["ID", "capacity", "diameter", "height", "manufacturer", "lon", "lat", "type"]]
        nl_md["manufacturer"] = nl_md["manufacturer"].astype(str).str.split(" ").str[0].str.strip("123-.,")
        return _standardise_turb_info_minimal(nl_md)

    if country == "BE":
        be_md = pd.read_csv(COUNTRY_DIR / "BE/be_md.csv")
        return _standardise_turb_info_minimal(be_md)

    if country == "NO":
        no_md = pd.read_csv(COUNTRY_DIR / "NO/no_md.csv")
        return _standardise_turb_info_minimal(no_md)

    raise ValueError(f"Unsupported country={country}")


# -----------------------------------------------------------------------------
# Turbine-level observations loaders
# -----------------------------------------------------------------------------
def load_turbine_observations(country: str, year_start: int, year_end: int) -> pd.DataFrame:
    """
    Return turbine-level monthly generation in wide form:
        columns: ID, year, 1..12 (or equivalent)
    """
    country = country.upper()

    if country == "DK":
        appended = []
        for y in range(year_start, year_end + 1):
            data = pd.read_excel(COUNTRY_DIR / f"DK/observations/Denmark_{y}.xlsx")
            data = data.iloc[3:, np.r_[0:1, 3:15]]
            data.columns = ["ID", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
            data["ID"] = data["ID"].astype(str)
            data["year"] = y
            appended.append(data[:-1])
        obs = pd.concat(appended).reset_index(drop=True).fillna(0)
        return obs

    if country == "DE":
        de_data = pd.read_csv(COUNTRY_DIR / "DE/observations/DE_data.csv")
        de_data = (
            de_data.loc[(de_data["Year"] >= year_start) & (de_data["Year"] <= year_end)]
            .drop(columns=["Downtime"])
            .reset_index(drop=True)
        )
        de_data.columns = ["ID", "year", "month", "output"]
        de_data = de_data.dropna(subset=["ID", "year", "month"])
        obs = de_data.pivot(index=["ID", "year"], columns="month", values="output").reset_index().fillna(0)
        return obs

    if country == "UK":
        # expecting a wide monthly dataset similar to your current usage
        obs = pd.read_csv(COUNTRY_DIR / "UK/observations/ukobs.csv")
        return obs

    raise ValueError(f"No turbine-level observation loader implemented for {country}.")


# -----------------------------------------------------------------------------
# Main orchestrator
# -----------------------------------------------------------------------------
def prep_country(country, year_test=None, *, obs_level: str = "turbine"):
    """
    Country specific preprocessing of observational data.

    obs_level:
        - "turbine": returns turbine-level monthly obs CF (wide: obs_1..obs_12) + turb_info
        - "country": returns country-level monthly generation tidy: ['year','month','output_kwh'] + turb_info
    """
    country = country.upper()
    train = year_test is None

    # Per-country training windows (fallback default)
    default_train = {
        "DK": (2015, 2019),
        "DE": (2015, 2018),
        "UK": (2015, 2018),
    }.get(country, (2015, 2018))

    year_start, year_end = _year_range(train, year_test, default_train=default_train)

    # ---- metadata ----
    turb_raw = load_turbine_metadata(country)

    # ---- country-level path (generic) ----
    if obs_level == "country":
        turb_info = turb_raw.dropna(subset=["capacity", "height", "lon", "lat"]).reset_index(drop=True)
        if "type" not in turb_info.columns:
            turb_info["type"] = "onshore"

        obs_country = load_country_generation_monthly_kwh(country, year_start, year_end)
        return obs_country, turb_info

    # ---- turbine-level path ----
    if obs_level != "turbine":
        raise ValueError("obs_level must be 'turbine' or 'country'")

    turb_info = add_models(turb_raw)

    obs_gen = load_turbine_observations(country, year_start, year_end).copy()
    if not {"ID", "year"}.issubset(obs_gen.columns):
        raise ValueError("Turbine observations must contain columns ['ID','year', ...months...]")

    # Standardise month columns to obs_1..obs_12 (if not already)
    month_cols = [c for c in obs_gen.columns if c not in ["ID", "year"]]
    # If columns are "obs_1"... already, keep them; else rename to obs_{col}
    if not any(str(c).startswith("obs_") for c in month_cols):
        obs_gen.columns = [f"obs_{c}" if c not in ["ID", "year"] else c for c in obs_gen.columns]

    # Merge capacity for CF conversion
    obs_gen["ID"] = obs_gen["ID"].astype(str)
    obs_gen["year"] = pd.to_numeric(obs_gen["year"], errors="coerce").astype("Int64")

    obs_gen = obs_gen.merge(turb_info[["ID", "capacity"]], how="left", on="ID")
    obs_gen = obs_gen.dropna(subset=["capacity", "year"]).reset_index(drop=True)

    # Convert monthly output -> CF using hours in month and turbine capacity (kW)
    for m in range(1, 13):
        col = f"obs_{m}"
        if col not in obs_gen.columns:
            continue
        days = obs_gen["year"].astype(int).map(lambda y: monthrange(int(y), int(m))[1]).astype(float)
        obs_gen[col] = pd.to_numeric(obs_gen[col], errors="coerce") / (days * 24.0 * obs_gen["capacity"].astype(float))

    obs_gen = obs_gen.drop(columns=["capacity"])
    return obs_gen, turb_info


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
    power_curves = load_power_curves()

    # -------------------------
    # Country-level branch
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
        
        sim_country = sim_country_cf.rename("sim").to_frame().reset_index()

        # -------------------------------------------------
        # Ensure column names and types
        # -------------------------------------------------
        sim_country.columns = ["time", "sim"]  # because it's a Series with datetime index
        sim_country["time"] = pd.to_datetime(sim_country["time"], errors="coerce")
        sim_country = sim_country.dropna(subset=["time"]).reset_index(drop=True)

        sim_country["year"] = sim_country["time"].dt.year.astype(int)
        sim_country["month"] = sim_country["time"].dt.month.astype(int)
        sim_country = sim_country[["year", "month", "sim"]]

        gen_cf = sim_country.merge(obs_country, on=["year", "month"], how="inner")
        gen_cf = add_time_res(gen_cf)

        return gen_cf.reset_index(drop=True), turb_info, reanalysis, power_curves

    # ---------------------------------
    # Turbine-level branch
    # ---------------------------------
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

    obs_cf = obs_cf[[
        "ID", "year",
        "obs_1","obs_2","obs_3","obs_4","obs_5","obs_6",
        "obs_7","obs_8","obs_9","obs_10","obs_11","obs_12"
    ]]
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
        
        if "model" not in turb_info.columns or turb_info["model"].isna().all():
            default_model = fix_turb if fix_turb is not None else _default_power_curve(power_curves)
            turb_info = turb_info.copy()
            turb_info["model"] = default_model
            
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
    Interpolate nans in observed data (expects long-form: ID, year, month, obs).
    """
    df = df.sort_values(["ID", "year", "month"]).copy()

    def _interp(g):
        g = g.copy()
        g["obs"] = g["obs"].interpolate(method="linear", limit=limit, limit_direction="both")
        return g

    return df.groupby("ID", group_keys=False).apply(_interp).reset_index(drop=True)


def add_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign model names to input turbines.

    Requires (at minimum) columns:
        ID, capacity, diameter, height, lon, lat
    Optional:
        manufacturer, type
    """
    models = pd.read_csv("input/models.csv")
    models["model"] = models["model"].astype("string")
    models["manufacturer"] = models["manufacturer"].astype("string").str.lower().fillna("")
    models = models.sort_values("p_density").reset_index(drop=True)

    df = df.copy()

    # --- Ensure required columns exist ---
    required = ["ID", "capacity", "diameter", "height", "lon", "lat"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"add_models: missing required columns: {missing}")

    # --- Coerce numerics safely ---
    for c in ["capacity", "diameter", "height", "lon", "lat"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows missing core numeric fields
    df = df.dropna(subset=["ID", "capacity", "diameter", "height", "lon", "lat"]).reset_index(drop=True)

    # Remove unrealistic turbines
    df = df.loc[df["height"] >= 1].reset_index(drop=True)

    # --- Manufacturer cleanup (difflib cannot handle pd.NA) ---
    if "manufacturer" in df.columns:
        df["manufacturer"] = df["manufacturer"].astype("string").str.lower().fillna("")
    else:
        df["manufacturer"] = ""

    # Default type
    if "type" not in df.columns:
        df["type"] = "onshore"
    else:
        df["type"] = df["type"].astype("string").fillna("onshore")

    df["ID"] = df["ID"].astype(str)

    # Compute power density (NOTE: your capacity is in kW; convert to W for density)
    df["p_density"] = (df["capacity"] * 1000.0) / (np.pi * (df["diameter"] / 2.0) ** 2)

    # --- Fuzzy match manufacturer against model manufacturers ---
    # Create candidate pairs by manufacturer similarity, then pick closest p_density
    # (This keeps your original logic but makes it robust.)
    cand = models.assign(
        match=models["manufacturer"].apply(lambda x: difflib.get_close_matches(x, df["manufacturer"].tolist(), cutoff=0.3, n=50))
    ).explode("match")

    if cand["match"].isna().all():
        # If no manufacturer matches at all, fall back to nearest p_density later
        df["model"] = pd.NA
    else:
        merged = df.merge(
            cand.drop_duplicates(subset=["manufacturer", "match", "model", "p_density"]),
            left_on="manufacturer",
            right_on="match",
            how="left",
            suffixes=("", "_m"),
        )

        # choose closest p_density among matched manufacturer candidates
        merged["closest"] = (merged["p_density"] - merged["p_density_m"]).abs()
        merged = merged.sort_values(["ID", "closest"])
        merged = merged.drop_duplicates(subset=["ID"], keep="first")

        # accept manufacturer-based match only if close enough
        merged["model"] = merged["model"].where(merged["closest"] < 1, pd.NA)

        df = merged[["ID", "type", "capacity", "diameter", "height", "lon", "lat", "p_density", "model"]].copy()

    # --- Final fallback: nearest p_density across all models ---
    # merge_asof requires sorted keys
    df = df.sort_values("p_density").reset_index(drop=True)
    fallback = pd.merge_asof(
        df[["p_density"]],
        models[["p_density", "model"]],
        on="p_density",
        direction="nearest",
        tolerance=100,
    )["model"]

    df["model"] = df["model"].fillna(fallback)

    # Drop if still no model
    df = df.dropna(subset=["model"]).reset_index(drop=True)

    # Keep types clean
    df["capacity"] = df["capacity"].astype(float)
    df["diameter"] = df["diameter"].astype(float)
    df["height"] = df["height"].astype(float)

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
