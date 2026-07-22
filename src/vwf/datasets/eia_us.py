"""Transforms for building the US (EIA) inputs from the public federal data.

Pure frame-to-frame logic for ``scripts/process_eia_us.py``: reshaping the
EIA-923 monthly generation schedule, summing EIA-860 wind nameplate to plant
capacity with coordinates, aggregating the USGS/LBNL US Wind Turbine Database
(USWTDB) turbine attributes to the plant, and emitting the ``EIAUSSource``
metadata contract.

Everything here takes and returns DataFrames so it is testable without the
raw federal files or an xlsx reader; file I/O lives in the script.

The unit of observation is the **plant** (``obs_unit = "plant"``): EIA-923
reports *net* generation once per plant per prime-mover/fuel per month, so the
plant is the site we simulate — exactly parallel to the AU adapter treating the
*farm* (DUID) as the unit while the mechanical ``obs_level`` stays "turbine".

Two data-hygiene decisions are encoded here and documented loudly in the join
report, because both would otherwise silently corrupt the seasonal-cycle target:

- **Net, not gross.** EIA-923 ``Netgen_*`` is generation net of station use.
  The correction therefore absorbs a small parasitic-load loss the European
  monthly *generation* data does not. Recorded per-region so α/β are not
  silently soaking up station use inconsistently across countries.
- **Annual respondents are imputed.** A plant flagged ``A`` (annual) in
  EIA-923 has its 12 monthly ``Netgen_*`` cells filled by an EIA *model*
  distributing the annual total, not by measurement. Training the seasonal
  cycle on those months would fit a synthetic profile. They are screened out
  in the source (``drop_annual_respondents=True``); this module carries the
  frequency flag through so the source can act on it.

Known limitations, stated up front (also in the join report):

- USWTDB carries hub height (``t_hh``) for most but not all turbines. Plants
  with no valid ``t_hh`` fall back to a uniform default; the ``height_source``
  column says which, per plant, so no downstream step can mistake a default
  for data. This is already better than the AU branch (which had no
  hub-height data at all), and the vintage-aware power-curve assignment
  remains a named follow-up (RQ7 territory), exactly as for AU.
- Capacity is the EIA-860 nameplate, static over the window. Plants built out
  in stages inside the training window have their early months biased low
  relative to the evolving in-service capacity; the commissioning mask removes
  pre-operating months but not partial staging (the AU DUDETAIL-style capacity
  history has no clean EIA-860 analogue at monthly resolution — a named
  follow-up).
"""
from __future__ import annotations

import calendar

import pandas as pd

#: EIA-923 fuel code for wind; the unambiguous wind selector on Page 1.
WIND_FUEL_CODE = "WND"

#: EIA-860 / USWTDB sentinel for "not available" (USWTDB uses -9999; EIA uses
#: blanks and a literal ".").
USWTDB_MISSING = -9999.0

#: Month-number → EIA-923 ``Netgen_<Month>`` suffix (the schedule spells the
#: month name in full, e.g. ``Netgen_January``).
_MONTH_NAME = {i: calendar.month_name[i] for i in range(1, 13)}


def _plant_id(series: pd.Series) -> pd.Series:
    """Canonicalise an EIA plant code to an integer string.

    Plant codes are integers, but pandas reads a sheet's ``Plant Code`` as
    ``float`` when any cell is blank (so ``1`` becomes ``1.0``) and as ``int``
    when none is — the same code then stringifies to ``"1.0"`` on one sheet and
    ``"1"`` on another, and the join silently finds nothing. Route every plant
    code through here so ``1``, ``1.0`` and ``"1"`` all become ``"1"``; codes
    that are not numeric become ``<NA>`` and drop out of any inner join.
    """
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)


def _to_number(series: pd.Series) -> pd.Series:
    """Coerce an EIA numeric column to float.

    EIA text exports carry thousands separators and use ``.`` for a withheld
    or not-applicable value; both must become NaN rather than 0 or a parse
    error.
    """
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({".": "", "": ""})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def wind_generation_from_eia923(page1: pd.DataFrame) -> pd.DataFrame:
    """Reshape EIA-923 Page 1 into long per-(plant, year, month) net generation.

    Selects wind rows by fuel code (``Reported Fuel Type Code == 'WND'`` — the
    unambiguous selector; the prime mover ``WT`` agrees but the fuel code is
    the schedule's own wind key), melts the twelve ``Netgen_<Month>`` columns,
    and sums to one value per (plant, year, month) in case a plant carries more
    than one wind row.

    Args:
        page1: The "Page 1 Generation and Fuel Data" sheet, header already
            resolved by the caller. Uses ``Plant Id``, ``Reported Fuel Type
            Code``, ``YEAR``, ``Respondent Frequency`` and the twelve
            ``Netgen_<Month>`` columns.

    Returns:
        Long frame with ``ID`` (plant code, str), ``year``, ``month``,
        ``net_gen_mwh`` (float, may be negative — net of station use), and
        ``respondent_frequency`` (single flag per plant-year, upper-cased).

    Raises:
        ValueError: If required columns are absent, or a plant-year mixes
            annual and monthly respondent flags (which must not happen and
            would make the imputation screen ambiguous).
    """
    # The real EIA-923 workbook spells multi-word headers across two lines, so
    # the raw columns arrive as "Reported\nFuel Type Code", "Respondent\n
    # Frequency", "Netgen\nJanuary". Collapse any whitespace run to a single
    # space so the contract below reads the same on the real file and on the
    # single-line synthetic fixtures.
    page1 = page1.rename(
        columns={c: " ".join(str(c).split()) for c in page1.columns}
    )

    required = {"Plant Id", "Reported Fuel Type Code", "YEAR"}
    missing = required - set(page1.columns)
    if missing:
        raise ValueError(f"EIA-923 Page 1 is missing required columns {sorted(missing)}")

    netgen_cols = {
        m: _find_netgen_column(page1.columns, name) for m, name in _MONTH_NAME.items()
    }

    wind = page1[
        page1["Reported Fuel Type Code"].astype(str).str.strip().str.upper()
        == WIND_FUEL_CODE
    ].copy()
    wind["ID"] = _plant_id(wind["Plant Id"])
    wind["year"] = pd.to_numeric(wind["YEAR"], errors="coerce").astype("Int64")

    freq_col = "Respondent Frequency"
    if freq_col in wind.columns:
        wind["respondent_frequency"] = wind[freq_col].astype(str).str.strip().str.upper()
    else:
        # Older vintages omit the flag; treat as unknown, not annual, so the
        # screen defaults to keeping data (the source's drop flag still applies
        # only to explicit "A").
        wind["respondent_frequency"] = "M"

    long_parts = []
    for month, col in netgen_cols.items():
        part = wind[["ID", "year", "respondent_frequency"]].copy()
        part["month"] = month
        part["net_gen_mwh"] = _to_number(wind[col])
        long_parts.append(part)
    long = pd.concat(long_parts, ignore_index=True)

    # One flag per plant-year: mixing A and M within a plant-year is invalid.
    flags = long.groupby(["ID", "year"])["respondent_frequency"].nunique()
    if (flags > 1).any():
        bad = flags[flags > 1].index.tolist()
        raise ValueError(
            f"plant-years carry conflicting respondent frequencies: {bad}"
        )

    grouped = (
        long.groupby(["ID", "year", "month"], as_index=False)
        .agg(
            net_gen_mwh=("net_gen_mwh", "sum"),
            respondent_frequency=("respondent_frequency", "first"),
        )
    )
    # A plant-year present only as all-missing melts to NaN sums via min_count
    # semantics; keep them (finalisation makes them NaN CF) but a genuine 0 is
    # kept as 0. groupby sum returns 0 for all-NaN, so re-mask:
    all_nan = (
        long.groupby(["ID", "year", "month"])["net_gen_mwh"]
        .apply(lambda s: s.isna().all())
        .reset_index(name="_all_nan")
    )
    grouped = grouped.merge(all_nan, on=["ID", "year", "month"], how="left")
    grouped.loc[grouped["_all_nan"], "net_gen_mwh"] = float("nan")
    return grouped.drop(columns="_all_nan")


def _find_netgen_column(columns, month_name: str) -> str:
    """Locate the ``Netgen_<Month>`` column tolerantly.

    EIA exports vary the separator (``Netgen_January``, ``Netgen January``,
    a newline in the xlsx header). Match on the ``netgen`` token plus the
    month name, case-insensitively.
    """
    target = month_name.lower()
    for col in columns:
        c = str(col).lower()
        if "netgen" in c and target in c:
            return col
    raise ValueError(f"no Netgen column found for {month_name!r} in {list(columns)}")


def wind_capacity_from_eia860(
    generators: pd.DataFrame,
    plants: pd.DataFrame,
) -> pd.DataFrame:
    """Sum EIA-860 wind nameplate to plant capacity and attach coordinates.

    Args:
        generators: EIA-860 generator table (the "Operable" sheet), header
            resolved. Uses ``Plant Code``, ``Prime Mover``, ``Nameplate
            Capacity (MW)`` and, when present, ``Operating Year`` /
            ``Operating Month`` (earliest across a plant's wind generators is
            the commissioning date).
        plants: EIA-860 plant table. Uses ``Plant Code``, ``Latitude``,
            ``Longitude`` and, when present, ``Plant Name``.

    Returns:
        One row per plant with ``ID`` (plant code, str), ``site_name``,
        ``lon``, ``lat``, ``capacity_mw`` (summed wind nameplate) and
        ``commissioning_date`` (earliest operating month, or NaT).

    Raises:
        ValueError: If required columns are absent.
    """
    greq = {"Plant Code", "Prime Mover", "Nameplate Capacity (MW)"}
    gmiss = greq - set(generators.columns)
    if gmiss:
        raise ValueError(f"EIA-860 generators missing columns {sorted(gmiss)}")
    preq = {"Plant Code", "Latitude", "Longitude"}
    pmiss = preq - set(plants.columns)
    if pmiss:
        raise ValueError(f"EIA-860 plants missing columns {sorted(pmiss)}")

    gen = generators.copy()
    gen["ID"] = _plant_id(gen["Plant Code"])
    wind = gen[gen["Prime Mover"].astype(str).str.strip().str.upper() == "WT"].copy()
    wind["capacity_mw"] = _to_number(wind["Nameplate Capacity (MW)"])

    if {"Operating Year", "Operating Month"}.issubset(wind.columns):
        oy = pd.to_numeric(wind["Operating Year"], errors="coerce")
        om = pd.to_numeric(wind["Operating Month"], errors="coerce").fillna(1)
        wind["op_date"] = pd.to_datetime(
            dict(year=oy, month=om.clip(1, 12), day=1), errors="coerce"
        )
    else:
        wind["op_date"] = pd.NaT

    per_plant = (
        wind.groupby("ID")
        .agg(
            capacity_mw=("capacity_mw", "sum"),
            commissioning_date=("op_date", "min"),
        )
        .reset_index()
    )

    pl = plants.copy()
    pl["ID"] = _plant_id(pl["Plant Code"])
    pl["lat"] = _to_number(pl["Latitude"])
    pl["lon"] = _to_number(pl["Longitude"])
    name_col = "Plant Name" if "Plant Name" in pl.columns else None
    keep = ["ID", "lon", "lat"] + ([name_col] if name_col else [])
    per_plant = per_plant.merge(pl[keep], on="ID", how="left")
    if name_col:
        per_plant = per_plant.rename(columns={name_col: "site_name"})
    else:
        per_plant["site_name"] = per_plant["ID"]
    return per_plant


def plant_hub_heights_from_uswtdb(uswtdb: pd.DataFrame) -> pd.DataFrame:
    """Aggregate USWTDB turbine attributes to the plant (EIA id).

    Hub height and rotor diameter are capacity-weighted means over the plant's
    turbines that carry a valid value; the model is the manufacturer/model with
    the largest installed capacity at the plant (the modal fleet by MW). Rows
    with a missing ``eia_id`` (turbines not yet linked to an EIA plant) are
    dropped — they cannot be joined to generation.

    Args:
        uswtdb: USWTDB rows. Uses ``eia_id``, ``t_cap`` (turbine capacity, kW),
            ``t_hh`` (hub height, m), ``t_rd`` (rotor diameter, m), ``t_manu``,
            ``t_model``. Missing numerics are the USWTDB ``-9999`` sentinel.

    Returns:
        One row per plant with ``ID`` (eia id, str), ``hub_height_m`` (float or
        NaN), ``rotor_diameter_m`` (float or NaN), ``uswtdb_model`` (str or
        ""), and ``n_turbines`` (count of linked turbines).
    """
    req = {"eia_id", "t_cap"}
    missing = req - set(uswtdb.columns)
    if missing:
        raise ValueError(f"USWTDB is missing required columns {sorted(missing)}")

    df = uswtdb.copy()
    df = df[df["eia_id"].notna()].copy()
    df["ID"] = _plant_id(df["eia_id"])
    df = df[df["ID"] != "<NA>"]

    for col in ("t_cap", "t_hh", "t_rd"):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            df[col] = v.where(v != USWTDB_MISSING)
        else:
            df[col] = float("nan")

    def _weighted(group: pd.DataFrame, value_col: str) -> float:
        v = group[value_col]
        w = group["t_cap"].where(group["t_cap"] > 0)
        mask = v.notna() & w.notna()
        if not mask.any():
            # No capacity weights available: fall back to a plain mean of the
            # values that exist, rather than dropping the plant's height.
            return float(v.dropna().mean()) if v.notna().any() else float("nan")
        return float((v[mask] * w[mask]).sum() / w[mask].sum())

    rows = []
    for plant_id, group in df.groupby("ID"):
        model = ""
        if "t_model" in group.columns and group["t_model"].notna().any():
            manu = group.get("t_manu", pd.Series("", index=group.index)).fillna("")
            label = (manu.astype(str).str.strip() + " " + group["t_model"].astype(str).str.strip()).str.strip()
            by_cap = (
                pd.DataFrame({"label": label, "cap": group["t_cap"].fillna(0.0)})
                .groupby("label")["cap"]
                .sum()
            )
            by_cap = by_cap[by_cap.index.str.len() > 0]
            if len(by_cap):
                model = by_cap.idxmax()
        rows.append(
            {
                "ID": plant_id,
                "hub_height_m": _weighted(group, "t_hh"),
                "rotor_diameter_m": _weighted(group, "t_rd"),
                "uswtdb_model": model,
                "n_turbines": int(len(group)),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["ID", "hub_height_m", "rotor_diameter_m", "uswtdb_model", "n_turbines"],
    )


def filter_to_bbox(
    metadata: pd.DataFrame, bbox: tuple[float, float, float, float]
) -> pd.DataFrame:
    """Drop plants outside the reanalysis domain.

    EIA-923/860 cover every US state, but a region's ERA5 box does not: the
    ``us.toml`` box is CONUS, so Alaska, Hawaii and Puerto Rico plants fall
    outside it. This matters because
    :func:`vwf.wind.aggregate_turbines_to_grid` assigns each turbine its
    nearest grid cell with an UNBOUNDED ``argmin`` — a Hawaii plant is snapped
    to the westernmost CONUS cell ~2000 km away and silently simulated with
    Californian wind. Screening here keeps the fleet and the domain consistent
    instead of relying on that fallback.

    Args:
        metadata: Plant metadata carrying ``lon``/``lat``.
        bbox: ``(lon_min, lon_max, lat_min, lat_max)``, the region's ERA5 box.

    Returns:
        Metadata restricted to plants inside the box (inclusive edges).
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    inside = (
        metadata["lon"].between(lon_min, lon_max)
        & metadata["lat"].between(lat_min, lat_max)
    )
    return metadata[inside].reset_index(drop=True)


def bin_hub_heights(metadata: pd.DataFrame, bin_m: float) -> pd.DataFrame:
    """Round hub heights onto a fixed grid, to bound the simulation's memory.

    :func:`vwf.wind.interpolate_wind` builds a wind field with ONE HEIGHT LEVEL
    PER UNIQUE HUB HEIGHT (``time x lat x lon x height``) before interpolating
    to turbine points. USWTDB capacity-weighted heights are continuous, so the
    real US fleet carries 233 distinct values — a ~51 GB array that is killed
    on any normal machine. Rounding to 10 m leaves 12 levels (~2.6 GB).

    This is the project's own established granularity:
    :func:`vwf.wind.aggregate_turbines_to_grid` already bins heights to 10 m
    (``height_bin``). Log-law wind speed varies by well under 1% across a +/-5 m
    hub-height shift, so this is a memory bound, not a modelling change.

    Args:
        metadata: Plant metadata carrying ``height`` (m).
        bin_m: Bin width in metres. Non-positive disables binning.

    Returns:
        Metadata with ``height`` snapped to the nearest multiple of ``bin_m``.
    """
    if bin_m is None or bin_m <= 0:
        return metadata
    out = metadata.copy()
    out["height"] = (out["height"] / bin_m).round() * bin_m
    return out


def build_us_metadata(
    capacity: pd.DataFrame,
    hub_heights: pd.DataFrame | None,
    *,
    default_height: float,
    model: str,
) -> pd.DataFrame:
    """Emit the ``EIAUSSource`` metadata contract from the joined tables.

    Capacity is the EIA-860 wind nameplate (MW → kW, the source contract unit).
    Hub height is the USWTDB capacity-weighted mean where available, else the
    uniform ``default_height``; the ``height_source`` column records which, per
    plant. ``model`` is a uniform power-curve key (must be a column of your
    ``power_curves.csv``); the USWTDB manufacturer/model string travels in
    ``uswtdb_model`` for the future vintage-aware assignment but never becomes
    the curve key itself (that key must exist in the curve library).

    Args:
        capacity: Output of :func:`wind_capacity_from_eia860`.
        hub_heights: Output of :func:`plant_hub_heights_from_uswtdb`, or None to
            use the default height for every plant.
        default_height: Uniform hub-height fallback, metres.
        model: Uniform power-curve key.

    Returns:
        The metadata contract: ``ID``, ``site_name``, ``lon``, ``lat``,
        ``height``, ``capacity`` (kW), ``model``, ``type``,
        ``commissioning_date``, plus the provenance columns ``height_source``,
        ``model_source``, ``uswtdb_model``, ``n_turbines``. Plants with no
        coordinates are dropped (they cannot be simulated); the count is the
        caller's to report.
    """
    md = capacity.copy()
    md["ID"] = md["ID"].astype(str)

    if hub_heights is not None and len(hub_heights):
        hh = hub_heights.copy()
        hh["ID"] = hh["ID"].astype(str)
        md = md.merge(
            hh[["ID", "hub_height_m", "uswtdb_model", "n_turbines"]],
            on="ID",
            how="left",
        )
    else:
        md["hub_height_m"] = float("nan")
        md["uswtdb_model"] = ""
        md["n_turbines"] = 0

    has_height = md["hub_height_m"].notna()
    md["height"] = md["hub_height_m"].where(has_height, float(default_height))
    md["height_source"] = has_height.map(
        {True: "uswtdb-capacity-weighted", False: "default-uniform"}
    )
    md["capacity"] = _to_number(md["capacity_mw"]) * 1000.0
    md["model"] = model
    md["model_source"] = "default-uniform"
    md["type"] = "onshore"  # US utility-scale wind is onshore over the historical window
    md["uswtdb_model"] = md["uswtdb_model"].fillna("")
    md["n_turbines"] = md["n_turbines"].fillna(0).astype(int)

    out = md[
        [
            "ID",
            "site_name",
            "lon",
            "lat",
            "height",
            "capacity",
            "model",
            "type",
            "commissioning_date",
            "height_source",
            "model_source",
            "uswtdb_model",
            "n_turbines",
        ]
    ].copy()
    out = out[out["lon"].notna() & out["lat"].notna()].reset_index(drop=True)
    return out
