"""Transforms for building the New Zealand (EMI) inputs from the public data.

Pure frame-to-frame logic for ``scripts/process/emi_nz.py``: mapping EMI
trading periods to UTC timestamps, reshaping the half-hourly ``Generation_MD``
energy series into monthly capacity factors, and building the capacity
history they are divided by.

Everything here takes and returns DataFrames so it is testable without the raw
EMI files; file I/O lives in the script.

The unit of observation is the **farm** (``obs_unit = "farm"``): EMI reports
metered grid injection per generating plant (keyed by point of connection +
unit), and the curated farm table (``configs/curation/nz_wind_farms.csv``) maps those
keys onto wind farms with coordinates, turbine model, and hub height. New
Zealand's fleet is small (~20 farms) but the registry-grade metadata is
hand-compiled, which is what makes it one of the few regions outside Europe
with per-farm hub heights.

Two NZ-specific facts shape the design:

- **Trading periods are civil time with DST.** A trading day normally has 48
  half-hourly periods (TP1 starts at 00:00 New Zealand civil time), but the
  spring-forward day has 46 and the fall-back day has 50. Mapping ``(trading
  date, TP)`` to UTC therefore goes through the ``Pacific/Auckland`` zone: the
  UTC instant of local midnight plus ``(TP - 1) x 30`` minutes. That is exact
  on DST days by construction (no period is skipped or double-counted),
  and the monthly bins downstream are UTC, matching the ERA5/simulation
  convention everywhere else (``time_convention = "utc-monthly-bins"``).
- **Capacity changes over time.** The monthly CF is computed against a
  then-current capacity, which keeps ramping builds (Turitea 2021-2023,
  Harapaki 2024) from biasing CF low. The processing step builds that history
  from the curated tables (``nz_capacity_stages.csv``, then each farm's final
  capacity from ``nz_wind_farms.csv``) and masks the commissioning windows in
  ``nz_mask_windows.csv``. The register-based alternatives here,
  :func:`capacity_history_from_register` and :func:`below_final_build_mask`,
  derive the same things from the EMI dispatched-plant register's
  effective-dated nameplate rows. They are tested but not used by the
  processing step, which uses :func:`capacity_history_from_curation` and
  :func:`mask_from_windows`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from vwf.time_utils import month_days

#: New Zealand civil time. EMI trading dates and periods are defined in this
#: zone (NZST/NZDT); the zone database handles the DST transitions.
NZ_TZ = ZoneInfo("Pacific/Auckland")

#: Months with less than this fraction of expected half-hourly samples are NaN.
DEFAULT_MIN_COVERAGE = 0.9

#: Below-final-build mask threshold: months where the then-current registered
#: capacity is under this fraction of the final build are listed for masking.
DEFAULT_BUILD_THRESHOLD = 0.95


def day_start_utc(trading_dates: pd.Series) -> pd.Series:
    """UTC instant of local midnight for each NZ trading date.

    Args:
        trading_dates: Date-like series (naive dates or date strings).

    Returns:
        Naive-UTC timestamps of 00:00 New Zealand civil time on each date.
    """
    dates = pd.to_datetime(trading_dates)
    if getattr(dates.dt, "tz", None) is not None:
        raise ValueError("trading dates must be naive local dates, got tz-aware")
    return dates.dt.tz_localize(NZ_TZ).dt.tz_convert("UTC").dt.tz_localize(None)


def expected_trading_periods(trading_dates: pd.Series) -> pd.Series:
    """Number of trading periods in each NZ trading day (46, 48, or 50).

    Derived from the UTC span between consecutive local midnights, so the DST
    transition days come out at 46 (spring forward) and 50 (fall back) without
    any hardcoded transition dates.
    """
    dates = pd.to_datetime(trading_dates)
    start = day_start_utc(dates)
    end = day_start_utc(dates + pd.Timedelta(days=1))
    return ((end - start) / pd.Timedelta(minutes=30)).astype(int)


def trading_period_start_utc(trading_dates: pd.Series, trading_periods: pd.Series) -> pd.Series:
    """UTC start instant of each ``(trading date, trading period)`` pair.

    TP ``n`` is the ``n``-th half hour of the local day counted in elapsed
    time from local midnight, which is exactly ``midnight_utc + (n-1) x 30
    min``, valid across DST transitions because elapsed UTC time is what the
    trading-period sequence counts.
    """
    tp = pd.to_numeric(trading_periods, errors="raise").astype(int)
    if (tp < 1).any() or (tp > 50).any():
        bad = sorted(tp[(tp < 1) | (tp > 50)].unique().tolist())
        raise ValueError(f"trading periods out of range 1..50: {bad}")
    return day_start_utc(trading_dates) + pd.to_timedelta((tp - 1) * 30, unit="m")


def half_hourly_from_generation_md(
    gen: pd.DataFrame,
    *,
    id_col: str,
    date_col: str = "Trading_Date",
    tp_prefix: str = "TP",
) -> pd.DataFrame:
    """Melt a wide EMI ``Generation_MD`` frame to half-hourly UTC energy rows.

    Args:
        gen: Frame with one row per (plant key, trading date) and one column
            per trading period (``TP1``..``TP50``) holding kWh. Non-existent
            periods on a given day (TP47+ on normal days, TP49/50 except on
            the fall-back day) must be NaN/empty; a value in a period beyond
            the day's actual count raises, because it means the TP-to-time
            mapping assumption is broken.
        id_col: Column holding the plant key to carry through.
        date_col: Column holding the trading date.
        tp_prefix: Prefix of the trading-period columns.

    Returns:
        Long frame with ``ID``, ``timestamp`` (naive UTC, period start), and
        ``kwh``. NaN periods (no data / non-existent) are dropped.
    """
    tp_cols = [c for c in gen.columns if c.startswith(tp_prefix) and c[len(tp_prefix) :].isdigit()]
    if not tp_cols:
        raise ValueError(f"no {tp_prefix}<n> trading-period columns found")

    long = gen.melt(
        id_vars=[id_col, date_col],
        value_vars=tp_cols,
        var_name="tp",
        value_name="kwh",
    )
    long["tp"] = long["tp"].str[len(tp_prefix) :].astype(int)
    long["kwh"] = pd.to_numeric(long["kwh"], errors="coerce")
    long = long[long["kwh"].notna()].copy()

    n_periods = expected_trading_periods(long[date_col])
    overrun = long["tp"] > n_periods
    if overrun.any():
        sample = long.loc[overrun, [id_col, date_col, "tp"]].head()
        raise ValueError(
            "generation reported in trading periods beyond the day's length "
            f"(DST mapping would be wrong):\n{sample}"
        )

    long["timestamp"] = trading_period_start_utc(long[date_col], long["tp"])
    out = long.rename(columns={id_col: "ID"})[["ID", "timestamp", "kwh"]]
    out["ID"] = out["ID"].astype(str)
    return out.reset_index(drop=True)


def capacity_history_from_register(
    register: pd.DataFrame,
    farm_of_key: pd.Series,
    *,
    key_col: str,
    capacity_col: str,
    start_col: str,
    end_col: str,
) -> pd.DataFrame:
    """Reduce the effective-dated EMI plant register to a farm capacity history.

    Args:
        register: EMI dispatched-plant register rows (one per unit and
            validity window).
        farm_of_key: Mapping from register plant key to farm ``ID`` (index =
            key, value = farm ID). Register keys not in the mapping are
            ignored (non-wind plant).
        key_col: Register column holding the plant key.
        capacity_col: Register column holding nameplate capacity in **MW**.
        start_col: Validity start date column.
        end_col: Validity end date column (NaN = open).

    Returns:
        Frame with ``ID``, ``effective_from`` (naive UTC midnight), and
        ``capacity`` (kW): the farm's total registered capacity from each
        change point onward.
    """
    reg = register.copy()
    reg["farm"] = reg[key_col].astype(str).map(farm_of_key)
    reg = reg[reg["farm"].notna()].copy()
    reg["mw"] = pd.to_numeric(reg[capacity_col], errors="coerce").fillna(0.0)
    reg["start"] = pd.to_datetime(reg[start_col])
    reg["end"] = pd.to_datetime(reg[end_col], errors="coerce")

    events: list[pd.DataFrame] = []
    for farm, rows in reg.groupby("farm"):
        points = pd.concat([rows["start"], rows["end"].dropna()]).sort_values()
        points = points.drop_duplicates().reset_index(drop=True)
        caps = []
        for t in points:
            live = (rows["start"] <= t) & (rows["end"].isna() | (rows["end"] > t))
            caps.append(rows.loc[live, "mw"].sum() * 1000.0)
        frame = pd.DataFrame({"ID": str(farm), "effective_from": points, "capacity": caps})
        # Collapse consecutive points with unchanged capacity.
        frame = frame[frame["capacity"].ne(frame["capacity"].shift())]
        events.append(frame)
    if not events:
        return pd.DataFrame(columns=["ID", "effective_from", "capacity"])
    return pd.concat(events, ignore_index=True)


def _capacity_at(history: pd.DataFrame, ids: pd.Series, timestamps: pd.Series) -> pd.Series:
    """Then-current capacity (kW) for each (ID, timestamp) pair."""
    lookup = history.sort_values("effective_from")
    frame = pd.DataFrame(
        {
            "ID": ids.astype(str).values,
            "timestamp": pd.to_datetime(timestamps).values,
        }
    )
    frame["_row"] = range(len(frame))
    merged = pd.merge_asof(
        frame.sort_values("timestamp"),
        lookup.rename(columns={"effective_from": "timestamp"}),
        on="timestamp",
        by="ID",
        direction="backward",
    )
    return merged.sort_values("_row")["capacity"].reset_index(drop=True)


def monthly_cf(
    half_hourly: pd.DataFrame,
    capacity_history: pd.DataFrame,
    year_start: int,
    year_end: int,
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> pd.DataFrame:
    """Aggregate half-hourly energy to monthly mean capacity factor, UTC bins.

    Each half-hour's CF is ``kwh / (capacity_kW x 0.5 h)`` against the
    then-current registered capacity, so ramping builds are not biased low the
    way a static-nameplate denominator would be. Months with less than
    ``min_coverage`` of their expected half-hourly samples are NaN rather than
    biased by a gap; half-hours before a farm's first registered capacity (or
    with zero capacity) are dropped.

    Args:
        half_hourly: Output of :func:`half_hourly_from_generation_md`,
            aggregated to one row per (ID, timestamp); sum multiple units
            of one farm before calling.
        capacity_history: ``ID``, ``effective_from``, ``capacity`` (kW). The
            processing step builds it from the curated tables;
            :func:`capacity_history_from_register` is the register-based
            alternative.
        year_start: First UTC year to include (inclusive).
        year_end: Last UTC year to include (inclusive).
        min_coverage: Minimum fraction of a month's half-hours required.

    Returns:
        Wide frame with ``ID``, ``year``, ``obs_1``..``obs_12`` (monthly mean
        capacity factor; missing months NaN).
    """
    hh = half_hourly.copy()
    hh["ID"] = hh["ID"].astype(str)
    hh["capacity"] = _capacity_at(capacity_history, hh["ID"], hh["timestamp"])
    hh = hh[hh["capacity"].notna() & (hh["capacity"] > 0)].copy()
    hh["cf"] = pd.to_numeric(hh["kwh"], errors="coerce") / (hh["capacity"] * 0.5)
    hh["year"] = hh["timestamp"].dt.year
    hh["month"] = hh["timestamp"].dt.month
    hh = hh[(hh["year"] >= int(year_start)) & (hh["year"] <= int(year_end))]
    if hh.empty:
        return pd.DataFrame(columns=["ID", "year"] + [f"obs_{m}" for m in range(1, 13)])

    agg = (
        hh.groupby(["ID", "year", "month"]).agg(cf=("cf", "mean"), n=("cf", "count")).reset_index()
    )
    expected = month_days(agg["year"], agg["month"]) * 48.0
    agg.loc[agg["n"] / expected < min_coverage, "cf"] = float("nan")

    wide = (
        agg.pivot(index=["ID", "year"], columns="month", values="cf")
        .reindex(columns=range(1, 13))
        .reset_index()
    )
    wide.columns = ["ID", "year"] + [f"obs_{m}" for m in range(1, 13)]
    return wide


def below_final_build_mask(
    capacity_history: pd.DataFrame,
    year_start: int,
    year_end: int,
    *,
    threshold: float = DEFAULT_BUILD_THRESHOLD,
) -> pd.DataFrame:
    """Months where a farm's registered capacity is below its final build.

    The then-current-capacity CF denominator already handles ramping fleets
    arithmetically, but months with turbines still being erected can be
    erratic (partial availability, commissioning tests). This lists every
    ``(ID, year, month)`` whose month-start registered capacity is under
    ``threshold`` of the farm's maximum over the history (the AU
    registered-capacity mask pattern) for the source layer to NaN.

    Returns:
        Frame with ``ID``, ``year``, ``month``.
    """
    rows = []
    final = capacity_history.groupby("ID")["capacity"].max()
    months = pd.date_range(f"{int(year_start)}-01-01", f"{int(year_end)}-12-01", freq="MS")
    for farm, cap_final in final.items():
        starts = pd.Series(months)
        caps = _capacity_at(capacity_history, pd.Series([farm] * len(starts)), starts)
        under = caps.isna() | (caps < float(threshold) * cap_final)
        for t in months[under.values]:
            rows.append({"ID": str(farm), "year": t.year, "month": t.month})
    return pd.DataFrame(rows, columns=["ID", "year", "month"])


# ---------------------------------------------------------------------------
# The production path: the curated tables in configs/curation/
# ---------------------------------------------------------------------------

#: Register-listed wind Gen_Codes that never carry wind rows in Generation_MD,
#: or embedded farms outside the dispatched dataset. Documented exclusions;
#: an unmapped Gen_Code NOT in this set is an error.
KNOWN_ABSENT = frozenset(
    {
        # Mahinerangi is metered inside the Waipori hydro scheme; its output never
        # appears as wind rows in Generation_MD (verified 2013/2025/2026 files).
        "mahinerangi",
    }
)

#: Fuel_Code spellings of wind; the coding changed with Kaiwera Downs 2.
WIND_FUELS = frozenset({"wind", "win"})


def load_curated_tables(configs: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three curated NZ tables.

    Returns:
        ``(farms, stages, windows)`` from ``nz_wind_farms.csv``,
        ``nz_capacity_stages.csv`` and ``nz_mask_windows.csv``, with the farm
        ``ID`` as a string.
    """
    farms = pd.read_csv(configs / "nz_wind_farms.csv")
    farms["ID"] = farms["ID"].astype(str)
    stages = pd.read_csv(configs / "nz_capacity_stages.csv")
    windows = pd.read_csv(configs / "nz_mask_windows.csv")
    return farms, stages, windows


def gen_code_map(farms: pd.DataFrame) -> dict[str, str]:
    """Lower-cased Gen_Code -> farm ID, from the farms table's ``gen_codes``."""
    mapping: dict[str, str] = {}
    for _, row in farms.iterrows():
        for code in str(row["gen_codes"]).split(";"):
            mapping[code.strip().lower()] = row["ID"]
    return mapping


def capacity_history_from_curation(farms: pd.DataFrame, stages: pd.DataFrame) -> pd.DataFrame:
    """Stable-plateau capacity history from the curated tables.

    Curated stages where a farm has them, otherwise one row per farm from its
    first generation at its final capacity. This, not
    :func:`capacity_history_from_register`, is what the NZ inputs are built on.

    Returns:
        Frame with ``ID``, ``effective_from`` and ``capacity`` (kW), sorted by
        farm and date.
    """
    rows = [
        pd.DataFrame(
            {
                "ID": stages["ID"].astype(str),
                "effective_from": pd.to_datetime(stages["effective_from"]),
                "capacity": pd.to_numeric(stages["capacity"]),
            }
        )
    ]
    staged = set(stages["ID"].astype(str))
    plain = farms[~farms["ID"].isin(staged)]
    rows.append(
        pd.DataFrame(
            {
                "ID": plain["ID"],
                "effective_from": pd.to_datetime(plain["first_generation"]),
                "capacity": pd.to_numeric(plain["capacity"]),
            }
        )
    )
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["ID", "effective_from"])
        .reset_index(drop=True)
    )


def mask_from_windows(windows: pd.DataFrame) -> pd.DataFrame:
    """Expand curated ``(from_month, to_month)`` windows to ``(ID, year, month)``."""
    rows = []
    for _, w in windows.iterrows():
        months = pd.period_range(w["from_month"], w["to_month"], freq="M")
        for p in months:
            rows.append({"ID": str(w["ID"]), "year": p.year, "month": p.month})
    return pd.DataFrame(rows, columns=["ID", "year", "month"])


def wind_half_hourly(
    paths: Iterable[Path],
    mapping: dict[str, str],
    *,
    known_absent: frozenset[str] = KNOWN_ABSENT,
) -> tuple[pd.DataFrame | None, set[str]]:
    """Melt every Generation_MD file to half-hourly UTC energy per farm.

    Wind rows are selected by ``Fuel_Code`` and keyed on the case-normalised
    ``Gen_Code``: ``Site_Code`` is not stable across years, and capitalisation
    varies. The header casing drifts too (the 2019 files say
    ``Trading_date``).

    Returns:
        ``(half_hourly, unmapped)``. ``half_hourly`` has ``ID``, ``timestamp``
        and ``kwh`` summed per farm and half-hour, or is None if no file has a
        mapped wind row. ``unmapped`` holds wind Gen_Codes with no farm that
        are not documented absentees; the caller decides that they are an
        error.
    """
    pieces = []
    unmapped: set[str] = set()
    for path in paths:
        gen = pd.read_csv(path, low_memory=False)
        gen = gen.rename(
            columns={c: "Trading_Date" for c in gen.columns if c.lower() == "trading_date"}
        )
        fuel = gen["Fuel_Code"].astype(str).str.strip().str.lower()
        wind = gen[fuel.isin(WIND_FUELS)].copy()
        if wind.empty:
            continue
        codes = wind["Gen_Code"].astype(str).str.strip().str.lower()
        unmapped |= set(codes[~codes.isin(mapping)]) - known_absent
        wind["farm"] = codes.map(mapping)
        wind = wind[wind["farm"].notna()]
        if wind.empty:
            continue
        pieces.append(half_hourly_from_generation_md(wind, id_col="farm"))
    if not pieces:
        return None, unmapped
    half_hourly = (
        pd.concat(pieces, ignore_index=True)
        .groupby(["ID", "timestamp"], as_index=False)["kwh"]
        .sum()
    )
    return half_hourly, unmapped


#: The columns of ``nz_md.csv``, the metadata contract EMINewZealandSource reads.
METADATA_COLUMNS = [
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
    "true_model",
    "n_turbines",
    "diameter",
    "operator",
]


def metadata_contract(farms: pd.DataFrame, fallback_model: str) -> pd.DataFrame:
    """The ``nz_md.csv`` frame: curated farms with a curve key assigned.

    Keys are assigned per farm by scale-then-specific-power matching against
    the active curve library, the guarded matcher the US fleet uses. The true
    manufacturer and model string is carried in ``true_model`` for provenance,
    never as the curve key.
    """
    from vwf.datasets.eia_us import assign_curves_from_library

    md = farms.rename(columns={"turbine_model": "true_model"}).copy()
    md["model"] = fallback_model
    md["model_source"] = "default-uniform"
    md = assign_curves_from_library(md, fallback_model=fallback_model)
    md["commissioning_date"] = pd.to_datetime(md["first_generation"])
    return md[METADATA_COLUMNS]
