"""Transforms for building the Chile (Coordinador / CEN) inputs.

Pure frame-to-frame logic for ``scripts/process/cen_cl.py``: reshaping the CEN
``generacion-real`` hourly series into monthly capacity factors, deriving the
wind fleet from that same stream, and emitting the source metadata contract
once coordinates are joined from an external register.

Everything here takes and returns DataFrames so it is testable without the raw
CEN JSON; file I/O lives in the script.

The unit of observation is the **plant** (``obs_unit = "plant"``): CEN's SIP
service reports generation per *central* (``id_central``), and the generation
rows carry the plant name, owner, technology, and ``potencia_maxima`` (the
plant's maximum power, MW), so the fleet and its capacities come from the
generation stream itself, exactly as Brazil's ONS ``FATOR_CAPACIDADE`` is
self-contained. What CEN does NOT expose anywhere is coordinates (the
``/centrales`` and Infotécnica registries advertise UTM fields but leave them
blank), so lon/lat must be joined from the Global Wind Power Tracker.

Two facts about the CEN series shape the design, verified against the real
2021-2024 data (docs/runbooks/cl.md):

- **Fixed offset, no DST.** ``fecha_hora`` runs a clean 24 hours every day,
  including Chile's 2024 spring-forward (7 Sep 2024 has 24 rows, not 23), so
  CEN publishes in fixed Chilean standard time, not civil ``America/Santiago``.
  The conversion to UTC is therefore a single fixed shift with no DST branch,
  simpler than the AEMO/EMI cases. The offset value (UTC-4 continental
  standard) is immaterial to a monthly mean, which spans ~720 hours; it only
  moves a few edge hours across month boundaries, and it is applied only to
  keep the bins consistent with the UTC-binned ERA5/simulation side.
- **Static nameplate.** No wind plant's ``potencia_maxima`` varies across the
  window (checked: 0 of 64), so the CF denominator is a plain per-plant
  constant. It is still divided per hour, which stays correct if a future
  edition adds a ramping plant.
"""
from __future__ import annotations

from calendar import monthrange
from datetime import timedelta, timezone
from collections.abc import Sequence

import pandas as pd

#: CEN generation timestamps are fixed Chilean standard time (UTC-4), with NO
#: daylight saving applied (verified on the 2024 spring-forward day). Adding
#: four hours takes them to UTC, matching the ERA5/simulation convention.
CHILE_STD = timezone(timedelta(hours=-4))

#: ``tipo_tecnologia`` value selecting wind plants. MUST carry the accent: the
#: unaccented ``"Eolica"`` silently matches nothing in the CEN data.
WIND = "eólica"

#: Months with less than this fraction of expected hourly samples are NaN.
DEFAULT_MIN_COVERAGE = 0.9


def _local_to_utc(timestamps: pd.Series) -> pd.Series:
    """Convert naive Chilean-standard (UTC-4) timestamps to naive UTC."""
    ts = pd.to_datetime(timestamps)
    if getattr(ts.dt, "tz", None) is not None:
        raise ValueError(
            "CEN timestamps must be naive Chilean standard time; got tz-aware"
        )
    return ts.dt.tz_localize(CHILE_STD).dt.tz_convert("UTC").dt.tz_localize(None)


def wind_rows(gen: pd.DataFrame) -> pd.DataFrame:
    """Select the wind rows of a CEN generation frame, accent/case tolerant.

    Matched on the exact technology label lower-cased. A substring test on
    ``"lica"`` would also catch ``"Hidráulica"`` (the overcount trap the Chile
    verification hit), so this is an equality test, not ``contains``.
    """
    tipo = gen["tipo_tecnologia"].astype(str).str.strip().str.lower()
    return gen[tipo == WIND].copy()


def wind_fleet_from_generation(gen: pd.DataFrame) -> pd.DataFrame:
    """Derive per-plant metadata from the CEN generation frame itself.

    Args:
        gen: One or more concatenated ``generacion-real`` frames (any
            technologies; this filters to wind). Uses ``id_central``,
            ``central``, ``propietario``, ``potencia_maxima``,
            ``subtipo_tecnologia``.

    Returns:
        One row per ``ID`` (``id_central`` as string) with ``site_name``,
        ``propietario``, ``capacity_mw`` (max ``potencia_maxima`` over the
        frame), ``subtipo``.
    """
    wind = wind_rows(gen)
    wind["ID"] = wind["id_central"].astype(str).str.strip()
    wind["cap_mw"] = pd.to_numeric(wind["potencia_maxima"], errors="coerce")
    grouped = wind.groupby("ID").agg(
        site_name=("central", "first"),
        propietario=("propietario", "first"),
        capacity_mw=("cap_mw", "max"),
        subtipo=("subtipo_tecnologia", "first"),
    ).reset_index()
    return grouped


def monthly_cf_from_generation(
    gen: pd.DataFrame,
    year_start: int,
    year_end: int,
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> pd.DataFrame:
    """Monthly-mean capacity factor from the hourly CEN generation series.

    Each hour's CF is ``gen_real_mw / potencia_maxima``; the monthly value is
    the mean of those over the month's UTC-binned hours. Months with less than
    ``min_coverage`` of their expected hourly samples are NaN rather than
    biased by a gap. Rows with a non-positive or missing ``potencia_maxima``
    are dropped (they carry no usable denominator).

    Args:
        gen: Concatenated ``generacion-real`` frames. Uses ``id_central``,
            ``tipo_tecnologia``, ``fecha_hora`` (naive Chilean standard),
            ``gen_real_mw``, ``potencia_maxima``.
        year_start, year_end: Inclusive UTC year bounds.
        min_coverage: Minimum fraction of a month's hourly samples required.

    Returns:
        Wide frame ``ID``, ``year``, ``obs_1``..``obs_12`` (monthly mean CF;
        missing months NaN).
    """
    wind = wind_rows(gen)
    wind["ID"] = wind["id_central"].astype(str).str.strip()
    cap = pd.to_numeric(wind["potencia_maxima"], errors="coerce")
    gen_mw = pd.to_numeric(wind["gen_real_mw"], errors="coerce")
    wind = wind[cap.notna() & (cap > 0) & gen_mw.notna()].copy()
    wind["cf"] = gen_mw[wind.index] / cap[wind.index]
    wind["timestamp"] = _local_to_utc(wind["fecha_hora"])
    wind["year"] = wind["timestamp"].dt.year
    wind["month"] = wind["timestamp"].dt.month
    wind = wind[(wind["year"] >= int(year_start)) & (wind["year"] <= int(year_end))]
    if wind.empty:
        return pd.DataFrame(
            columns=["ID", "year"] + [f"obs_{m}" for m in range(1, 13)]
        )

    agg = (
        wind.groupby(["ID", "year", "month"])
        .agg(cf=("cf", "mean"), n=("cf", "count"))
        .reset_index()
    )
    expected = agg.apply(
        lambda r: monthrange(int(r["year"]), int(r["month"]))[1] * 24.0, axis=1
    )
    agg.loc[agg["n"] / expected < min_coverage, "cf"] = float("nan")

    wide = (
        agg.pivot(index=["ID", "year"], columns="month", values="cf")
        .reindex(columns=range(1, 13))
        .reset_index()
    )
    wide.columns = ["ID", "year"] + [f"obs_{m}" for m in range(1, 13)]
    return wide


def strip_commissioning_prefix(
    wide: pd.DataFrame, *, threshold: float = 0.02
) -> pd.DataFrame:
    """NaN each plant's leading months before it is operational.

    ``potencia_maxima`` is the full nameplate from the first row, but CEN
    reports pre-commissioning months as ~0 MW, so those months have monthly CF
    near zero and are NOT resource signal: the static-nameplate ramp trap the
    AU and US adapters guard against. This strips the leading run of months
    with CF below ``threshold`` (per plant, up to its first operational month);
    later sub-threshold months are KEPT, since after commissioning a near-zero
    month is a real outage or curtailment event in the delivered CF, not a
    fleet-entry artefact.

    This does NOT touch curtailment in operational months. The CEN
    ``gen_real_mw`` is *delivered* energy, and Chile's northern grid curtails
    wind heavily ("vertimiento"), so the observed CF understates the resource
    where curtailment bites: the same standing caveat as Brazil's Nordeste.
    Screening that needs a CEN wind-curtailment series (not wired here).

    Args:
        wide: ``ID``, ``year``, ``obs_1``..``obs_12`` frame.
        threshold: monthly CF below which a leading month is pre-operational.

    Returns:
        The frame with leading pre-operational months set to NaN.
    """
    out = wide.sort_values(["ID", "year"]).copy()
    for pid, block in out.groupby("ID"):
        # flatten to a chronological (year, month) series
        series = []
        for _, row in block.iterrows():
            for m in range(1, 13):
                series.append((row.name, f"obs_{m}", row[f"obs_{m}"]))
        # walk forward, NaNing until the first month at/above threshold
        for idx, col, val in series:
            if pd.notna(val) and val >= threshold:
                break
            if pd.notna(val):
                out.at[idx, col] = float("nan")
    return out


def build_cl_metadata(
    fleet: pd.DataFrame,
    coords: pd.DataFrame,
    *,
    height: float,
    model: str,
    exclude: Sequence[str] = (),
) -> pd.DataFrame:
    """Emit the CENChileSource metadata contract from the fleet + coordinates.

    Coordinates are NOT in the CEN API, so they are joined from ``coords`` (a
    curated frame of ``ID``, ``lon``, ``lat`` built from the Global Wind Power
    Tracker in the processing step). ``height``/``model`` are uniform defaults
    recorded in ``*_source`` columns (CEN carries no hub height or turbine
    model: the same vintage-aware follow-up as AU/BR/US).

    A plant that survives to here with no coordinate is a hard error, not a
    silent drop: without lon/lat it would be simulated at the wrong location or
    quietly dropped from the fleet. Plants intentionally out of scope must be
    named in ``exclude`` (e.g. the isolated Magallanes system south of -44).

    Args:
        fleet: Output of :func:`wind_fleet_from_generation`.
        coords: Frame with ``ID``, ``lon``, ``lat``.
        height: Uniform hub-height default, metres.
        model: Uniform power-curve key (a column of ``power_curves.csv``).
        exclude: Plant IDs allowed to have no coordinate (dropped, logged).

    Returns:
        ``ID``, ``site_name``, ``lon``, ``lat``, ``height``, ``capacity`` (kW),
        ``model``, ``type``, plus provenance ``height_source``,
        ``model_source``, ``propietario``.
    """
    md = fleet.copy()
    md["ID"] = md["ID"].astype(str)
    coords = coords.copy()
    coords["ID"] = coords["ID"].astype(str)
    md = md.merge(coords[["ID", "lon", "lat"]], on="ID", how="left")

    missing = md[md["lon"].isna() | md["lat"].isna()]
    missing = missing[~missing["ID"].isin(set(exclude))]
    if len(missing):
        raise ValueError(
            f"{len(missing)} Chile wind plants have no coordinate after the "
            f"GWPT join and are not in the exclude list: "
            f"{missing['site_name'].tolist()[:10]}. Add them to the coordinate "
            "override table or the exclude list; never leave a plant "
            "coordinate-less, it would be mis-located silently."
        )
    md = md[md["lon"].notna() & md["lat"].notna()].copy()

    md["height"] = float(height)
    md["height_source"] = "default-uniform"
    md["model_source"] = "default-uniform"
    md["model"] = model
    md["capacity"] = pd.to_numeric(md["capacity_mw"], errors="coerce") * 1000.0
    md["type"] = "onshore"
    return md[
        ["ID", "site_name", "lon", "lat", "height", "capacity", "model",
         "type", "height_source", "model_source", "propietario"]
    ].reset_index(drop=True)
