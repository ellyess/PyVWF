"""Transforms for building the Brazil (ONS) inputs from the public open data.

Pure frame-to-frame logic for ``scripts/process/ons_br.py``: reshaping the ONS
``FATOR_CAPACIDADE`` hourly capacity-factor series into monthly means, deriving
the complex metadata the same file already carries, and turning the ONS
constrained-off (``RESTRICAO_COFF_EOLICA``) series into a curtailment account.

Everything here takes and returns DataFrames so it is testable without the raw
ONS/ANEEL files; file I/O lives in the script.

The unit of observation is the **complex** (``obs_unit = "complex"``): ONS
reports wind at the *conjunto de usinas* level, one series per ``id_ons`` (e.g.
``CJU_MAPLN``), which aggregates the several ANEEL plants (CEGs) behind one
grid connection. This is the natural ONS unit and the one its capacity factor
is defined against; it parallels the AU adapter treating the farm (DUID) and
the US adapter treating the plant as the unit while the mechanical
``obs_level`` stays "turbine".

Two facts about the ONS FC series shape the design and are documented in the
region config and the join report:

- **Self-contained metadata.** ``FATOR_CAPACIDADE`` carries, per hour and
  complex, the capacity factor (``val_fatorcapacidade``), the *time-varying*
  installed capacity (``val_capacidadeinstalada``), the collector-substation
  coordinates, the name, and the subsystem/state. So the observation *and* the
  simulation metadata come from one file; ANEEL SIGA is optional enrichment
  (commissioning date, turbine models via CEG), not on the critical path — and
  the ONS ``ceg`` field is frequently ``-`` at conjunto level, which is exactly
  the plant-vs-complex mapping problem SIGA would otherwise force.
- **The CF denominator already tracks the build-out.** ONS computes
  ``val_fatorcapacidade`` against the *then-current* installed capacity, so a
  complex ramping up is not biased low the way the AU/US static-nameplate CFs
  are. A below-final-build mask is still offered (for parity, and because the
  first partial month can be erratic), but it is optional here rather than
  structural.

Curtailment is the headline scientific reason for this region (research doc
§1): the constrained-off series lets us *separate curtailment from resource
bias* by energy accounting — ``val_geracao`` is delivered, ``val_geracaolimitada``
is constrained off, so the resource that the wind actually offered is their sum.
Pre-2021 has no constrained-off series, so its CF carries unscreened curtailment
(a documented caveat, and the Nordeste is where it bites hardest).
"""
from __future__ import annotations

from calendar import monthrange
from datetime import timedelta, timezone

import pandas as pd

#: ONS timestamps are Brasília civil time, UTC-3 with no daylight saving over
#: the usable window (DST was abolished in 2019, and the Nordeste wind fleet
#: never observed it). A fixed -3 h shift takes them to UTC, matching the
#: ERA5/simulation convention used everywhere else.
BRASILIA = timezone(timedelta(hours=-3))

#: ONS ``nom_tipousina`` value selecting wind complexes.
WIND_TIPO = "eólica"

#: Months with less than this fraction of expected hourly samples are NaN.
DEFAULT_MIN_COVERAGE = 0.9


def _brasilia_to_utc(timestamps: pd.Series) -> pd.Series:
    """Convert naive Brasília (UTC-3) timestamps to naive UTC."""
    ts = pd.to_datetime(timestamps)
    if getattr(ts.dt, "tz", None) is not None:
        raise ValueError(
            "ONS timestamps must be naive Brasília civil time; got tz-aware input"
        )
    return ts.dt.tz_localize(BRASILIA).dt.tz_convert("UTC").dt.tz_localize(None)


def wind_rows(fc: pd.DataFrame) -> pd.DataFrame:
    """Select the wind-complex rows of an FC frame, case/accent tolerant.

    ``nom_tipousina`` is ``"Eólica"`` for wind; matched lower-cased so encoding
    or capitalisation drift does not silently drop the whole fleet.
    """
    tipo = fc["nom_tipousina"].astype(str).str.strip().str.lower()
    return fc[tipo == WIND_TIPO].copy()


def wind_complexes_from_fc(fc: pd.DataFrame) -> pd.DataFrame:
    """Derive per-complex metadata from the ONS FC frame itself.

    Capacity is the MAXIMUM ``val_capacidadeinstalada`` observed over the frame
    (the built-out capacity; the series ramps as turbines commission).
    Coordinates are the collector-substation location, taken from the first row
    that carries them.

    Args:
        fc: One or more concatenated ``FATOR_CAPACIDADE`` frames (wind and
            non-wind rows; this filters to wind). Uses ``id_ons``,
            ``nom_usina_conjunto``, ``nom_tipousina``, ``val_capacidadeinstalada``,
            ``val_latitudesecoletora``, ``val_longitudesecoletora``,
            ``id_subsistema``, ``nom_estado``.

    Returns:
        One row per ``ID`` (``id_ons``) with ``site_name``, ``lon``, ``lat``,
        ``capacity_mw``, ``subsystem``, ``state``.
    """
    wind = wind_rows(fc)
    wind["ID"] = wind["id_ons"].astype(str).str.strip()
    for col in ("val_capacidadeinstalada", "val_latitudesecoletora",
                "val_longitudesecoletora"):
        wind[col] = pd.to_numeric(wind[col], errors="coerce")

    def _first_valid(series: pd.Series):
        nn = series.dropna()
        return nn.iloc[0] if len(nn) else float("nan")

    grouped = wind.groupby("ID").agg(
        site_name=("nom_usina_conjunto", "first"),
        lat=("val_latitudesecoletora", _first_valid),
        lon=("val_longitudesecoletora", _first_valid),
        capacity_mw=("val_capacidadeinstalada", "max"),
        subsystem=("id_subsistema", "first"),
        state=("nom_estado", "first"),
    ).reset_index()
    return grouped


def monthly_cf_from_fc(
    fc: pd.DataFrame,
    year_start: int,
    year_end: int,
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> pd.DataFrame:
    """Aggregate the hourly ONS capacity factor to monthly means in UTC bins.

    The ONS ``val_fatorcapacidade`` is already a dimensionless CF per hour and
    complex, so the monthly value is simply its mean over the month — no
    capacity arithmetic (that is baked into the ONS figure, against the
    then-current installed capacity). Months with less than ``min_coverage`` of
    their expected hourly samples are NaN rather than biased by a gap.

    Args:
        fc: Concatenated ``FATOR_CAPACIDADE`` frames. Uses ``id_ons``,
            ``nom_tipousina``, ``din_instante`` (naive Brasília), and
            ``val_fatorcapacidade``.
        year_start: First UTC year to include (inclusive).
        year_end: Last UTC year to include (inclusive).
        min_coverage: Minimum fraction of a month's hourly samples that must be
            present; months below it are NaN.

    Returns:
        Wide frame with ``ID``, ``year``, ``obs_1``..``obs_12`` (monthly mean
        capacity factor; missing months NaN).
    """
    wind = wind_rows(fc)
    wind["ID"] = wind["id_ons"].astype(str).str.strip()
    wind["cf"] = pd.to_numeric(wind["val_fatorcapacidade"], errors="coerce")
    wind["timestamp"] = _brasilia_to_utc(wind["din_instante"])
    wind["year"] = wind["timestamp"].dt.year
    wind["month"] = wind["timestamp"].dt.month
    wind = wind[(wind["year"] >= int(year_start)) & (wind["year"] <= int(year_end))]

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


def constrained_off_account(coff: pd.DataFrame) -> pd.DataFrame:
    """Reduce the ONS constrained-off series to a monthly curtailment account.

    Separates curtailment from the resource by energy accounting: ``val_geracao``
    is what the complex delivered, ``val_geracaolimitada`` is what it was
    constrained off, so the resource offered is their sum. Both are MWmed per
    hour; summed over the month (UTC bins) they give delivered and curtailed
    energy, and the curtailed *fraction* of the available resource.

    Args:
        coff: Concatenated ``RESTRICAO_COFF_EOLICA`` frames. Uses ``id_ons``,
            ``din_instante`` (naive Brasília), ``val_geracao``,
            ``val_geracaolimitada``.

    Returns:
        Long frame with ``ID``, ``year``, ``month``, ``delivered_mwmed``,
        ``curtailed_mwmed``, ``curtailed_fraction`` (curtailed / (delivered +
        curtailed); 0 when nothing was available).
    """
    df = coff.copy()
    df["ID"] = df["id_ons"].astype(str).str.strip()
    df["delivered"] = pd.to_numeric(df["val_geracao"], errors="coerce").fillna(0.0)
    df["curtailed"] = pd.to_numeric(df["val_geracaolimitada"], errors="coerce").fillna(0.0)
    df["timestamp"] = _brasilia_to_utc(df["din_instante"])
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    agg = (
        df.groupby(["ID", "year", "month"])
        .agg(delivered_mwmed=("delivered", "sum"), curtailed_mwmed=("curtailed", "sum"))
        .reset_index()
    )
    available = agg["delivered_mwmed"] + agg["curtailed_mwmed"]
    agg["curtailed_fraction"] = (agg["curtailed_mwmed"] / available).where(available > 0, 0.0)
    return agg


def curtailment_mask_months(
    account: pd.DataFrame,
    *,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Months whose observed CF understates the resource through curtailment.

    A month is masked when its curtailed fraction exceeds ``threshold``: the ONS
    capacity factor there reflects constrained-off delivery, not the wind
    resource, and would pull the correction toward absorbing curtailment as if
    it were reanalysis bias — the exact contaminant the research doc flags for
    the Nordeste. This is the Brazil analogue of the AU registered-capacity
    mask: an explicit ``(ID, year, month)`` list the source sets to NaN.

    Args:
        account: Output of :func:`constrained_off_account`.
        threshold: Curtailed-fraction above which the month is masked.

    Returns:
        Frame with ``ID``, ``year``, ``month``, ``curtailed_fraction``.
    """
    masked = account[account["curtailed_fraction"] > float(threshold)]
    return masked[["ID", "year", "month", "curtailed_fraction"]].reset_index(drop=True)


def build_br_metadata(
    complexes: pd.DataFrame,
    *,
    height: float,
    model: str,
    commissioning: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Emit the ``ONSBrazilSource`` metadata contract from the FC-derived table.

    Capacity comes from the ONS installed capacity (MW → kW, the source
    contract unit). ``height``/``model`` are uniform defaults recorded in
    ``*_source`` columns (ONS carries no hub height or turbine model — the
    vintage-aware assignment is the same named follow-up as AU/US). Commissioning
    is joined from ``commissioning`` (e.g. earliest SIGA operating date per
    complex) when provided, else left NaT.

    Args:
        complexes: Output of :func:`wind_complexes_from_fc`.
        height: Uniform hub-height default, metres.
        model: Uniform power-curve key (must be a column of your
            ``power_curves.csv``).
        commissioning: Optional frame with ``ID`` and ``commissioning_date``.

    Returns:
        The contract: ``ID``, ``site_name``, ``lon``, ``lat``, ``height``,
        ``capacity`` (kW), ``model``, ``type``, ``commissioning_date``, plus
        provenance ``height_source``, ``model_source`` and the context columns
        ``subsystem``, ``state``. Complexes without coordinates are dropped.
    """
    md = complexes.copy()
    md["ID"] = md["ID"].astype(str)
    md["height"] = float(height)
    md["height_source"] = "default-uniform"
    md["model"] = model
    md["model_source"] = "default-uniform"
    md["capacity"] = pd.to_numeric(md["capacity_mw"], errors="coerce") * 1000.0
    md["type"] = "onshore"

    if commissioning is not None and len(commissioning):
        comm = commissioning.copy()
        comm["ID"] = comm["ID"].astype(str)
        md = md.merge(comm[["ID", "commissioning_date"]], on="ID", how="left")
    else:
        md["commissioning_date"] = pd.NaT

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
            "subsystem",
            "state",
        ]
    ].copy()
    out = out[out["lon"].notna() & out["lat"].notna()].reset_index(drop=True)
    return out


def commissioning_from_siga(siga: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """Earliest ANEEL operating date per ONS complex, joined through CEG.

    Optional enrichment: ONS ``id_ons`` groups several ANEEL plants (CEGs); this
    maps each ONS complex to the earliest ``DatEntradaOperacao`` among the wind
    plants (``SigTipoGeracao == 'EOL'``) whose CEG appears against that complex
    in the FC frame. Complexes whose FC ``ceg`` is ``-`` throughout get no date
    (the plant-vs-complex gap), which is why commissioning is optional.

    Args:
        siga: ANEEL SIGA frame (semicolon/latin-1/decimal-comma already parsed).
            Uses ``CodCEG``, ``SigTipoGeracao``, ``DatEntradaOperacao``.
        fc: FC frame carrying ``id_ons`` and ``ceg``.

    Returns:
        Frame with ``ID`` (id_ons) and ``commissioning_date``.
    """
    links = wind_rows(fc)[["id_ons", "ceg"]].copy()
    links["ID"] = links["id_ons"].astype(str).str.strip()
    links["ceg"] = links["ceg"].astype(str).str.strip()
    links = links[links["ceg"].str.len() > 0]
    links = links[links["ceg"] != "-"].drop_duplicates()

    wind = siga[siga["SigTipoGeracao"].astype(str).str.strip().str.upper() == "EOL"].copy()
    wind["ceg"] = wind["CodCEG"].astype(str).str.strip()
    wind["op"] = pd.to_datetime(wind["DatEntradaOperacao"], errors="coerce")

    merged = links.merge(wind[["ceg", "op"]], on="ceg", how="inner")
    out = merged.groupby("ID")["op"].min().reset_index()
    out.columns = ["ID", "commissioning_date"]
    return out
