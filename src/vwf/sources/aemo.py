"""Farm-level observation source for the Australian NEM (design §2).

Phase 1 delivers the adapter contract and the pure transformation logic,
exercised against synthetic fixtures; fetching real AEMO data is a Phase 2
step. The transformations encode two decisions made at design review:

- **Timezone**: AEMO SCADA timestamps are market time (AEST, UTC+10, no
  DST). They are converted to UTC at ingest and monthly bins are UTC,
  the same convention as the ERA5/simulation side everywhere else. Recorded
  in every manifest as ``time_convention = "utc-monthly-bins"``.
- **Granularity**: one row per DUID, which is a wind *farm*
  (``obs_unit = "farm"`` in the region config and manifest).

Standing caveat (config comment, design §2): SCADA records curtailed output,
so AU corrections absorb curtailment more than European monthly generation
data does.
"""
from __future__ import annotations

from calendar import monthrange
from datetime import timedelta, timezone
from typing import ClassVar

import pandas as pd

from vwf.config import PyVWFPaths
from vwf.sources.base import ObservationSource, ObsLevel
from vwf.sources.registry import register

#: AEMO market time: UTC+10, fixed, no daylight saving.
AEST = timezone(timedelta(hours=10))

#: SCADA dispatch interval in minutes.
SCADA_INTERVAL_MINUTES = 5

#: Months with less than this fraction of expected SCADA intervals are NaN.
DEFAULT_MIN_COVERAGE = 0.9

#: Training window used when no explicit year range is requested.
#: Trimmed to 2020-2022 at D2 sign-off: the four validation pillars need a
#: couple of full held-out years, not six, and it cuts the SCADA download.
DEFAULT_TRAIN_YEARS: tuple[int, int] = (2020, 2022)


def aest_to_utc(timestamps: pd.Series) -> pd.Series:
    """Convert naive AEST (market-time) timestamps to naive UTC.

    AEMO publishes market time with no DST, so the conversion is a fixed
    -10 h shift; the result is naive UTC to match the rest of the pipeline.
    """
    ts = pd.to_datetime(timestamps)
    if getattr(ts.dt, "tz", None) is not None:
        raise ValueError(
            "AEMO SCADA timestamps must be naive market time (AEST); got "
            "timezone-aware input"
        )
    return ts.dt.tz_localize(AEST).dt.tz_convert("UTC").dt.tz_localize(None)


def scada_partial_aggregate(scada: pd.DataFrame) -> pd.DataFrame:
    """Reduce raw 5-minute SCADA to per-(ID, UTC year, month) partials.

    The pure aggregation half of :func:`scada_to_monthly_cf`: AEST→UTC
    conversion, then energy and interval counts per month. Partials from
    separate chunks (e.g. AEMO's monthly archive files, which are cut on
    MARKET-time month boundaries and therefore straddle UTC months) can be
    summed with :func:`combine_partials` before finalisation: the whole
    point of the split, since concatenating finalised monthly CFs from
    market-month chunks would get the straddled edge hours wrong.

    Args:
        scada: Long frame with columns ``timestamp`` (naive AEST), ``ID``
            (DUID), and ``mw`` (dispatch-interval average output in MW).

    Returns:
        Frame with ``ID``, ``year``, ``month``, ``energy_mwh``,
        ``n_intervals``.
    """
    required = {"timestamp", "ID", "mw"}
    if not required.issubset(scada.columns):
        raise ValueError(f"SCADA frame must have columns {sorted(required)}")

    scada = scada.copy()
    scada["ID"] = scada["ID"].astype(str)
    # The timezone decision: convert AEST -> UTC BEFORE any binning.
    scada["timestamp"] = aest_to_utc(scada["timestamp"])
    scada["year"] = scada["timestamp"].dt.year
    scada["month"] = scada["timestamp"].dt.month

    interval_hours = SCADA_INTERVAL_MINUTES / 60.0
    scada["energy_mwh"] = pd.to_numeric(scada["mw"], errors="coerce") * interval_hours

    return (
        scada.groupby(["ID", "year", "month"])
        .agg(energy_mwh=("energy_mwh", "sum"), n_intervals=("energy_mwh", "count"))
        .reset_index()
    )


def combine_partials(partials: list[pd.DataFrame]) -> pd.DataFrame:
    """Sum per-month partials from several chunks into one table."""
    if not partials:
        raise ValueError("no partials to combine")
    stacked = pd.concat(partials, ignore_index=True)
    return (
        stacked.groupby(["ID", "year", "month"], as_index=False)[
            ["energy_mwh", "n_intervals"]
        ].sum()
    )


def finalise_monthly_cf(
    partials: pd.DataFrame,
    metadata: pd.DataFrame,
    year_start: int,
    year_end: int,
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    capacity_mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Turn per-month energy partials into the wide monthly-CF contract.

    The judgement half of :func:`scada_to_monthly_cf`: capacity conversion,
    coverage floor, commissioning mask, wide pivot. Kept separate so
    precomputed partials (fast path) run through exactly this audited code
    with the CURRENT metadata at load time.

    Args:
        capacity_mask: Optional frame of (``ID``, ``year``, ``month``) rows
            whose CF must be NaN because the farm's REGISTERED capacity that
            month is unreliable as a denominator (mid-month change, below
            final build, or pre-registration), built from AEMO DUDETAIL by
            ``vwf.datasets.aemo_au.capacity_mask_months``. A ramping farm
            otherwise injects a spurious sub-annual signal into exactly the
            seasonal cycle pillar A judges.
    """
    monthly = partials.copy()
    monthly["ID"] = monthly["ID"].astype(str)
    # Fixed dtypes so precomputed partials (CSV round-trip) and the in-memory
    # path produce byte-identical frames.
    monthly["year"] = monthly["year"].astype("int64")
    monthly["month"] = monthly["month"].astype("int64")
    monthly = monthly[
        (monthly["year"] >= int(year_start)) & (monthly["year"] <= int(year_end))
    ]

    meta = metadata.copy()
    meta["ID"] = meta["ID"].astype(str)
    monthly = monthly.merge(meta[["ID", "capacity"]], on="ID", how="inner")

    hours = monthly.apply(
        lambda r: monthrange(int(r["year"]), int(r["month"]))[1] * 24.0, axis=1
    )
    expected_intervals = hours * 60.0 / SCADA_INTERVAL_MINUTES

    # capacity is kW (source contract); energy is MWh: align units.
    monthly["cf"] = (monthly["energy_mwh"] * 1000.0) / (
        hours * monthly["capacity"].astype(float)
    )
    monthly.loc[monthly["n_intervals"] / expected_intervals < min_coverage, "cf"] = float(
        "nan"
    )

    # Commissioning mask: any month that STARTS before the commissioning date
    # is NaN; the first fully post-commissioning month is the first valid one.
    if "commissioning_date" in meta.columns:
        commissioning = meta.set_index("ID")["commissioning_date"].dropna()
        commissioning = pd.to_datetime(commissioning)
        month_start = pd.to_datetime(
            monthly["year"].astype(str) + "-" + monthly["month"].astype(str) + "-01"
        )
        commissioned = monthly["ID"].map(commissioning)
        monthly.loc[
            commissioned.notna() & (month_start < commissioned), "cf"
        ] = float("nan")

    # Registered-capacity mask (see docstring): explicit month list wins over
    # everything computed above.
    if capacity_mask is not None and len(capacity_mask):
        mask_keys = set(
            zip(
                capacity_mask["ID"].astype(str),
                capacity_mask["year"].astype(int),
                capacity_mask["month"].astype(int),
            )
        )
        in_mask = [
            (i, y, m) in mask_keys
            for i, y, m in zip(monthly["ID"], monthly["year"], monthly["month"])
        ]
        monthly.loc[in_mask, "cf"] = float("nan")

    wide = (
        monthly.pivot(index=["ID", "year"], columns="month", values="cf")
        .reindex(columns=range(1, 13))
        .reset_index()
    )
    wide.columns = ["ID", "year"] + [f"obs_{m}" for m in range(1, 13)]
    return wide


def scada_to_monthly_cf(
    scada: pd.DataFrame,
    metadata: pd.DataFrame,
    year_start: int,
    year_end: int,
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> pd.DataFrame:
    """Aggregate 5-minute SCADA MW to monthly capacity factors in UTC bins.

    Composition of :func:`scada_partial_aggregate` and
    :func:`finalise_monthly_cf`; see those for the semantics. Note the
    partials are computed over the whole input and the year filter is applied
    at finalisation: identical result, since UTC binning precedes filtering
    either way.

    Args:
        scada: Long frame with columns ``timestamp`` (naive AEST), ``ID``
            (DUID), and ``mw`` (dispatch-interval average output in MW).
        metadata: Farm metadata with ``ID``, ``capacity`` (kW, per the
            ObservationSource contract) and optionally ``commissioning_date``.
        year_start: First UTC year to include (inclusive).
        year_end: Last UTC year to include (inclusive).
        min_coverage: Minimum fraction of a month's SCADA intervals that must
            be present; months below it are NaN rather than biased low.

    Returns:
        Wide frame with ``ID``, ``year``, ``obs_1``..``obs_12``. Months before
        (or containing) a farm's commissioning date are NaN.
    """
    return finalise_monthly_cf(
        scada_partial_aggregate(scada),
        metadata,
        year_start,
        year_end,
        min_coverage=min_coverage,
    )


@register
class AEMONemSource(ObservationSource):
    """Per-farm (DUID) monthly capacity factors for the Australian NEM.

    Reads pre-downloaded files from ``input/observations/turbine/AU_NEM/``:

    ``au_nem_md.csv``
        Farm metadata: ``ID`` (DUID), ``lon``, ``lat``, ``height`` (m),
        ``capacity`` (kW), ``model``, ``type``, optional
        ``commissioning_date``. From the user's public-source turbine
        compilation plus the AEMO registration lists.
    ``au_nem_scada.csv``
        Long SCADA: ``timestamp`` (naive AEST market time), ``ID``, ``mw``.
    ``au_nem_scada_monthly_partials.csv`` (optional fast path)
        Per-(ID, UTC year, month) ``energy_mwh`` + ``n_intervals`` partials,
        as written by ``scripts/process/aemo_au.py`` via
        :func:`scada_partial_aggregate`/:func:`combine_partials`. Preferred
        when present: four years of 5-minute SCADA is ~45M rows, while the
        partials are a few thousand. Finalisation (capacity, coverage
        floor, commissioning mask) still runs through
        :func:`finalise_monthly_cf` with the current metadata at load time,
        so the fast path cannot drift from the audited transform.
    """

    name: ClassVar[str] = "aemo-nem"
    obs_level: ClassVar[ObsLevel] = "turbine"  # mechanical contract; obs_unit is "farm"
    countries: ClassVar[tuple[str, ...]] = ("AU-NEM", "AU")

    def __init__(self, country: str = "AU-NEM") -> None:
        country = country.upper()
        if country not in self.countries:
            raise ValueError(
                f"{type(self).__name__} supports {self.countries}, got {country!r}"
            )
        self.country: str = country

    @property
    def default_train_years(self) -> tuple[int, int]:
        """Inclusive training window."""
        return DEFAULT_TRAIN_YEARS

    def _data_dir(self):
        return PyVWFPaths.TURBINE_DATA / "AU_NEM"

    def load_metadata(self) -> pd.DataFrame:
        path = self._data_dir() / "au_nem_md.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"AEMO farm metadata not found at {path}. This adapter reads "
                "pre-downloaded files; see the class docstring for the schema "
                "(data acquisition is a Phase 2 step)."
            )
        meta = pd.read_csv(path)
        required = {"ID", "lon", "lat", "height", "capacity", "model"}
        missing = required - set(meta.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns {sorted(missing)}")
        meta["ID"] = meta["ID"].astype(str)
        if "type" not in meta.columns:
            meta["type"] = "onshore"
        return meta

    def load_observations(
        self,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> pd.DataFrame:
        """Monthly capacity factors from 5-minute SCADA, UTC-binned."""
        if year_start is None or year_end is None:
            year_start, year_end = self.default_train_years

        # Registered-capacity mask, if the processing step produced one.
        mask_path = self._data_dir() / "au_nem_capacity_mask.csv"
        capacity_mask = pd.read_csv(mask_path) if mask_path.is_file() else None

        partials_path = self._data_dir() / "au_nem_scada_monthly_partials.csv"
        if partials_path.is_file():
            partials = pd.read_csv(partials_path)
            return finalise_monthly_cf(
                partials,
                self.load_metadata(),
                int(year_start),
                int(year_end),
                capacity_mask=capacity_mask,
            )

        path = self._data_dir() / "au_nem_scada.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"AEMO SCADA not found at {path} (and no precomputed partials at "
                f"{partials_path.name}). This adapter reads pre-downloaded files; "
                "see the class docstring for the schema (data acquisition is a "
                "Phase 2 step)."
            )
        scada = pd.read_csv(path)
        return finalise_monthly_cf(
            scada_partial_aggregate(scada),
            self.load_metadata(),
            int(year_start),
            int(year_end),
            capacity_mask=capacity_mask,
        )
