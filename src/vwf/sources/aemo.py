"""Farm-level observation source for the Australian NEM (design §2).

Phase 1 delivers the adapter contract and the pure transformation logic,
exercised against synthetic fixtures; fetching real AEMO data is a Phase 2
step. The transformations encode two decisions made at design review:

- **Timezone**: AEMO SCADA timestamps are market time (AEST, UTC+10, no
  DST). They are converted to UTC at ingest and monthly bins are UTC —
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
DEFAULT_TRAIN_YEARS: tuple[int, int] = (2018, 2022)


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


def scada_to_monthly_cf(
    scada: pd.DataFrame,
    metadata: pd.DataFrame,
    year_start: int,
    year_end: int,
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> pd.DataFrame:
    """Aggregate 5-minute SCADA MW to monthly capacity factors in UTC bins.

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
    required = {"timestamp", "ID", "mw"}
    if not required.issubset(scada.columns):
        raise ValueError(f"SCADA frame must have columns {sorted(required)}")

    scada = scada.copy()
    scada["ID"] = scada["ID"].astype(str)
    # The timezone decision: convert AEST -> UTC BEFORE any binning.
    scada["timestamp"] = aest_to_utc(scada["timestamp"])
    scada["year"] = scada["timestamp"].dt.year
    scada["month"] = scada["timestamp"].dt.month
    scada = scada[(scada["year"] >= int(year_start)) & (scada["year"] <= int(year_end))]

    interval_hours = SCADA_INTERVAL_MINUTES / 60.0
    scada["energy_mwh"] = pd.to_numeric(scada["mw"], errors="coerce") * interval_hours

    monthly = (
        scada.groupby(["ID", "year", "month"])
        .agg(energy_mwh=("energy_mwh", "sum"), n_intervals=("energy_mwh", "count"))
        .reset_index()
    )

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
    # is NaN — the first fully post-commissioning month is the first valid one.
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

    wide = (
        monthly.pivot(index=["ID", "year"], columns="month", values="cf")
        .reindex(columns=range(1, 13))
        .reset_index()
    )
    wide.columns = ["ID", "year"] + [f"obs_{m}" for m in range(1, 13)]
    return wide


@register
class AEMONemSource(ObservationSource):
    """Per-farm (DUID) monthly capacity factors for the Australian NEM.

    Reads pre-downloaded files from ``input/turbine_level_data/AU_NEM/``:

    ``au_nem_md.csv``
        Farm metadata: ``ID`` (DUID), ``lon``, ``lat``, ``height`` (m),
        ``capacity`` (kW), ``model``, ``type``, optional
        ``commissioning_date``. From the user's public-source turbine
        compilation plus the AEMO registration lists.
    ``au_nem_scada.csv``
        Long SCADA: ``timestamp`` (naive AEST market time), ``ID``, ``mw``.

    Phase 2 wires the actual data acquisition; this adapter only transforms.
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

        path = self._data_dir() / "au_nem_scada.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"AEMO SCADA not found at {path}. This adapter reads "
                "pre-downloaded files; see the class docstring for the schema "
                "(data acquisition is a Phase 2 step)."
            )
        scada = pd.read_csv(path)
        return scada_to_monthly_cf(
            scada, self.load_metadata(), int(year_start), int(year_end)
        )
