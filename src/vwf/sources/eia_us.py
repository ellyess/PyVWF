"""Plant-level observation source for the United States (EIA + USWTDB).

The unit of observation is the **plant** (``obs_unit = "plant"``): EIA-923
reports monthly *net* generation once per plant, so the plant is the site we
simulate — the same design as the AU adapter, where the mechanical
``obs_level`` is "turbine" but the real unit is the farm (DUID).

Two decisions are baked into the CF finalisation, both documented in the
region config and the class docstring:

- **UTC / calendar-month bins.** EIA-923 generation is already a monthly
  total on the US calendar month. There is no sub-monthly timestamp to
  convert, so the bins are simply the reported calendar months — recorded as
  ``time_convention = "calendar-month"`` (distinct from the AU
  ``utc-monthly-bins``, which had to convert 5-minute AEST SCADA). This is
  exact for monthly training; it means the US months are local calendar
  months, not UTC-edge-aligned, which for a monthly-mean target is immaterial.
- **Annual respondents are dropped.** A plant flagged ``A`` in EIA-923 has its
  twelve monthly cells *imputed* by EIA from the annual total, not measured.
  Keeping them would train the seasonal cycle on a synthetic profile, so they
  are set to NaN by default (``drop_annual_respondents=True``). Monthly (``M``)
  and combined (``AM``) respondents are kept.

Standing caveat (config comment): EIA-923 ``Netgen`` is *net* of station use,
and curtailment in ERCOT/SPP contaminates observed CF where it occurs — US
corrections absorb both more than European monthly *generation* data does.
Curtailment screening is a shared follow-up (``pyvwf.qc``), not yet applied
here.
"""
from __future__ import annotations

from calendar import monthrange
from typing import ClassVar

import pandas as pd

from vwf.config import PyVWFPaths
from vwf.sources.base import ObservationSource, ObsLevel
from vwf.sources.registry import register

#: Respondent-frequency flags whose monthly cells are EIA-imputed, not measured.
IMPUTED_RESPONDENT_FLAGS: frozenset[str] = frozenset({"A"})

#: Training window used when no explicit year range is requested. EIA-923 runs
#: continuously from 2001; a recent multi-year window keeps the fleet close to
#: the EIA-860 nameplate and the USWTDB metadata release.
DEFAULT_TRAIN_YEARS: tuple[int, int] = (2019, 2021)


def netgen_to_monthly_cf(
    netgen: pd.DataFrame,
    metadata: pd.DataFrame,
    year_start: int,
    year_end: int,
    *,
    drop_annual_respondents: bool = True,
) -> pd.DataFrame:
    """Turn EIA-923 monthly net generation into the wide monthly-CF contract.

    ``CF = net_gen_mwh / (capacity_MW × hours_in_month)``, with capacity taken
    from ``metadata`` (kW, the source contract unit) at load time so the CF
    always uses the current nameplate. Net generation may be negative in a calm
    month (parasitic load exceeds output); it is kept as-is (a small negative
    CF), not clipped, so the correction sees the real signal — the same
    non-clipping choice the AU adapter makes.

    Args:
        netgen: Long frame with ``ID`` (plant code), ``year``, ``month``,
            ``net_gen_mwh`` and ``respondent_frequency`` (per-plant-year flag).
        metadata: Plant metadata with ``ID``, ``capacity`` (kW) and optionally
            ``commissioning_date``.
        year_start: First year to include (inclusive).
        year_end: Last year to include (inclusive).
        drop_annual_respondents: When True (default), plant-years whose
            respondent frequency is annual have every month set to NaN — their
            monthly split is imputed, not observed.

    Returns:
        Wide frame with ``ID``, ``year``, ``obs_1``..``obs_12``. Months before
        (or containing) a plant's commissioning date are NaN.
    """
    monthly = netgen.copy()
    monthly["ID"] = monthly["ID"].astype(str)
    monthly["year"] = pd.to_numeric(monthly["year"], errors="coerce").astype("int64")
    monthly["month"] = pd.to_numeric(monthly["month"], errors="coerce").astype("int64")
    monthly["net_gen_mwh"] = pd.to_numeric(monthly["net_gen_mwh"], errors="coerce")
    monthly = monthly[
        (monthly["year"] >= int(year_start)) & (monthly["year"] <= int(year_end))
    ]

    if drop_annual_respondents and "respondent_frequency" in monthly.columns:
        imputed = (
            monthly["respondent_frequency"].astype(str).str.strip().str.upper()
            .isin(IMPUTED_RESPONDENT_FLAGS)
        )
        monthly.loc[imputed, "net_gen_mwh"] = float("nan")

    meta = metadata.copy()
    meta["ID"] = meta["ID"].astype(str)
    monthly = monthly.merge(meta[["ID", "capacity"]], on="ID", how="inner")

    hours = monthly.apply(
        lambda r: monthrange(int(r["year"]), int(r["month"]))[1] * 24.0, axis=1
    )
    # capacity is kW (source contract); net_gen is MWh: align units.
    monthly["cf"] = (monthly["net_gen_mwh"] * 1000.0) / (
        hours * monthly["capacity"].astype(float)
    )

    # Commissioning mask: any month that STARTS before the commissioning date
    # is NaN — the first fully post-commissioning month is the first valid one.
    if "commissioning_date" in meta.columns:
        commissioning = pd.to_datetime(
            meta.set_index("ID")["commissioning_date"], errors="coerce"
        ).dropna()
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
class EIAUSSource(ObservationSource):
    """Per-plant monthly capacity factors for the United States.

    Reads pre-downloaded files from ``input/turbine_level_data/US/``:

    ``us_md.csv``
        Plant metadata: ``ID`` (EIA plant code), ``lon``, ``lat``, ``height``
        (m), ``capacity`` (kW), ``model``, ``type``, optional
        ``commissioning_date``, plus provenance columns (``height_source``,
        ``uswtdb_model``). Built by ``scripts/process_eia_us.py`` from EIA-860
        (capacity, coordinates, operating date) joined to the USWTDB
        (capacity-weighted hub height, model).
    ``us_eia923_netgen.csv``
        Long EIA-923 net generation: ``ID``, ``year``, ``month``,
        ``net_gen_mwh``, ``respondent_frequency``. Written by the same script
        from the EIA-923 Page 1 schedule via
        :func:`vwf.datasets.eia_us.wind_generation_from_eia923`.

    Finalisation (capacity conversion, annual-respondent screen, commissioning
    mask, wide pivot) runs through :func:`netgen_to_monthly_cf` with the
    current metadata at load time, so the on-disk generation table cannot drift
    from the audited transform.
    """

    name: ClassVar[str] = "eia-us"
    obs_level: ClassVar[ObsLevel] = "turbine"  # mechanical contract; obs_unit is "plant"
    countries: ClassVar[tuple[str, ...]] = ("US", "USA")

    def __init__(self, country: str = "US") -> None:
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
        return PyVWFPaths.TURBINE_DATA / "US"

    def load_metadata(self) -> pd.DataFrame:
        path = self._data_dir() / "us_md.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"EIA-US plant metadata not found at {path}. This adapter reads "
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
        """Monthly capacity factors from EIA-923 net generation."""
        if year_start is None or year_end is None:
            year_start, year_end = self.default_train_years

        path = self._data_dir() / "us_eia923_netgen.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"EIA-923 net generation not found at {path}. This adapter reads "
                "pre-downloaded files; see the class docstring for the schema "
                "(data acquisition is a Phase 2 step)."
            )
        netgen = pd.read_csv(path)
        return netgen_to_monthly_cf(
            netgen,
            self.load_metadata(),
            int(year_start),
            int(year_end),
        )
