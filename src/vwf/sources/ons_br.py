"""Complex-level observation source for Brazil (ONS open data).

The unit of observation is the **complex** (``obs_unit = "complex"``): ONS
publishes wind at the *conjunto de usinas* level, one hourly capacity-factor
series per ``id_ons``. That is the natural ONS unit and the one its capacity
factor is defined against, the same design as the AU (farm) and US (plant)
adapters, where the mechanical ``obs_level`` stays "turbine".

Decisions baked into the finalisation, documented in the region config:

- **UTC bins from Brasília time.** ONS ``din_instante`` is Brasília civil time,
  fixed UTC-3 over the usable window (no DST since 2019; the Nordeste fleet
  never observed it). It is converted to UTC before binning, matching the
  ERA5/simulation convention, recorded as ``time_convention =
  "utc-monthly-bins"``. For a monthly-mean target the shift only moves a few
  edge hours, but the convention is kept consistent across regions.
- **Curtailment screening is a first-class option.** The Nordeste is heavily
  constrained off, and the ONS capacity factor reflects *delivered* energy, so
  in a curtailed month it understates the wind resource. Where the ONS
  constrained-off series is available (2021+), a curtailment mask (built by
  ``vwf.datasets.ons_br.curtailment_mask_months``) sets the affected months to
  NaN so the correction is not fitted to absorb curtailment as reanalysis bias.
  Pre-2021 has no such series and its CF carries unscreened curtailment: a
  standing caveat.
"""
from __future__ import annotations

from typing import ClassVar

import pandas as pd

from vwf.config import PyVWFPaths
from vwf.datasets.ons_br import DEFAULT_MIN_COVERAGE, monthly_cf_from_fc
from vwf.sources.base import ObservationSource, ObsLevel
from vwf.sources.registry import register

#: Training window used when no explicit year range is requested. The ONS
#: constrained-off series (curtailment screening) begins in 2021, so a window
#: from 2021 keeps every training month screenable.
DEFAULT_TRAIN_YEARS: tuple[int, int] = (2021, 2023)


def fc_to_monthly_cf(
    fc: pd.DataFrame,
    year_start: int,
    year_end: int,
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    curtailment_mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Monthly-mean CF from the ONS FC series, with optional curtailment mask.

    Thin finalisation over :func:`vwf.datasets.ons_br.monthly_cf_from_fc`: the
    pure monthly aggregation, then the curtailment screen. Kept here (source
    layer) rather than in the pure transform so the datasets module stays policy
    free: the same split as the AU adapter's capacity mask.

    Args:
        fc: Hourly wind FC frame (``id_ons``, ``nom_tipousina``,
            ``din_instante``, ``val_fatorcapacidade``).
        year_start, year_end: Inclusive UTC year bounds.
        min_coverage: Minimum fraction of a month's hourly samples required.
        curtailment_mask: Optional frame of ``(ID, year, month)`` rows whose CF
            must be NaN because curtailment there makes the observed CF
            unrepresentative of the resource.

    Returns:
        Wide frame ``ID``, ``year``, ``obs_1``..``obs_12``.
    """
    wide = monthly_cf_from_fc(fc, year_start, year_end, min_coverage=min_coverage)

    if curtailment_mask is not None and len(curtailment_mask):
        mask_keys = set(
            zip(
                curtailment_mask["ID"].astype(str),
                curtailment_mask["year"].astype(int),
                curtailment_mask["month"].astype(int),
            )
        )
        for m in range(1, 13):
            col = f"obs_{m}"
            hit = [
                (str(i), int(y), m) in mask_keys
                for i, y in zip(wide["ID"], wide["year"])
            ]
            wide.loc[hit, col] = float("nan")
    return wide


@register
class ONSBrazilSource(ObservationSource):
    """Per-complex monthly capacity factors for Brazil.

    Reads pre-downloaded files from ``input/observations/turbine/BR/``:

    ``br_md.csv``
        Complex metadata: ``ID`` (``id_ons``), ``lon``, ``lat``, ``height``
        (m), ``capacity`` (kW), ``model``, ``type``, optional
        ``commissioning_date``, plus provenance/context (``height_source``,
        ``subsystem``, ``state``). Built by ``scripts/process/ons_br.py`` from
        the ONS ``FATOR_CAPACIDADE`` file (which carries the coordinates and
        installed capacity itself), optionally enriched with ANEEL SIGA
        commissioning dates.
    ``br_fc.csv``
        Hourly wind FC: ``id_ons``, ``nom_tipousina``, ``din_instante`` (naive
        Brasília), ``val_fatorcapacidade``. Finalised to monthly means through
        :func:`fc_to_monthly_cf` at load time.
    ``br_curtailment_mask.csv`` (optional)
        ``ID``, ``year``, ``month`` months to NaN for curtailment, as written by
        the processing step from the ONS constrained-off series via
        ``vwf.datasets.ons_br.curtailment_mask_months``.
    """

    name: ClassVar[str] = "ons-br"
    obs_level: ClassVar[ObsLevel] = "turbine"  # mechanical contract; obs_unit is "complex"
    countries: ClassVar[tuple[str, ...]] = ("BR", "BRA")

    def __init__(self, country: str = "BR") -> None:
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
        return PyVWFPaths.TURBINE_DATA / "BR"

    def load_metadata(self) -> pd.DataFrame:
        path = self._data_dir() / "br_md.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"ONS-BR complex metadata not found at {path}. This adapter reads "
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
        """Monthly capacity factors from the hourly ONS FC series."""
        if year_start is None or year_end is None:
            year_start, year_end = self.default_train_years

        mask_path = self._data_dir() / "br_curtailment_mask.csv"
        curtailment_mask = pd.read_csv(mask_path) if mask_path.is_file() else None

        path = self._data_dir() / "br_fc.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"ONS FC series not found at {path}. This adapter reads "
                "pre-downloaded files; see the class docstring for the schema "
                "(data acquisition is a Phase 2 step)."
            )
        fc = pd.read_csv(path)
        return fc_to_monthly_cf(
            fc,
            int(year_start),
            int(year_end),
            curtailment_mask=curtailment_mask,
        )
