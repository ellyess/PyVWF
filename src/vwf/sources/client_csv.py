"""Turbine-level observation source backed by two user-supplied CSV files.

The general "bring your own fleet" adapter: point it at your own fleet metadata
and monthly generation as CSVs, and it maps them onto the
:class:`~vwf.sources.base.ObservationSource` contract so the standard
``run_train`` / ``run_evaluate`` pipeline produces a validated correction for
that fleet, with no new adapter code. Use this instead of writing a bespoke
source when the data is a one-off or private rather than a public national
dataset. Construct it explicitly and pass it to the driver via ``source=`` (it is
deliberately not auto-resolved from a country code).

Expected inputs (column names are remappable via ``column_map``):

- ``metadata_csv``: one row per site, with at least an id, longitude, latitude,
  rated capacity, and hub height. A power-curve key (``model``) is used directly
  if present; otherwise, if a rotor ``diameter`` is present, a curve is matched
  from the library by specific power (the same path the DK/DE fleets use).
- ``generation_csv``: long form, one row per (site, year, month), holding either
  generated energy (converted to capacity factor using the site's capacity and
  the hours in the month) or a capacity factor directly.

See ``docs/guides/your-own-data.md`` for a worked example and the exact
column contract.
"""
from __future__ import annotations

from calendar import monthrange
from pathlib import Path
from typing import ClassVar

import pandas as pd

from vwf.sources.base import ObservationSource, ObsLevel
from vwf.sources.registry import register

#: Default source-column -> standard-column mapping. Override per client with the
#: ``column_map`` constructor argument (only the keys that differ need listing).
DEFAULT_COLUMN_MAP: dict[str, str] = {
    "ID": "ID",
    "lon": "lon",
    "lat": "lat",
    "capacity": "capacity",   # rated capacity (see capacity_unit)
    "height": "height",       # hub height, metres
    "diameter": "diameter",   # rotor diameter, metres (optional, for curve match)
    "model": "model",         # power-curve key (optional; matched if absent)
    "type": "type",           # onshore/offshore (optional; defaults onshore)
    "manufacturer": "manufacturer",  # optional, aids curve matching
}


@register
class ClientCsvTurbineSource(ObservationSource):
    """Fleet metadata and monthly generation read from two client CSV files.

    ``countries`` is empty: this source is always constructed by the caller with
    the client's files, never resolved from a country code.
    """

    name: ClassVar[str] = "client-csv-turbine"
    obs_level: ClassVar[ObsLevel] = "turbine"
    countries: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        metadata_csv: str | Path,
        generation_csv: str | Path,
        *,
        country: str = "XX",
        column_map: dict[str, str] | None = None,
        capacity_unit: str = "kw",
        generation_column: str = "generation_mwh",
        generation_is_cf: bool = False,
    ) -> None:
        """Bind the client's files and their column conventions.

        Args:
            metadata_csv: Path to the fleet metadata CSV.
            generation_csv: Path to the long-form monthly generation CSV, with
                columns ``ID``, ``year``, ``month`` and the value column named by
                ``generation_column``.
            country: Region code, used only for the ERA5 subset and messages.
            column_map: Standard-column -> source-column overrides for the
                metadata file. Only differing columns need listing.
            capacity_unit: ``"kw"`` (default) or ``"mw"``; converted to kW.
            generation_column: Name of the value column in ``generation_csv``.
            generation_is_cf: If True, the value column is already a capacity
                factor in [0, 1]; otherwise it is energy (same unit basis as
                ``capacity_unit``, e.g. MWh against MW capacity) and is converted
                to CF using the site capacity and hours in the month.
        """
        self.country = country.upper()
        self._metadata_csv = Path(metadata_csv)
        self._generation_csv = Path(generation_csv)
        self._map = {**DEFAULT_COLUMN_MAP, **(column_map or {})}
        self._capacity_unit = capacity_unit.lower()
        self._generation_column = generation_column
        self._generation_is_cf = generation_is_cf
        self._metadata: pd.DataFrame | None = None

    def _standardise_metadata(self) -> pd.DataFrame:
        raw = pd.read_csv(self._metadata_csv)
        # Rename source columns to the standard names where present.
        rename = {src: std for std, src in self._map.items() if src in raw.columns}
        md = raw.rename(columns=rename)

        required = ["ID", "lon", "lat", "capacity", "height"]
        missing = [c for c in required if c not in md.columns]
        if missing:
            raise ValueError(
                f"{self._metadata_csv.name}: missing required column(s) {missing} "
                f"after applying column_map. Present: {list(md.columns)}"
            )

        md["ID"] = md["ID"].astype(str)
        for c in ["lon", "lat", "capacity", "height"]:
            md[c] = pd.to_numeric(md[c], errors="coerce")
        if self._capacity_unit == "mw":
            md["capacity"] = md["capacity"] * 1000.0  # -> kW
        if "type" not in md.columns:
            md["type"] = "onshore"
        md = md.dropna(subset=["lon", "lat", "capacity", "height"]).reset_index(drop=True)
        md = md[md["height"] > 1.0]

        if "model" not in md.columns:
            # No explicit curve key: match one by specific power if a rotor
            # diameter is present, else fail loudly (the pipeline needs a curve).
            if "diameter" not in md.columns:
                raise ValueError(
                    f"{self._metadata_csv.name}: no 'model' curve key and no "
                    "'diameter' to match one. Provide either a power-curve key "
                    "per site or a rotor diameter (plus manufacturer if known)."
                )
            from vwf.data import add_models  # lazy: vwf.data imports this package

            if "manufacturer" not in md.columns:
                md["manufacturer"] = ""
            md = add_models(md)
        return md

    def load_metadata(self) -> pd.DataFrame:
        """Return the client's sites, with a power curve assigned to each."""
        if self._metadata is None:
            self._metadata = self._standardise_metadata()
        return self._metadata.copy()

    def load_observations(
        self,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> pd.DataFrame:
        """Return monthly capacity factors, wide (``ID``, ``year``, ``obs_1..12``)."""
        gen = pd.read_csv(self._generation_csv)
        for req in ("ID", "year", "month", self._generation_column):
            if req not in gen.columns:
                raise ValueError(
                    f"{self._generation_csv.name}: missing column '{req}'. "
                    f"Present: {list(gen.columns)}"
                )
        gen["ID"] = gen["ID"].astype(str)
        gen["year"] = pd.to_numeric(gen["year"], errors="coerce").astype("Int64")
        gen["month"] = pd.to_numeric(gen["month"], errors="coerce").astype("Int64")
        gen = gen.dropna(subset=["ID", "year", "month"])
        if year_start is not None:
            gen = gen[gen["year"] >= year_start]
        if year_end is not None:
            gen = gen[gen["year"] <= year_end]

        val = pd.to_numeric(gen[self._generation_column], errors="coerce")
        if self._generation_is_cf:
            gen["cf"] = val
        else:
            cap = self.load_metadata().set_index("ID")["capacity"]  # kW
            cap_kw = gen["ID"].map(cap)
            days = [monthrange(int(y), int(m))[1] for y, m in zip(gen["year"], gen["month"])]
            hours = pd.Series(days, index=gen.index) * 24.0
            # energy unit matches capacity basis; if capacity is kW, energy is kWh.
            energy = val * (1000.0 if self._capacity_unit == "mw" else 1.0)
            gen["cf"] = energy / (hours * cap_kw)

        wide = (
            gen.pivot_table(index=["ID", "year"], columns="month", values="cf")
            .rename(columns={m: f"obs_{m}" for m in range(1, 13)})
            .reset_index()
        )
        wide.columns.name = None
        return wide
