"""The Global Wind Power Tracker (GWPT): loading, filtering and plant-name keys.

Global Energy Monitor's tracker (CC-BY-4.0) supplies coordinates and capacity
where an observation source has none: the Argentina and Chile coordinate
joins, the WindStats coordinates, and the capacity weights and repairs of the
country-level grids. It is read from
``<input-root>/reference/gwpt/Global-Wind-Power-Tracker-February-2026.xlsx``.

The callers filter it in two ways, and both are kept, because moving one onto
the other would change which projects a result used:

- :func:`fleet_for` compares ``Country/Area`` as written and drops projects
  without coordinates or capacity. It serves the country-level grids.
- :func:`operating_projects` strips the country string first and keeps
  every operating row. It serves the per-plant coordinate joins.

Likewise the plant-name key, :func:`plant_key`, takes the word list each
region's join was built with (:data:`DROP_AR`, :data:`DROP_CL`).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from pyvwf.config import PyVWFPaths

#: The tracker release every committed result used.
FILENAME = "Global-Wind-Power-Tracker-February-2026.xlsx"

#: Region code to the tracker's ``Country/Area`` spelling, for the
#: country-level grids.
COUNTRY_NAME = {
    "BE": "Belgium",
    "ES": "Spain",
    "FR": "France",
    "IE": "Ireland",
    "IT": "Italy",
    "NL": "Netherlands",
    "NO": "Norway",
    "PT": "Portugal",
    "SE": "Sweden",
}

#: Words the Argentina join ignores in plant names.
DROP_AR = frozenset(
    {
        "PARQUE",
        "EOLICO",
        "EOLICA",
        "PE",
        "WIND",
        "FARM",
        "DEL",
        "DE",
        "LA",
        "LOS",
        "LAS",
        "EL",
        "GENNEIA",
        "SA",
        "S",
        "P",
        "AG",
    }
)

#: Words the Chile join ignores in plant names.
DROP_CL = frozenset(
    {
        "PARQUE",
        "EOLICO",
        "EOLICA",
        "PMGD",
        "PE",
        "WIND",
        "FARM",
        "CHILE",
        "ENEL",
        "DEL",
        "DE",
        "LA",
        "LOS",
        "LAS",
        "EL",
    }
)

_ROMAN = re.compile(r"I{1,3}V?|IV")


def default_path() -> Path:
    """Where the tracker lives under the current input root."""
    return PyVWFPaths.INPUT_ROOT / "reference" / "gwpt" / FILENAME


def load_gwpt(path: Path) -> pd.DataFrame:
    """Read the tracker's ``Data`` sheet."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Global Wind Power Tracker not found at {path}. "
            "See docs/guides/data-sources.md for where to download it."
        )
    return pd.read_excel(path, sheet_name="Data")


def load_exclusions(path: Path) -> set[str]:
    """GEM phase IDs to drop, keyed so a renamed project stays excluded.

    The table is ``configs/curation/gwpt_exclusions.csv``: records the tracker
    marks operating that independent registers contradict, each with its
    evidence. A missing file excludes nothing.
    """
    if not path.is_file():
        return set()
    return set(pd.read_csv(path)["gem_phase_id"].astype(str))


def fleet_for(
    gwpt: pd.DataFrame,
    country: str,
    year: int | None,
    exclusions: set[str] | frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Operating, geolocated projects for one country, optionally as of a year.

    Args:
        gwpt: The tracker's ``Data`` sheet.
        country: A region code in :data:`COUNTRY_NAME`.
        year: Keep projects started by and not retired in this year. None
            keeps every operating project.
        exclusions: GEM phase IDs to drop (:func:`load_exclusions`).

    Returns:
        Frame with ``lat``, ``lon`` and ``mw``.
    """
    name = COUNTRY_NAME.get(country.upper())
    if name is None:
        raise KeyError(f"No GWPT country name mapped for {country!r}")

    fleet = gwpt[gwpt["Country/Area"] == name].copy()
    fleet = fleet[fleet["Status"].astype(str).str.lower() == "operating"]
    fleet = fleet.dropna(subset=["Latitude", "Longitude", "Capacity (MW)"])

    if exclusions and "GEM phase ID" in fleet.columns:
        drop = fleet["GEM phase ID"].astype(str).isin(exclusions)
        if drop.any():
            names = ", ".join(fleet.loc[drop, "Project Name"].astype(str))
            print(f"  excluding {int(drop.sum())} curated GWPT record(s): {names}")
            fleet = fleet[~drop]

    if year is not None:
        start = pd.to_numeric(fleet["Start year"], errors="coerce")
        retired = pd.to_numeric(fleet["Retired year"], errors="coerce")
        # Projects with no start year are kept: the tracker often omits it for
        # older sites, and dropping them would understate the historic fleet.
        fleet = fleet[(start.isna() | (start <= year)) & (retired.isna() | (retired > year))]

    return fleet[["Latitude", "Longitude", "Capacity (MW)"]].rename(
        columns={"Latitude": "lat", "Longitude": "lon", "Capacity (MW)": "mw"}
    )


def operating_projects(gwpt: pd.DataFrame, country: str) -> pd.DataFrame:
    """Every operating row for one ``Country/Area``, compared after stripping."""
    return gwpt[
        (gwpt["Country/Area"].astype(str).str.strip() == country)
        & (gwpt["Status"].astype(str).str.lower() == "operating")
    ]


def plant_key(name: str, drop: frozenset[str], *, drop_roman: bool = False) -> str:
    """Reduce a plant name to a join key.

    Accents are removed, the name is upper-cased, anything not a letter, digit
    or space becomes a space, and the words in ``drop`` go. With
    ``drop_roman``, stage numerals (I to IV) go too, so that phases of one
    farm share a key.
    """
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9 ]", " ", s.upper())
    toks = [t for t in s.split() if t not in drop and not (drop_roman and _ROMAN.fullmatch(t))]
    return " ".join(toks).strip()


def projects_with_keys(gwpt: pd.DataFrame, country: str, key: Callable[[str], str]) -> pd.DataFrame:
    """Operating projects of one country with a join key and numeric capacity.

    Returns:
        Frame with ``Project Name``, ``norm`` (the key), ``cap`` (MW),
        ``Latitude`` and ``Longitude``.
    """
    g = operating_projects(gwpt, country).copy()
    g["norm"] = g["Project Name"].map(key)
    g["cap"] = pd.to_numeric(g["Capacity (MW)"], errors="coerce")
    return g[["Project Name", "norm", "cap", "Latitude", "Longitude"]]
