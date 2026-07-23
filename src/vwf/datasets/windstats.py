"""Transforms for the CONFIDENTIAL WindStats regions (ES now; SE/FI later).

⚠ CONFIDENTIAL / COMMERCIAL (WindStats). The generation and turbine-spec data
is a licensed WindStats extract; neither it nor anything derived from it may be
committed or redistributed. `input/` is git-ignored, and these transforms only
reshape files the user already holds locally. The COORDINATES for Spain are
joined from the open Global Wind Power Tracker (CC-BY-4.0), so Spain is a
**mixed-licence** region: confidential generation + open coordinates.

WindStats gives, per country, a metadata table, a monthly-generation table, and
a `geolocate.<country>.csv` that maps the WindStats id/name (`ws`) to a
thewindpower.net farm name (`twp`), but NOT coordinates. Coordinates come from
matching `twp` to a coordinate source:

- **Spain**: GWPT covers the Spanish fleet well, so `twp` is fuzzy-matched to
  GWPT project names (:func:`match_twp_to_coords`).
- **Sweden / Finland**: GWPT under-covers their fragmented fleets (7-9% match),
  so they need the thewindpower.net coordinate table the `twp` names point at;
  the same `match_twp_to_coords` accepts any `twp -> lon/lat` frame, so they
  slot in once that table is supplied.

Per-country column layouts differ (WindStats is not uniform), captured in
:data:`SCHEMA`.
"""
from __future__ import annotations

import re
import unicodedata
from calendar import monthrange
from collections.abc import Sequence

import pandas as pd

#: Per-country WindStats column layout. ``id`` is the metadata key column;
#: ``link`` is the metadata column the geolocate ``ws`` matches (the id itself,
#: or a separate ``Location`` column). ``onshore_default`` is used where the
#: country carries no on/offshore flag.
SCHEMA: dict[str, dict] = {
    "ES": {"id": "Unnamed: 0", "cap": "kW", "rotor": "Rotor (m)",
           "tower": "Tower (m)", "manuf": "Manufacturer", "onoff": None,
           "link": "id"},
    "SE": {"id": "Unnamed: 0", "cap": "kW", "rotor": "Rotor (m)",
           "tower": "Tower (m)", "manuf": "Manufacturer", "onoff": "On/offshore",
           "link": "Location"},
    "FI": {"id": "Unnamed: 0", "cap": "kW", "rotor": "Rotor (m)",
           "tower": "Tower (m)", "manuf": "Manufacturer", "onoff": "On/offshore",
           "link": "id"},
}

_JUNK = {"I", "II", "III", "IV", "DE", "LA", "EL", "THE", "OF", "AND"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    for j in ("WIND FARM", "WINDFARM", "WIND POWER", "PARQUE EOLICO", "WINDPARK",
              "PARK", "PARQUE"):
        s = s.replace(j, " ")
    toks = [t for t in re.sub(r"[^A-Z0-9 ]", " ", s).split() if t not in _JUNK]
    return " ".join(toks).strip()


def windstats_metadata(md: pd.DataFrame, country: str) -> pd.DataFrame:
    """Normalise a WindStats country metadata table to common columns.

    Returns ``ID`` (string), ``link`` (the value the geolocate ``ws`` matches),
    ``capacity`` (kW), ``diameter`` (m), ``height`` (m), ``model``, ``type``.
    """
    sc = SCHEMA[country]
    df = md.rename(columns={sc["id"]: "ID"}).copy()
    df["ID"] = df["ID"].astype(str).str.strip()
    link_col = "ID" if sc["link"] == "id" else sc["link"]
    df["link"] = df[link_col].astype(str).str.strip()
    df["capacity"] = pd.to_numeric(df[sc["cap"]], errors="coerce")
    df["diameter"] = pd.to_numeric(df[sc["rotor"]], errors="coerce")
    df["height"] = pd.to_numeric(df[sc["tower"]], errors="coerce")
    df["model"] = df[sc["manuf"]].astype(str)
    if sc["onoff"] and sc["onoff"] in df.columns:
        off = df[sc["onoff"]].astype(str).str.lower().str.contains("off")
        df["type"] = off.map({True: "offshore", False: "onshore"})
    else:
        df["type"] = "onshore"
    return df[["ID", "link", "capacity", "diameter", "height", "model", "type"]]


def windstats_monthly_cf(
    data: pd.DataFrame,
    capacity: pd.DataFrame,
    year_start: int,
    year_end: int,
) -> pd.DataFrame:
    """Monthly capacity factor from WindStats monthly output (kWh).

    ``CF = Output_kWh / (capacity_kW x days x 24)``. Rows with no positive
    capacity are dropped.

    Args:
        data: WindStats generation: ``ID``, ``Year``, ``Month``, ``Output``.
        capacity: ``ID`` -> ``capacity`` (kW).
        year_start, year_end: Inclusive year bounds.

    Returns:
        Wide ``ID``, ``year``, ``obs_1``..``obs_12`` (monthly CF; missing NaN).
    """
    df = data.rename(columns={"Year": "year", "Month": "month",
                              "Output": "output"}).copy()
    df["ID"] = df["ID"].astype(str).str.strip()
    cap = capacity.copy()
    cap["ID"] = cap["ID"].astype(str).str.strip()
    df = df.merge(cap[["ID", "capacity"]], on="ID", how="left")
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
    df["output"] = pd.to_numeric(df["output"], errors="coerce")
    df = df[df["capacity"].notna() & (df["capacity"] > 0)]
    df = df[(df["year"] >= int(year_start)) & (df["year"] <= int(year_end))]
    if df.empty:
        return pd.DataFrame(columns=["ID", "year"] + [f"obs_{m}" for m in range(1, 13)])
    days = df.apply(lambda r: monthrange(int(r["year"]), int(r["month"]))[1], axis=1)
    df["cf"] = df["output"] / (df["capacity"] * days * 24.0)
    wide = (df.pivot_table(index=["ID", "year"], columns="month", values="cf",
                           aggfunc="mean")
            .reindex(columns=range(1, 13)).reset_index())
    wide.columns = ["ID", "year"] + [f"obs_{m}" for m in range(1, 13)]
    return wide


def match_twp_to_coords(
    geolocate: pd.DataFrame,
    coords: pd.DataFrame,
    *,
    coord_name_col: str = "Project Name",
    lon_col: str = "Longitude",
    lat_col: str = "Latitude",
) -> pd.DataFrame:
    """Match the WindStats ``ws`` key to lon/lat via its ``twp`` name.

    ``geolocate`` maps ``ws`` (the WindStats id/link) to ``twp`` (the
    thewindpower farm name). ``coords`` is any farm coordinate table (GWPT for
    Spain, or a thewindpower export for SE/FI) with a name column. Names are
    matched exactly on a normalised form, then by substring either direction.

    Returns:
        ``ws``, ``lon``, ``lat``, ``matched_name`` for the matched rows only
        (unmatched ``ws`` are absent; the processor reports them).
    """
    c = coords.copy()
    c["_n"] = c[coord_name_col].map(_norm)
    exact = {n: (lo, la) for n, lo, la in zip(c["_n"], c[lon_col], c[lat_col]) if n}
    names = [n for n in exact]
    rows = []
    for ws, twp in zip(geolocate["ws"].astype(str), geolocate["twp"].astype(str)):
        n = _norm(twp)
        hit = exact.get(n)
        matched = n
        if hit is None:
            cand = [gn for gn in names
                    if gn and min(len(gn), len(n)) >= 4 and (gn in n or n in gn)]
            if cand:
                matched = min(cand, key=lambda x: abs(len(x) - len(n)))
                hit = exact[matched]
        if hit is not None:
            rows.append((str(ws).strip(), hit[0], hit[1], matched))
    return pd.DataFrame(rows, columns=["ws", "lon", "lat", "matched_name"]).drop_duplicates("ws")


def build_windstats_metadata(
    station_md: pd.DataFrame,
    ws_coords: pd.DataFrame,
    *,
    height_default: float = 80.0,
    model: str = "2019COE_Market_Average_2.6MW_121",
    exclude: Sequence[str] = (),
) -> pd.DataFrame:
    """Join coordinates onto WindStats metadata; fail loudly on the coordinateless.

    Args:
        station_md: Output of :func:`windstats_metadata`.
        ws_coords: Output of :func:`match_twp_to_coords` (``ws``, ``lon``, ``lat``).
        height_default: Hub height (m) where WindStats gives none.
        model: Uniform power-curve key where the WindStats manufacturer string
            is not a library key (matched downstream by specific power).
        exclude: ``link`` values allowed to have no coordinate (dropped).

    Returns:
        ``ID``, ``lon``, ``lat``, ``height``, ``capacity`` (kW), ``model``,
        ``type``, plus provenance ``height_source``, ``model_source``,
        ``coord_source``.
    """
    md = station_md.copy()
    md = md.merge(ws_coords.rename(columns={"ws": "link"})[["link", "lon", "lat"]],
                  on="link", how="left")
    bad = md[(md["lon"].isna() | md["lat"].isna()) & ~md["link"].isin(set(exclude))]
    if len(bad):
        raise ValueError(
            f"{len(bad)} WindStats turbines ({bad['link'].nunique()} farms) have "
            "no coordinate after the twp->coords match and are not excluded. "
            "Add their farms to the coordinate table or the exclude list; never "
            "leave a turbine coordinateless (it would be mis-located)."
        )
    md = md[md["lon"].notna() & md["lat"].notna()
            & md["capacity"].notna() & (md["capacity"] > 0)].copy()
    has_height = pd.to_numeric(md["height"], errors="coerce") > 1
    md["height_source"] = has_height.map({True: "windstats", False: "default-uniform"})
    md["height"] = pd.to_numeric(md["height"], errors="coerce").where(
        has_height, float(height_default))
    md["diameter"] = pd.to_numeric(md["diameter"], errors="coerce")
    # WindStats manufacturer strings are not curve-library keys (and are often
    # mojibake), so the curve key is the uniform default (as AU/BR/CL/AR do for
    # metadata-poor regions) and the raw string is kept as provenance. Diameter
    # is retained so a specific-power match can upgrade this downstream.
    md["windstats_manufacturer"] = md["model"]
    md["model"] = model
    md["model_source"] = "default-uniform"
    md["coord_source"] = "gwpt-open"
    return md[["ID", "lon", "lat", "height", "capacity", "diameter", "model",
               "type", "height_source", "model_source", "coord_source",
               "windstats_manufacturer"]].reset_index(drop=True)
