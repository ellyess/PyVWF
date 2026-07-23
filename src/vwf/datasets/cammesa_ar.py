"""Transforms for building the Argentina (CAMMESA) inputs.

Pure frame-to-frame logic for ``scripts/process_cammesa_ar.py``: turning the
CAMMESA per-central monthly *energy* series into monthly capacity factors, and
emitting the source metadata contract once coordinates AND capacities are
joined from an external register.

Everything here takes and returns DataFrames so it is testable without the raw
CAMMESA workbook; file I/O lives in the scripts.

The unit of observation is the **plant** (``obs_unit = "plant"``): CAMMESA's
renewables database reports generated energy per *central* (one row per plant,
one column per month), already at the monthly resolution PyVWF trains on — so
there is NO sub-hourly reshaping, timezone conversion, or DST handling at all,
the simplest ingest of any region.

The one thing CAMMESA does NOT carry is capacity: every sheet is energy (GWh).
So unlike every other adapter, the CF *denominator* is external — coordinates
and installed capacity both come from the Global Wind Power Tracker join in the
processing step. That makes the join load-bearing twice over, and gives a
built-in sanity check: a plant whose GWPT capacity is wrong shows an
implausible CF (a whole-fleet Argentine median CF sits near 0.38; a plant
whose median exceeds ``CAP_SUSPECT_CF`` almost certainly matched a too-small
GWPT capacity and must be re-curated, not trusted).
"""
from __future__ import annotations

from calendar import monthrange
from collections.abc import Sequence

import pandas as pd

#: Months with a monthly CF below this are treated as pre-operational when they
#: form a plant's leading run (the static-nameplate ramp trap; see
#: :func:`strip_commissioning_prefix`).
COMMISSIONING_CF = 0.02

#: A plant whose MEDIAN monthly CF exceeds this has almost certainly been given
#: too small a capacity by the GWPT join (real Argentine wind farms top out
#: around 0.5-0.6 monthly CF). Such plants are flagged for re-curation rather
#: than trusted — a capacity that is too low inflates CF above what any wind
#: resource delivers.
CAP_SUSPECT_CF = 0.65


def monthly_cf_from_gwh(
    gwh: pd.DataFrame,
    capacity: pd.DataFrame,
    year_start: int,
    year_end: int,
) -> pd.DataFrame:
    """Monthly capacity factor from per-plant monthly energy and a capacity join.

    ``CF = MWh / (capacity_MW * hours_in_month)`` with ``MWh = GWh * 1000`` and
    ``hours_in_month = days * 24``. Plants without a joined capacity are
    dropped here (the processing step is responsible for failing loudly on
    them first).

    Args:
        gwh: Long frame with ``ID``, ``year``, ``month``, ``gwh``.
        capacity: Frame with ``ID`` and ``capacity_mw``.
        year_start, year_end: Inclusive year bounds.

    Returns:
        Wide frame ``ID``, ``year``, ``obs_1``..``obs_12`` (monthly CF; missing
        months NaN).
    """
    df = gwh.copy()
    df["ID"] = df["ID"].astype(str)
    cap = capacity.copy()
    cap["ID"] = cap["ID"].astype(str)
    df = df.merge(cap[["ID", "capacity_mw"]], on="ID", how="left")
    df["capacity_mw"] = pd.to_numeric(df["capacity_mw"], errors="coerce")
    df["gwh"] = pd.to_numeric(df["gwh"], errors="coerce")
    df = df[df["capacity_mw"].notna() & (df["capacity_mw"] > 0)]
    df = df[(df["year"] >= int(year_start)) & (df["year"] <= int(year_end))]
    if df.empty:
        return pd.DataFrame(columns=["ID", "year"] + [f"obs_{m}" for m in range(1, 13)])

    days = df.apply(lambda r: monthrange(int(r["year"]), int(r["month"]))[1], axis=1)
    df["cf"] = df["gwh"] * 1000.0 / (df["capacity_mw"] * days * 24.0)

    wide = (
        df.pivot_table(index=["ID", "year"], columns="month", values="cf",
                       aggfunc="first")
        .reindex(columns=range(1, 13))
        .reset_index()
    )
    wide.columns = ["ID", "year"] + [f"obs_{m}" for m in range(1, 13)]
    return wide


def capacity_suspect_ids(
    wide: pd.DataFrame, *, threshold: float = CAP_SUSPECT_CF
) -> list[str]:
    """Plant IDs whose median monthly CF exceeds ``threshold``.

    A too-low capacity from the join inflates every month's CF, so an
    implausibly high median is the signature of a bad capacity match — the
    Argentina analogue of the Chile ``build_cl_metadata`` loud-failure guard,
    but for the denominator rather than the location.
    """
    oc = [f"obs_{m}" for m in range(1, 13)]
    med = wide.set_index("ID")[oc].stack().groupby(level=0).median()
    return sorted(med[med > threshold].index.astype(str))


def strip_commissioning_prefix(
    wide: pd.DataFrame, *, threshold: float = COMMISSIONING_CF
) -> pd.DataFrame:
    """NaN each plant's leading months before it is operational.

    A farm that came online mid-window reports near-zero energy before it runs;
    against the static GWPT nameplate that reads as CF~0. This strips the
    leading run of sub-``threshold`` months per plant; later low months are
    kept (a real low-wind or partial-availability month, not a fleet-entry
    artefact). Note this does NOT fix the *partial-build* ramp — a farm at
    half its turbines for a year reads at half CF against the full nameplate;
    a below-plateau mask is the named follow-up (cf. the NZ capacity history).

    Args:
        wide: ``ID``, ``year``, ``obs_1``..``obs_12`` frame.
        threshold: monthly CF below which a leading month is pre-operational.

    Returns:
        The frame with leading pre-operational months set to NaN.
    """
    out = wide.sort_values(["ID", "year"]).copy()
    for _, block in out.groupby("ID"):
        cells = [(row.name, f"obs_{m}", row[f"obs_{m}"])
                 for _, row in block.iterrows() for m in range(1, 13)]
        for idx, col, val in cells:
            if pd.notna(val) and val >= threshold:
                break
            if pd.notna(val):
                out.at[idx, col] = float("nan")
    return out


def build_ar_metadata(
    fleet: pd.DataFrame,
    join: pd.DataFrame,
    *,
    height: float,
    model: str,
    exclude: Sequence[str] = (),
) -> pd.DataFrame:
    """Emit the CAMMESAArgentinaSource metadata contract from the GWPT join.

    Both coordinates and capacity come from ``join`` (built in the processing
    step); CAMMESA carries neither. ``height``/``model`` are uniform defaults
    recorded in ``*_source`` columns (no hub height or turbine model — the same
    follow-up as AU/BR/US/CL).

    A plant that survives to here with no coordinate OR no capacity is a hard
    error, not a silent drop: without a coordinate it is mis-located, without a
    capacity it has no CF denominator. Plants intentionally out of scope must
    be named in ``exclude``.

    Args:
        fleet: Frame with ``ID`` (and optional ``site_name``, ``region``,
            ``provincia``).
        join: Frame with ``ID``, ``lon``, ``lat``, ``capacity_mw``.
        height: Uniform hub-height default, metres.
        model: Uniform power-curve key (a column of ``power_curves.csv``).
        exclude: Plant IDs allowed to be missing from the join (dropped).

    Returns:
        ``ID``, ``site_name``, ``lon``, ``lat``, ``height``, ``capacity`` (kW),
        ``model``, ``type``, plus provenance ``height_source``,
        ``model_source``, ``region``, ``provincia``.
    """
    md = fleet.copy()
    md["ID"] = md["ID"].astype(str)
    if "site_name" not in md.columns:
        md["site_name"] = md["ID"]
    for col in ("region", "provincia"):
        if col not in md.columns:
            md[col] = ""
    j = join.copy()
    j["ID"] = j["ID"].astype(str)
    md = md.merge(j[["ID", "lon", "lat", "capacity_mw"]], on="ID", how="left")

    bad = md[md["lon"].isna() | md["lat"].isna()
             | md["capacity_mw"].isna() | (md["capacity_mw"] <= 0)]
    bad = bad[~bad["ID"].isin(set(exclude))]
    if len(bad):
        raise ValueError(
            f"{len(bad)} Argentina wind plants lack a coordinate or capacity "
            f"after the GWPT join and are not excluded: "
            f"{bad['site_name'].tolist()[:10]}. Add them to the override table "
            "(ID,lon,lat,capacity_mw) or the exclude list — never leave a plant "
            "without a location (mis-located) or capacity (no CF denominator)."
        )
    md = md[md["lon"].notna() & md["lat"].notna()
            & md["capacity_mw"].notna() & (md["capacity_mw"] > 0)].copy()

    md["height"] = float(height)
    md["height_source"] = "default-uniform"
    md["model_source"] = "default-uniform"
    md["model"] = model
    md["capacity"] = pd.to_numeric(md["capacity_mw"], errors="coerce") * 1000.0
    md["type"] = "onshore"
    return md[
        ["ID", "site_name", "lon", "lat", "height", "capacity", "model",
         "type", "height_source", "model_source", "region", "provincia"]
    ].reset_index(drop=True)
