"""Transforms for building the AU-NEM inputs from raw AEMO + GWPT data.

Pure frame-to-frame logic for ``scripts/process/aemo_au.py``: parsing AEMO
MMS archive CSVs, extracting the operating wind fleet from the Generation
Information workbook, joining coordinates from the Global Wind Power Tracker
(the user's public-source turbine compilation), and emitting the
``AEMONemSource`` metadata contract.

Everything here takes and returns DataFrames so it is testable without the
raw files or the xlsx reader; file I/O lives in the script.

Known limitations, stated up front (also in the join report the script
writes):

- GWPT carries no hub height or turbine model, so ``height`` and ``model``
  are uniform defaults recorded in ``height_source``/``model_source``
  columns. A vintage-aware assignment is a named follow-up, along with
  hub-height and turbine-vintage covariates generally.
- Capacity is the Generation Information nameplate, static over the window.
  Farms with STAGED commissioning inside 2020-2023 (e.g. multi-stage sites)
  have their early months biased low relative to the evolving registered
  capacity; the commissioning mask removes pre-FCUD months but not partial
  staging. DUDETAILSUMMARY-based capacity histories are the named follow-up
  before any seasonal-cycle result is trusted.
"""
from __future__ import annotations

import re

import pandas as pd

#: The NEM regions; the Generation Information workbook also lists WEM (WA)
#: projects in other files, but its Region column for this sheet is NEM-only.
NEM_REGIONS = ("NSW1", "QLD1", "SA1", "TAS1", "VIC1")


def parse_mms_table(lines) -> pd.DataFrame:
    """Parse one AEMO MMS C/I/D/F CSV into a DataFrame.

    MMS files carry a comment (``C``) header line, one ``I`` line naming the
    columns, ``D`` data rows, and a trailing ``C`` footer. The first four
    fields of the I/D rows are row-type bookkeeping and are dropped.

    Args:
        lines: An iterable of text lines (an open file object works).

    Returns:
        DataFrame with the I-row's column names (bookkeeping fields dropped).

    Raises:
        ValueError: If no I row is found, or the file holds more than one
            table (the DVD archive files hold exactly one).
    """
    frame = pd.read_csv(lines, header=None, skiprows=1, dtype=str, engine="c")
    kinds = frame.iloc[:, 0]

    header_rows = frame[kinds == "I"]
    if len(header_rows) == 0:
        raise ValueError("no 'I' header row found: not an MMS table file?")
    if len(header_rows) > 1:
        raise ValueError(
            f"{len(header_rows)} 'I' rows found; multi-table MMS files are not "
            "supported (the monthly DVD archives hold one table per file)"
        )

    columns = header_rows.iloc[0, 4:].dropna().tolist()
    data = frame[kinds == "D"].iloc[:, 4 : 4 + len(columns)]
    data.columns = columns
    return data.reset_index(drop=True)


def normalise_farm_name(name: str) -> str:
    """Reduce a wind-farm name to a join key.

    Lowercase, parentheticals removed, generic suffixes (wind farm / wf /
    project / stage-N) stripped, everything non-alphanumeric collapsed. Kept
    deliberately conservative: a missed match lands in the unmatched report
    for a manual alias, which is recoverable; a wrong merge is not.
    """
    s = str(name).lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"\b(wind\s*farms?|wind\s*project|wf)\b", " ", s)
    s = re.sub(r"\bstage\s*\d+\b", " ", s)
    # AEMO workbook annotation ("Key Connection Information"), never part of
    # a farm's actual name.
    s = re.sub(r"\bkci\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def wind_fleet_from_gen_info(gen_info: pd.DataFrame) -> pd.DataFrame:
    """Extract the operating wind fleet, one row per DUID.

    Args:
        gen_info: The "Generator Information" sheet (header row resolved by
            the caller). Uses ``Technology Type``, ``DUID``,
            ``Commitment Status``, ``Site Name``, ``Region``,
            ``Agg Nameplate Capacity (MW AC)``, ``Full Commercial Use Date``.

    Returns:
        Frame with ``ID``, ``site_name``, ``region``, ``capacity_mw``
        (summed over unit groups; 'Agg Nameplate' is per unit-group row,
        e.g. Boco Rock 9x1.60 + 58x1.70), and ``fcud``.
    """
    wind = gen_info[
        gen_info["Technology Type"].astype(str).str.strip().str.lower().eq("wind")
        & gen_info["DUID"].notna()
        & (gen_info["DUID"].astype(str).str.strip() != "")
        & gen_info["Commitment Status"]
        .astype(str)
        .str.contains("In Service|In Commissioning", case=False, na=False)
    ].copy()

    wind["DUID"] = wind["DUID"].astype(str).str.strip()
    wind["capacity_mw"] = pd.to_numeric(
        wind["Agg Nameplate Capacity (MW AC)"], errors="coerce"
    )
    wind["fcud"] = pd.to_datetime(wind["Full Commercial Use Date"], errors="coerce")

    fleet = (
        wind.groupby("DUID")
        .agg(
            site_name=("Site Name", "first"),
            region=("Region", "first"),
            capacity_mw=("capacity_mw", "sum"),
            fcud=("fcud", "min"),
        )
        .reset_index()
        .rename(columns={"DUID": "ID"})
    )
    return fleet[fleet["region"].isin(NEM_REGIONS)].reset_index(drop=True)


def farms_from_gwpt(gwpt: pd.DataFrame) -> pd.DataFrame:
    """Collapse GWPT AU operating rows to one row per project.

    Multi-phase projects become a capacity-weighted centroid with summed
    capacity and the earliest start year.

    Args:
        gwpt: The GWPT "Data" sheet. Uses ``Country/Area``, ``Status``,
            ``Project Name``, ``Capacity (MW)``, ``Latitude``, ``Longitude``,
            ``Start year``.
    """
    au = gwpt[
        gwpt["Country/Area"].astype(str).str.contains("Australia", case=False, na=False)
        & gwpt["Status"].astype(str).str.strip().str.lower().eq("operating")
    ].copy()

    au["capacity_mw"] = pd.to_numeric(au["Capacity (MW)"], errors="coerce")
    au["lat"] = pd.to_numeric(au["Latitude"], errors="coerce")
    au["lon"] = pd.to_numeric(au["Longitude"], errors="coerce")
    au = au.dropna(subset=["capacity_mw", "lat", "lon"])
    au["_wlat"] = au["lat"] * au["capacity_mw"]
    au["_wlon"] = au["lon"] * au["capacity_mw"]

    farms = (
        au.groupby("Project Name")
        .agg(
            gwpt_capacity_mw=("capacity_mw", "sum"),
            _wlat=("_wlat", "sum"),
            _wlon=("_wlon", "sum"),
            start_year=("Start year", "min"),
        )
        .reset_index()
        .rename(columns={"Project Name": "gwpt_name"})
    )
    farms["lat"] = farms["_wlat"] / farms["gwpt_capacity_mw"]
    farms["lon"] = farms["_wlon"] / farms["gwpt_capacity_mw"]
    return farms.drop(columns=["_wlat", "_wlon"])


def join_fleet_to_gwpt(
    fleet: pd.DataFrame,
    gwpt_farms: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Join DUIDs to GWPT coordinates by normalised name.

    Args:
        fleet: Output of :func:`wind_fleet_from_gen_info`.
        gwpt_farms: Output of :func:`farms_from_gwpt`.

    Returns:
        ``(matched, unmatched_fleet, unmatched_gwpt)``. ``matched`` carries
        both capacities so gross mismatches (wrong-farm joins) are visible in
        the report; no automatic fuzzy matching, so misses go to the report,
        and human-approved DUID aliases (:func:`resolve_duid_aliases`) handle
        the tail.
    """
    fleet = fleet.copy()
    gwpt_farms = gwpt_farms.copy()
    fleet["_key"] = fleet["site_name"].map(normalise_farm_name)
    gwpt_farms["_key"] = gwpt_farms["gwpt_name"].map(normalise_farm_name)

    dup = gwpt_farms[gwpt_farms["_key"].duplicated(keep=False)]
    if len(dup):
        raise ValueError(
            "GWPT project names collide after normalisation; disambiguate "
            f"before joining: {sorted(dup['gwpt_name'])}"
        )

    matched = fleet.merge(gwpt_farms, on="_key", how="inner")
    unmatched_fleet = fleet[~fleet["_key"].isin(gwpt_farms["_key"])].drop(columns="_key")
    unmatched_gwpt = gwpt_farms[~gwpt_farms["_key"].isin(fleet["_key"])].drop(
        columns="_key"
    )
    matched = matched.drop(columns="_key")
    matched["match_source"] = "name"
    return matched, unmatched_fleet, unmatched_gwpt


def resolve_duid_aliases(
    unmatched_fleet: pd.DataFrame,
    alias_map: dict[str, str],
    gwpt_data: pd.DataFrame,
    gwpt_below: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve human-approved DUID aliases to GWPT rows (phase-aware).

    Aliases are keyed by DUID (site names are not unique across stages) and
    map to one or more semicolon-separated targets:

    - ``"Project Name"``: all phases of a GWPT Data project, ANY status
      (the alias itself is the human approval, so GWPT status lag, e.g.
      Golden Plains East listed "mothballed", does not block);
    - ``"Project Name|Phase"``: one phase row (per-phase coordinates);
    - ``"BT:Project Name"``: a Below-Threshold sheet row.

    One DUID with several targets (Woolnorth = Studland Bay + Bluff Point)
    gets the capacity-weighted centroid of the target rows. GI capacity
    remains authoritative throughout; target capacity is context only.

    Returns:
        ``(alias_matched, still_unmatched)`` where ``alias_matched`` has the
        same shape as the name-join output plus ``match_source="alias"``.

    Raises:
        ValueError: If an alias target matches no GWPT/Below-Threshold row.
            A typo in a human-approved alias must fail loudly, not silently
            drop a farm.
    """
    au = gwpt_data[
        gwpt_data["Country/Area"].astype(str).str.contains("Australia", na=False)
    ].copy()
    au_below = None
    if gwpt_below is not None:
        au_below = gwpt_below[
            gwpt_below["Country/Area"].astype(str).str.contains("Australia", na=False)
        ].copy()

    def target_rows(token: str) -> pd.DataFrame:
        token = token.strip()
        if token.startswith("BT:"):
            if au_below is None:
                raise ValueError(f"alias target {token!r} needs the Below Threshold sheet")
            sel = au_below[au_below["Project Name"].astype(str) == token[3:]]
        else:
            name, _, phase = token.partition("|")
            sel = au[au["Project Name"].astype(str) == name]
            if phase:
                sel = sel[sel["Phase Name"].astype(str).str.strip() == phase.strip()]
        if sel.empty:
            raise ValueError(f"alias target {token!r} matched no GWPT row")
        return sel

    rows, resolved_ids = [], []
    for duid, targets in alias_map.items():
        fleet_row = unmatched_fleet[unmatched_fleet["ID"] == duid]
        if fleet_row.empty:
            continue  # already matched by name, or not in this fleet subset
        fleet_row = fleet_row.iloc[0]

        sel = pd.concat([target_rows(t) for t in str(targets).split(";")])
        cap = pd.to_numeric(sel["Capacity (MW)"], errors="coerce").fillna(0.0)
        lat = pd.to_numeric(sel["Latitude"], errors="coerce")
        lon = pd.to_numeric(sel["Longitude"], errors="coerce")
        # Capacity-weighted centroid; uniform weights if capacities are absent.
        weights = cap if cap.sum() > 0 else pd.Series(1.0, index=cap.index)
        start = pd.to_numeric(sel.get("Start year"), errors="coerce").min()

        rows.append(
            {
                "ID": duid,
                "site_name": fleet_row["site_name"],
                "region": fleet_row["region"],
                "capacity_mw": fleet_row["capacity_mw"],
                "fcud": fleet_row["fcud"],
                "gwpt_name": str(targets),
                "gwpt_capacity_mw": float(cap.sum()),
                "lat": float((lat * weights).sum() / weights.sum()),
                "lon": float((lon * weights).sum() / weights.sum()),
                "start_year": start,
                "match_source": "alias",
            }
        )
        resolved_ids.append(duid)

    alias_matched = pd.DataFrame(rows)
    still_unmatched = unmatched_fleet[~unmatched_fleet["ID"].isin(resolved_ids)]
    return alias_matched, still_unmatched


def registered_capacity_history(dudetail: pd.DataFrame) -> pd.DataFrame:
    """Effective-dated registered capacity per DUID from DUDETAIL rows.

    The monthly archives repeat full-history rows; duplicates collapse and
    the highest VERSIONNO wins per (DUID, EFFECTIVEDATE).

    Args:
        dudetail: Concatenated DUDETAIL tables with at least ``DUID``,
            ``EFFECTIVEDATE``, ``VERSIONNO``, ``REGISTEREDCAPACITY``.

    Returns:
        Frame with ``DUID``, ``EFFECTIVEDATE`` (datetime), and
        ``REGISTEREDCAPACITY`` (float), sorted by DUID then date.
    """
    hist = dudetail[["DUID", "EFFECTIVEDATE", "VERSIONNO", "REGISTEREDCAPACITY"]].copy()
    hist = hist.drop_duplicates()
    hist["EFFECTIVEDATE"] = pd.to_datetime(hist["EFFECTIVEDATE"])
    hist["VERSIONNO"] = pd.to_numeric(hist["VERSIONNO"], errors="coerce")
    hist["REGISTEREDCAPACITY"] = pd.to_numeric(hist["REGISTEREDCAPACITY"], errors="coerce")
    hist = (
        hist.sort_values("VERSIONNO")
        .groupby(["DUID", "EFFECTIVEDATE"], as_index=False)
        .last()
        .drop(columns="VERSIONNO")
    )
    return hist.sort_values(["DUID", "EFFECTIVEDATE"]).reset_index(drop=True)


def capacity_mask_months(
    hist: pd.DataFrame,
    scada_months: pd.DataFrame,
    *,
    tolerance: float = 0.02,
) -> pd.DataFrame:
    """Months whose registered capacity is unreliable as a CF denominator.

    A month is masked when the following holds. The rule deliberately
    prefers clean denominators over data retention; the measured cost is
    about 1% of farm-months.

    - the registered capacity changes WITHIN the month (ambiguous
      denominator),
    - the month's capacity is below ``(1 - tolerance) ×`` the final
      registered capacity (farm not fully built: a ramping farm injects a
      spurious sub-annual signal into exactly the seasonal cycle the
      Australia validation measures), or
    - the month predates the DUID's first registration (trial generation).

    Args:
        hist: Output of :func:`registered_capacity_history`.
        scada_months: Frame with ``ID``, ``year``, ``month``: the months
            that actually carry SCADA (e.g. the partials table).

    Returns:
        Frame with ``ID``, ``year``, ``month``, ``reason``.
    """
    rows = []
    months = scada_months[["ID", "year", "month"]].drop_duplicates()
    for duid, group in months.groupby("ID"):
        dh = hist[hist["DUID"] == str(duid)]
        if dh.empty:
            # No registration history at all: nothing to judge against; the
            # caller keeps static nameplate (capacity_history="absent").
            continue
        final_cap = dh["REGISTEREDCAPACITY"].iloc[-1]
        for _, r in group.iterrows():
            start = pd.Timestamp(int(r["year"]), int(r["month"]), 1)
            end = start + pd.offsets.MonthEnd(1)
            at_start = dh[dh["EFFECTIVEDATE"] <= start]["REGISTEREDCAPACITY"]
            at_end = dh[dh["EFFECTIVEDATE"] <= end]["REGISTEREDCAPACITY"]
            if at_start.empty:
                reason = "pre-registration"
            elif at_start.iloc[-1] != at_end.iloc[-1]:
                reason = "mid-month-change"
            elif at_end.iloc[-1] < (1.0 - tolerance) * final_cap:
                reason = "below-final-build"
            else:
                continue
            rows.append(
                {"ID": str(duid), "year": int(r["year"]), "month": int(r["month"]),
                 "reason": reason}
            )
    return pd.DataFrame(rows, columns=["ID", "year", "month", "reason"])


def build_au_metadata(
    matched: pd.DataFrame,
    *,
    height: float,
    model: str,
) -> pd.DataFrame:
    """Emit the AEMONemSource metadata contract from the joined fleet.

    Capacity comes from the Generation Information nameplate (MW → kW, the
    source contract unit, matching the SCADA registration basis, not GWPT).
    Commissioning is FCUD where present, else 1 January of the GWPT start
    year. ``height``/``model`` are uniform defaults; their ``*_source``
    columns say so, loudly, so no downstream step can mistake them for data.
    """
    out = pd.DataFrame(
        {
            "ID": matched["ID"].astype(str),
            "site_name": matched["site_name"],
            "region": matched["region"],
            "lon": matched["lon"].astype(float),
            "lat": matched["lat"].astype(float),
            "height": float(height),
            "capacity": matched["capacity_mw"].astype(float) * 1000.0,
            "model": model,
            "type": "onshore",
            "gwpt_capacity_mw": matched["gwpt_capacity_mw"].astype(float),
            "start_year": matched["start_year"],
            "height_source": "default-uniform",
            "model_source": "default-uniform",
        }
    )

    fcud = pd.to_datetime(matched["fcud"], errors="coerce")
    fallback = pd.to_datetime(
        matched["start_year"].astype("Int64").astype(str) + "-01-01",
        errors="coerce",
    )
    out["commissioning_date"] = fcud.fillna(fallback)
    return out
