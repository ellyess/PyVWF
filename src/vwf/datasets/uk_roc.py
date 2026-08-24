"""Transforms for building the UK (Ofgem ROC + REPD) inputs.

Pure frame-to-frame logic for ``scripts/process/uk.py``. The UK region is
served by the shared ``european-turbine`` adapter, which reads two CSVs for
Great Britain:

    uk_md.csv    per-turbine metadata (one row per turbine, pseudo-replicated
                 from the accredited generating station: every turbine of a
                 station shares the station's specs and an equal share of its
                 generation). ID = ``<ROC accreditation no.>-<turbine index>``.
    ukobs.csv    monthly generation (columns ``1``..``12`` kWh + ``year``),
                 identical across a station's turbine rows.

The two observed-source components (both described in the co-authored UK
methodology) and their reproducibility:

- **Observations: Ofgem Renewables Obligation certificates.** Accredited wind
  stations receive monthly ROCs; energy is recovered as ``MWh = ROCs / band``,
  where the band is grandfathered by accreditation vintage (:data:`ROC_BANDING`
  below, from Ofgem's RO Guidance for Generators, Appendix 4). The per-station,
  per-month issuance is only available via Ofgem's Renewable Electricity
  Register export (gated; the login-free historical bulk file is
  technology-aggregate only), so :func:`roc_issuance_to_station_monthly` takes
  a user-supplied export and is documented, not auto-fetched.
- **Metadata: reconstructed from open data.** The UK Renewable Energy Planning
  Database (REPD, OGL, gov.uk) gives per-site location (OSGB, reprojected
  here), installed capacity, turbine count, and a partial tip height, but NOT
  turbine model, rotor diameter, or (mostly) hub height. So the open
  reconstruction (:func:`repd_wind_metadata`) is a location/capacity table with
  uniform-default specs, the same approach the AU/BR/CL/AR adapters take;
  :func:`divergence_report` quantifies how it differs from the committed
  curated ``uk_md.csv`` (which carries real per-station models and heights from
  a source that is not openly redistributable).
"""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

#: ROCs issued per MWh, by technology and accreditation vintage (grandfathered
#: for the station's life). From Ofgem "Renewables Obligation (RO): Guidance
#: for Generators", Appendix 4, Table A4.1 (RO/ROS). Before 1 April 2009 it was
#: a flat 1.0 ROC/MWh for all technologies. Keyed by the RO year (April-March)
#: in which the station's capacity was accredited; ``default`` applies when the
#: accreditation vintage is unknown (onshore RO wind is 0.9 for all 2013/14+
#: vintages, which is the overwhelming majority).
ROC_BANDING: dict[str, dict] = {
    "onshore": {"pre_2013": 1.0, "from_2013": 0.9, "default": 0.9},
    "offshore": {"to_2014_15": 2.0, "2015_16": 1.9, "post_2016": 1.8,
                 "default": 2.0},
}


def band_for(technology: str, accreditation_year: int | None) -> float:
    """ROCs/MWh for a station, given its technology and accreditation RO year.

    Args:
        technology: ``"onshore"`` or ``"offshore"`` (case/label tolerant).
        accreditation_year: The calendar year the station's capacity was
            accredited (start of the RO year). ``None`` uses the technology
            default.

    Returns:
        The grandfathered banding factor (ROCs per MWh).
    """
    tech = "offshore" if "off" in str(technology).lower() else "onshore"
    b = ROC_BANDING[tech]
    if accreditation_year is None:
        return b["default"]
    if tech == "onshore":
        return b["pre_2013"] if accreditation_year < 2013 else b["from_2013"]
    if accreditation_year <= 2014:
        return b["to_2014_15"]
    return b["2015_16"] if accreditation_year == 2015 else b["post_2016"]


def osgb_to_wgs84(easting: pd.Series, northing: pd.Series) -> pd.DataFrame:
    """Reproject OSGB36 (EPSG:27700) easting/northing to WGS84 lon/lat.

    REPD gives X/Y in OSGB National Grid metres; the pipeline needs degrees.
    """
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    x = pd.to_numeric(easting, errors="coerce")
    y = pd.to_numeric(northing, errors="coerce")
    lon, lat = tf.transform(x.values, y.values)
    return pd.DataFrame({"lon": lon, "lat": lat}, index=easting.index)


# --------------------------------------------- CONFIDENTIAL Ofgem export path
#: Column map for the Ofgem "CertificatesExternalPublicDataWarehouse" report
#: export (the confidential per-station certificate warehouse). The SSRS report
#: names columns ``textbox<n>``; a blank title line means the header sits on
#: row index 2. Mapped by those stable textbox ids, not by position.
_OFGEM_COLS = {
    "textbox4": "station", "textbox13": "name", "textbox5": "scheme",
    "textbox15": "technology", "textbox18": "period", "textbox21": "certs",
    "textbox37": "mwh_per_cert", "textbox33": "status",
}


def read_ofgem_confidential_certificates(
    paths: Sequence[str], *, scheme: str = "RO",
) -> pd.DataFrame:
    """Station-monthly MWh from the CONFIDENTIAL Ofgem certificate warehouse.

    CONFIDENTIAL / DIFFERENTLY LICENSED. These per-station certificate
    exports are not the open REPD/RER path: they carry their own licence and
    must never be committed or redistributed. This reader only transforms
    files the user already holds locally.

    Unlike the open RER route (:func:`roc_issuance_to_station_monthly`, which
    applies a banding lookup), this export carries the exact MWh-per-certificate
    factor per row (``textbox37``), so energy is recovered directly and every
    grandfathered vintage is handled exactly: ``MWh = certificates x factor``.
    Only ``scheme`` certificates are kept (RO by default, since REGO would
    double-count the same generation), and **revoked** certificates are dropped.

    Args:
        paths: One or more Ofgem warehouse CSV exports (per RO year).
        scheme: Certificate scheme to keep (``"RO"``; the committed ukobs is
            RO-derived).

    Returns:
        Long frame ``ID`` (station), ``year``, ``month``, ``mwh``.
    """
    frames = []
    for p in paths:
        df = pd.read_csv(p, header=2, dtype=str, low_memory=False)
        df = df.rename(columns=_OFGEM_COLS)
        keep = [c for c in _OFGEM_COLS.values() if c in df.columns]
        frames.append(df[keep])
    cat = pd.concat(frames, ignore_index=True)
    cat = cat[cat["scheme"].astype(str).str.strip() == scheme]
    cat = cat[cat["status"].astype(str).str.strip().str.lower() != "revoked"]
    cat["ID"] = cat["station"].astype(str).str.strip()
    ts = pd.to_datetime(cat["period"], format="%b-%Y", errors="coerce")
    cat = cat[ts.notna()].copy()
    cat["year"], cat["month"] = ts.dt.year, ts.dt.month
    certs = pd.to_numeric(cat["certs"], errors="coerce").fillna(0.0)
    factor = pd.to_numeric(cat["mwh_per_cert"], errors="coerce").fillna(0.0)
    cat["mwh"] = certs * factor
    return (cat.groupby(["ID", "year", "month"])["mwh"].sum()
            .reset_index())


# --------------------------------------------------------------- observations
def roc_issuance_to_station_monthly(
    roc: pd.DataFrame,
    *,
    station_col: str,
    period_col: str,
    certs_col: str,
    tech_col: str | None = None,
    accreditation_year: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Convert a per-station ROC issuance export to station-monthly MWh.

    Args:
        roc: Ofgem RER issuance export: one row (or several) per station and
            output period, with a certificate count. Column names vary between
            exports, so they are passed explicitly.
        station_col: Column holding the station accreditation number
            (e.g. ``R00116SQSC``).
        period_col: Column holding the output period, parseable to a date
            (month resolution).
        certs_col: Column holding the number of ROCs issued.
        tech_col: Optional technology column; if absent the accreditation
            number's technology code is used (``RQ``/``SQ``/``NQ`` = onshore,
            ``RP``/``SP``/``NP`` = offshore).
        accreditation_year: Optional ``{station: RO accreditation year}`` for
            exact grandfathered banding; unknown stations use the tech default.

    Returns:
        Long frame ``ID`` (station), ``year``, ``month``, ``mwh``.
    """
    df = roc.copy()
    df["ID"] = df[station_col].astype(str).str.strip()
    ts = pd.to_datetime(df[period_col], errors="coerce")
    df = df[ts.notna()].copy()
    df["year"], df["month"] = ts.dt.year, ts.dt.month
    df["certs"] = pd.to_numeric(df[certs_col], errors="coerce").fillna(0.0)

    def tech_of(row) -> str:
        if tech_col and pd.notna(row.get(tech_col)):
            return str(row[tech_col])
        code = row["ID"][6:8].upper() if len(row["ID"]) >= 8 else ""
        return "offshore" if code in {"RP", "SP", "NP"} else "onshore"

    acc = accreditation_year or {}
    monthly = (
        df.groupby(["ID", "year", "month"])
        .agg(certs=("certs", "sum"), tech=(tech_col or "ID",
             "first" if tech_col else (lambda s: tech_of({"ID": s.iloc[0]}))))
        .reset_index()
    )
    monthly["band"] = [
        band_for(r.tech if tech_col else tech_of({"ID": r.ID, tech_col: r.tech}),
                 acc.get(r.ID))
        for r in monthly.itertuples()
    ]
    monthly["mwh"] = monthly["certs"] / monthly["band"]
    return monthly[["ID", "year", "month", "mwh"]]


def pseudo_replicate_observations(
    station_monthly: pd.DataFrame,
    turbines_per_station: pd.Series,
) -> pd.DataFrame:
    """Expand station-monthly MWh to the pseudo-replicated ``ukobs`` format.

    Each station's monthly energy is split EQUALLY across its turbines (the
    committed convention), so every turbine row carries ``station_kWh / n``,
    identical within a station. Output columns match ``ukobs.csv``: ``ID``,
    ``1``..``12`` (kWh), ``year``.

    Args:
        station_monthly: ``ID`` (station), ``year``, ``month``, ``mwh``.
        turbines_per_station: ``station -> turbine count``.

    Returns:
        Wide per-turbine frame in the ``ukobs.csv`` schema.
    """
    mcols = [str(m) for m in range(1, 13)]
    wide = (
        station_monthly.pivot_table(index=["ID", "year"], columns="month",
                                    values="mwh", aggfunc="sum")
        .reindex(columns=range(1, 13))
        .reset_index()
    )
    wide.columns = ["station", "year"] + mcols
    rows = []
    for _, r in wide.iterrows():  # explicit column access, no positional itertuples
        st = r["station"]
        n = int(turbines_per_station.get(st, 1) or 1)
        per_turbine = {m: (0.0 if pd.isna(r[m]) else r[m]) * 1000.0 / n for m in mcols}
        for t in range(1, n + 1):
            rows.append({"ID": f"{st}-{t}", **per_turbine, "year": int(r["year"])})
    return pd.DataFrame(rows, columns=["ID"] + mcols + ["year"])


# ------------------------------------------------------------------ metadata
def repd_wind_metadata(
    repd: pd.DataFrame,
    *,
    height_default: float = 100.0,
    model: str = "2019COE_Market_Average_2.6MW_121",
    operational_only: bool = True,
) -> pd.DataFrame:
    """Per-station open metadata from the Renewable Energy Planning Database.

    Produces the fields REPD actually carries (reprojected lon/lat, installed
    capacity, turbine count, per-turbine capacity, and a partial tip height),
    filling the gaps REPD does NOT carry (turbine model, rotor diameter, most
    hub heights) with uniform defaults, exactly as the AU/BR/CL/AR adapters do
    for metadata-poor regions. Coordinates come from OSGB X/Y reprojected to
    WGS84.

    Args:
        repd: The REPD extract (gov.uk CSV, latin-1).
        height_default: Uniform hub height (m) where REPD gives none.
        model: Uniform power-curve key for every station (no model in REPD).
        operational_only: Keep only ``Operational`` sites.

    Returns:
        One row per REPD wind site: ``station`` (Ref ID), ``site_name``,
        ``type`` (onshore/offshore), ``n_turbines``, ``capacity`` (per-turbine
        kW), ``lon``, ``lat``, ``height``, ``height_source``, ``model``,
        ``model_source``, ``operational``.
    """
    df = repd.copy()
    df = df[df["Technology Type"].astype(str).str.startswith("Wind")].copy()
    if operational_only:
        status = df.get("Development Status (short)", df.get("Development Status"))
        df = df[status.astype(str).str.contains("Operational", case=False, na=False)]

    coords = osgb_to_wgs84(df["X-coordinate"], df["Y-coordinate"])
    df["lon"], df["lat"] = coords["lon"], coords["lat"]
    df["n_turbines"] = pd.to_numeric(df["No. of Turbines"], errors="coerce")
    installed_kw = pd.to_numeric(df["Installed Capacity (MWelec)"], errors="coerce") * 1000.0
    turb_cap = pd.to_numeric(df.get("Turbine Capacity (MW)"), errors="coerce") * 1000.0
    df["capacity"] = turb_cap.where(turb_cap > 0, installed_kw / df["n_turbines"])
    tip = pd.to_numeric(df.get("Height of Turbines (m)"), errors="coerce")
    df["height"] = tip.where(tip > 1, float(height_default))
    df["height_source"] = tip.where(tip > 1).notna().map(
        {True: "repd-tip-height", False: "default-uniform"})
    df["type"] = df["Technology Type"].str.contains("Offshore").map(
        {True: "offshore", False: "onshore"})
    df["model"] = model
    df["model_source"] = "default-uniform"
    df["operational"] = pd.to_datetime(df.get("Operational"), errors="coerce",
                                       dayfirst=True)
    out = df.rename(columns={"Ref ID": "station", "Site Name": "site_name"})
    out = out[out["n_turbines"].notna() & (out["n_turbines"] > 0)
              & out["lon"].notna() & out["capacity"].notna()]
    return out[["station", "site_name", "type", "n_turbines", "capacity",
                "lon", "lat", "height", "height_source", "model",
                "model_source", "operational"]].reset_index(drop=True)


def pseudo_replicate_metadata(station_md: pd.DataFrame) -> pd.DataFrame:
    """Expand per-station open metadata to per-turbine rows (ID ``<st>-<n>``)."""
    rows = []
    for r in station_md.itertuples(index=False):
        for t in range(1, int(r.n_turbines) + 1):
            rows.append({
                "ID": f"{r.station}-{t}", "manufacturer": r.model,
                "capacity": r.capacity, "diameter": float("nan"),
                "height": r.height, "lon": r.lon, "lat": r.lat, "type": r.type,
            })
    return pd.DataFrame(rows)


def divergence_report(open_md: pd.DataFrame, curated_md: pd.DataFrame) -> dict:
    """Summarise how the open (REPD) reconstruction differs from the curated table.

    Both are per-turbine frames with ``ID``, ``capacity``, ``height``, ``lon``,
    ``lat``. Returns coverage and distributional differences, not a row match
    (the keys differ: curated uses ROC accreditation numbers, open uses REPD
    Ref IDs), so this is a fleet-level comparison.
    """
    def stats(df):
        return {
            "turbine_rows": len(df),
            "stations": df["ID"].str.replace(r"-\d+$", "", regex=True).nunique(),
            "total_capacity_gw": round(df["capacity"].sum() / 1e6, 2),
            "median_height_m": round(float(df["height"].median()), 1),
            "has_model": int(df["manufacturer"].notna().sum()) if "manufacturer" in df else 0,
        }
    return {"open": stats(open_md), "curated": stats(curated_md)}
