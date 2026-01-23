from __future__ import annotations

import io
import time
import math
import random
import re
from typing import Iterable, Optional

import pandas as pd
import numpy as np
import requests
import geopandas as gpd

# -----------------------------
# Shared config / schema
# -----------------------------
SCHEMA = ["ID", "capacity", "diameter", "height", "manufacturer", "lon", "lat", "type"]

# Belgium (Flanders) — Mercator OGC API Features
BE_MERCATOR_BASE = "https://www.mercator.vlaanderen.be/raadpleegdienstenmercatorpubliek/ogc/features/v1"
BE_MERCATOR_COLLECTION = "er:er_windturb_omv"

# Belgium offshore — RBINS WFS
BE_RBINS_WFS = "https://spatial.naturalsciences.be/geoserver/od_nature/ows"
BE_RBINS_LAYER = "od_nature:MUMM_windmill_locations_ETRS89"

# Norway — NVE operational plants (blocks)
NO_NVE_IN_OPERATION = "https://api.nve.no/web/WindPowerplant/GetWindPowerPlantsInOperation"

# Norway enrichment (coords + avg hub/rotor) — downloaded as CSV (no datasets/pyarrow required)
NO_REBASE_META_CSV_URL = (
    "https://huggingface.co/datasets/rebase-energy/nve-windpower-data/resolve/main/"
    "nve-windpower-metadata.csv"
)

# OSM Overpass mirrors
OVERPASS_MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
]

# -----------------------------
# Small helpers
# -----------------------------
def _to_num(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return pd.NA
    try:
        return float(x)
    except Exception:
        s = str(x).strip()
        m = re.search(r"[-+]?\d*\.?\d+", s.replace(",", "."))
        return float(m.group(0)) if m else pd.NA


def _ensure_lonlat_from_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(4326, allow_override=True)
    gdf = gdf.to_crs(4326)
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    return gdf


def _dedup_by_distance(df: pd.DataFrame, *, km: float = 0.25, prefer_sources: Optional[list[str]] = None):
    """
    Simple greedy spatial de-dup on lon/lat within 'km' radius.
    If a 'source' column exists and prefer_sources given, higher priority kept.
    """
    if df.empty:
        return df

    R = 6371.0
    lat0 = math.radians(float(df["lat"].dropna().mean())) if df["lat"].notna().any() else 0.0

    def xy(lon, lat):
        return (math.radians(lon) * math.cos(lat0) * R, math.radians(lat) * R)

    # priority ordering
    if prefer_sources and "source" in df.columns:
        pr = {s: i for i, s in enumerate(prefer_sources)}
        df = df.copy()
        df["_prio"] = df["source"].map(lambda s: pr.get(s, 9999))
        df = df.sort_values(["_prio"]).drop(columns=["_prio"])
    else:
        df = df.copy()

    keep = []
    kept_xy = []
    km2 = km * km

    for _, row in df.iterrows():
        if pd.isna(row["lon"]) or pd.isna(row["lat"]):
            keep.append(True)
            continue
        x, y = xy(float(row["lon"]), float(row["lat"]))
        ok = True
        for (kx, ky) in kept_xy:
            dx = kx - x
            dy = ky - y
            if dx * dx + dy * dy <= km2:
                ok = False
                break
        keep.append(ok)
        if ok:
            kept_xy.append((x, y))

    return df.loc[keep].reset_index(drop=True)


def _overpass_post(query: str, *, timeout: int = 240, tries_per_mirror: int = 2):
    last_err = None
    for base in OVERPASS_MIRRORS:
        for attempt in range(tries_per_mirror):
            try:
                r = requests.post(base, data={"data": query}, timeout=timeout)
                if r.status_code == 429:
                    time.sleep(10 + attempt * 10)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                time.sleep(5 + attempt * 5)
    raise last_err


# -----------------------------
# Belgium fetchers
# -----------------------------
def _fetch_be_flanders_mercator(*, timeout: int = 120) -> gpd.GeoDataFrame:
    """
    Mercator OGC API Features collection: er:er_windturb_omv
    """
    limit = 1000
    start = 0
    feats = []
    while True:
        url = f"{BE_MERCATOR_BASE}/collections/{BE_MERCATOR_COLLECTION}/items"
        params = {"f": "application/geo+json", "limit": limit, "startIndex": start}
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        gj = r.json()
        batch = gj.get("features", [])
        feats.extend(batch)
        if len(batch) < limit:
            break
        start += limit

    gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
    return gdf


def _normalise_be_flanders(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    gdf = _ensure_lonlat_from_geometry(gdf)
    # Known columns you showed:
    # fid, ashoogte (m), rotordiameter (m), vermogenmax (kW)
    out = pd.DataFrame({
        "ID": "BE_FLA_" + gdf["fid"].astype(str),
        "capacity": pd.to_numeric(gdf.get("vermogenmax"), errors="coerce") / 1000.0,  # kW -> MW
        "diameter": pd.to_numeric(gdf.get("rotordiameter"), errors="coerce"),
        "height": pd.to_numeric(gdf.get("ashoogte"), errors="coerce"),
        "manufacturer": pd.NA,
        "lon": pd.to_numeric(gdf["lon"], errors="coerce"),
        "lat": pd.to_numeric(gdf["lat"], errors="coerce"),
        "type": "onshore",
    }).reindex(columns=SCHEMA)
    return out


def _fetch_be_offshore_rbines(*, timeout: int = 120) -> gpd.GeoDataFrame:
    url = (
        f"{BE_RBINS_WFS}"
        f"?service=WFS&version=2.0.0&request=GetFeature"
        f"&typeName={BE_RBINS_LAYER}"
        f"&outputFormat=application/json"
        f"&srsName=EPSG:4326"
    )
    return gpd.read_file(url)


def _normalise_be_offshore(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    gdf = _ensure_lonlat_from_geometry(gdf)
    # RBINS layer often has sparse attributes; make IDs from index unless something obvious exists
    id_col = None
    for c in ["id", "ID", "fid", "FID", "objectid", "OBJECTID", "uuid", "name", "Naam", "gml_id"]:
        if c in gdf.columns:
            id_col = c
            break
    if id_col is None:
        ids = ["BE_OFF_" + str(i) for i in range(len(gdf))]
    else:
        ids = gdf[id_col].astype(str).tolist()

    out = pd.DataFrame({
        "ID": ids,
        "capacity": pd.NA,
        "diameter": pd.NA,
        "height": pd.NA,
        "manufacturer": pd.NA,
        "lon": pd.to_numeric(gdf["lon"], errors="coerce"),
        "lat": pd.to_numeric(gdf["lat"], errors="coerce"),
        "type": "offshore",
    }).reindex(columns=SCHEMA)
    return out


def _fetch_be_onshore_osm(*, timeout: int = 240) -> gpd.GeoDataFrame:
    template = """
    [out:json][timeout:180];
    area["ISO3166-1"="BE"][admin_level=2]->.be;
    (
      {ELEMENT}(area.be)["power"="generator"]["generator:source"="wind"];
    );
    out center tags;
    """
    elements = []
    for element in ["node", "way", "relation"]:
        data = _overpass_post(template.replace("{ELEMENT}", element), timeout=timeout)
        elements.extend(data.get("elements", []))

    records = []
    for el in elements:
        tags = el.get("tags", {}) or {}
        if el["type"] == "node":
            lon, lat = el.get("lon"), el.get("lat")
        else:
            center = el.get("center") or {}
            lon, lat = center.get("lon"), center.get("lat")
        if lon is None or lat is None:
            continue
        rec = {f"tag_{k}": v for k, v in tags.items()}
        rec.update({"osm_type": el["type"], "osm_id": el["id"], "lon": lon, "lat": lat})
        records.append(rec)

    df = pd.DataFrame(records)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    return gdf


def _normalise_be_osm(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    def parse_capacity_mw(x):
        if pd.isna(x):
            return pd.NA
        s = str(x).strip()
        m = re.search(r"([-+]?\d*\.?\d+)", s.replace(",", "."))
        if not m:
            return pd.NA
        val = float(m.group(1))
        if re.search(r"\bMW\b", s, re.I):
            return val
        if re.search(r"\bkW\b", s, re.I):
            return val / 1000.0
        if re.search(r"\bW\b", s) and not re.search(r"\bkW\b|\bMW\b", s, re.I):
            return val / 1e6
        # unknown unit; assume MW-like
        return val

    def parse_m(x):
        if pd.isna(x):
            return pd.NA
        s = str(x).strip()
        m = re.search(r"([-+]?\d*\.?\d+)", s.replace(",", "."))
        return float(m.group(1)) if m else pd.NA

    gdf = gdf.copy()
    out = pd.DataFrame(index=gdf.index)
    out["ID"] = "OSM_" + gdf["osm_type"].astype(str) + "_" + gdf["osm_id"].astype(str)

    cap = gdf.get("tag_generator:output:electricity", pd.Series([pd.NA] * len(gdf)))
    out["capacity"] = cap.map(parse_capacity_mw)

    dia = gdf.get("tag_rotor:diameter", gdf.get("tag_rotor_diameter", pd.Series([pd.NA] * len(gdf))))
    out["diameter"] = dia.map(parse_m)

    hub = gdf.get("tag_hub_height", gdf.get("tag_height", pd.Series([pd.NA] * len(gdf))))
    out["height"] = hub.map(parse_m)

    manu = gdf.get("tag_manufacturer", gdf.get("tag_brand", pd.Series([pd.NA] * len(gdf))))
    out["manufacturer"] = manu.astype(str)
    out.loc[out["manufacturer"].isin(["nan", "None", ""]), "manufacturer"] = pd.NA

    out["lon"] = pd.to_numeric(gdf["lon"], errors="coerce")
    out["lat"] = pd.to_numeric(gdf["lat"], errors="coerce")
    out["type"] = "onshore"

    out = out.reindex(columns=SCHEMA)
    return out


# -----------------------------
# Norway fetchers / normalisers
# -----------------------------
def _fetch_no_nve_operational(*, timeout: int = 60, retries: int = 4) -> list[dict]:
    last = None
    for i in range(retries):
        try:
            r = requests.get(NO_NVE_IN_OPERATION, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                raise ValueError(f"Unexpected JSON root type: {type(data)}")
            return data
        except Exception as e:
            last = e
            time.sleep(2 + i * 2)
    raise last


def _fetch_no_enrichment_csv(*, url: str = NO_REBASE_META_CSV_URL, timeout: int = 120) -> pd.DataFrame:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def _build_no_enrichment(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Expect (common) columns:
      WindPowerPlantId, lat, lon, AvgHubHeight, AvgRotorDiameter
    If upstream changes, we keep best-effort.
    """
    # best-effort column picks
    def pick(cols: Iterable[str]) -> Optional[str]:
        for c in cols:
            if c in meta.columns:
                return c
        return None

    idc = pick(["WindPowerPlantId", "windpowerplantid", "VindkraftAnleggId", "VindkraftanleggId"])
    latc = pick(["lat", "Latitude", "latitude"])
    lonc = pick(["lon", "Longitude", "longitude"])
    hubc = pick(["AvgHubHeight", "avg_hub_height", "HubHeight", "GjsnittNavhoeyde", "GjsnittNavhoyde"])
    rotc = pick(["AvgRotorDiameter", "avg_rotor_diameter", "RotorDiameter", "GjsnittRotordiameter"])

    if idc is None:
        raise ValueError(f"Could not find plant-id column in Norway enrichment CSV. Columns: {list(meta.columns)}")

    out = pd.DataFrame({
        "VindkraftAnleggId": pd.to_numeric(meta[idc], errors="coerce").astype("Int64"),
        "lat": pd.to_numeric(meta[latc], errors="coerce") if latc else pd.NA,
        "lon": pd.to_numeric(meta[lonc], errors="coerce") if lonc else pd.NA,
        "avg_hub_height": pd.to_numeric(meta[hubc], errors="coerce") if hubc else pd.NA,
        "avg_rotor_diameter": pd.to_numeric(meta[rotc], errors="coerce") if rotc else pd.NA,
    })
    return out.dropna(subset=["VindkraftAnleggId"]).drop_duplicates(subset=["VindkraftAnleggId"])


def _normalise_no_pseudoturbines(
    nve_data: list[dict],
    enrich: pd.DataFrame,
    *,
    jitter_km: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Expand turbine blocks -> pseudo-turbines at plant centroid.
    capacity = per-turbine MW from TurbinStorrelse_kW / 1000 (if available)
    diameter/height/lon/lat from enrichment (plant-level averages & coords)
    manufacturer from block TurbinProdusent
    Optionally add small spatial jitter to pseudo-turbines (for plotting / spatial joins).
    """
    rng = random.Random(seed)
    idx = enrich.set_index("VindkraftAnleggId")

    rows = []
    for plant in nve_data:
        pid = plant.get("VindkraftAnleggId")
        if pid is None:
            continue
        try:
            pid = int(pid)
        except Exception:
            continue

        if pid in idx.index:
            lon = idx.loc[pid, "lon"]
            lat = idx.loc[pid, "lat"]
            diam = idx.loc[pid, "avg_rotor_diameter"]
            hub = idx.loc[pid, "avg_hub_height"]
        else:
            lon = lat = diam = hub = pd.NA

        blocks = plant.get("Turbiner") or []
        if not blocks:
            rows.append({
                "ID": f"NO_{pid}_plant",
                "capacity": _to_num(plant.get("InstallertEffekt_MW")),
                "diameter": _to_num(diam),
                "height": _to_num(hub),
                "manufacturer": pd.NA,
                "lon": _to_num(lon),
                "lat": _to_num(lat),
                "type": "onshore",
            })
            continue

        for b_i, blk in enumerate(blocks):
            n = blk.get("AntallTurbiner")
            try:
                n = int(n) if n is not None else 0
            except Exception:
                n = 0

            size_kw = blk.get("TurbinStorrelse_kW")
            cap_mw = pd.to_numeric(size_kw, errors="coerce") / 1000.0 if size_kw is not None else pd.NA
            manu = blk.get("TurbinProdusent")
            manu = manu if manu not in (None, "") else pd.NA

            if n <= 0:
                n = 1  # at least one pseudo-turbine for the block

            # jitter: uniform in circle radius jitter_km
            for t_i in range(n):
                jl = _to_num(lon)
                jt = _to_num(lat)
                if jitter_km and (jl is not pd.NA) and (jt is not pd.NA):
                    # crude jitter in degrees (OK for small distances)
                    # 1 deg lat ~ 111 km; lon scales by cos(lat)
                    r = jitter_km * math.sqrt(rng.random())
                    ang = 2 * math.pi * rng.random()
                    dlat = (r * math.sin(ang)) / 111.0
                    dlon = (r * math.cos(ang)) / (111.0 * max(0.2, math.cos(math.radians(float(jt)))))
                    jl = float(jl) + dlon
                    jt = float(jt) + dlat

                rows.append({
                    "ID": f"NO_{pid}_blk_{b_i}_t_{t_i}",
                    "capacity": cap_mw,
                    "diameter": _to_num(diam),
                    "height": _to_num(hub),
                    "manufacturer": manu,
                    "lon": jl,
                    "lat": jt,
                    "type": "onshore",
                })

    df = pd.DataFrame(rows).reindex(columns=SCHEMA)
    for c in ["capacity", "diameter", "height", "lon", "lat"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# -----------------------------
# PUBLIC FUNCTIONS YOU ASKED FOR
# -----------------------------
def create_turbine_metadata_be(
    *,
    include_osm_wallonia_brussels: bool = True,
    dedup_km: float = 0.25,
    timeout: int = 120,
) -> pd.DataFrame:
    """
    Build Belgium turbine metadata (onshore + offshore) into one DataFrame with SCHEMA.
    Sources:
      - Flanders (onshore): Mercator OGC API (rich specs)
      - Offshore: RBINS WFS points
      - Wallonia+Brussels (onshore, best-effort): OSM Overpass (mirrors + retries)

    Returns: DataFrame with columns SCHEMA.
    """
    parts = []

    # Offshore (RBINS)
    be_off_gdf = _fetch_be_offshore_rbines(timeout=timeout)
    be_off = _normalise_be_offshore(be_off_gdf)
    be_off["source"] = "RBINS"
    parts.append(be_off)

    # Flanders onshore (Mercator)
    be_fla_gdf = _fetch_be_flanders_mercator(timeout=timeout)
    be_fla = _normalise_be_flanders(be_fla_gdf)
    be_fla["source"] = "MERCATOR"
    parts.append(be_fla)

    # Wallonia+Brussels onshore (OSM)
    if include_osm_wallonia_brussels:
        be_osm_gdf = _fetch_be_onshore_osm(timeout=max(timeout, 180))
        be_osm = _normalise_be_osm(be_osm_gdf)
        be_osm["source"] = "OSM"
        parts.append(be_osm)

    be = pd.concat(parts, ignore_index=True, sort=False)

    # Prefer authoritative sources if duplicates
    be = _dedup_by_distance(be, km=dedup_km, prefer_sources=["RBINS", "MERCATOR", "OSM"])

    # Ensure schema only
    be = be.reindex(columns=SCHEMA)
    return be


def create_turbine_metadata_no(
    *,
    jitter_km: float = 0.0,
    seed: int = 0,
    timeout: int = 120,
) -> pd.DataFrame:
    """
    Build Norway pseudo-turbine metadata (onshore) into SCHEMA.
    Sources:
      - NVE operational API: turbine blocks (count, size, manufacturer)
      - Enrichment CSV: plant lon/lat + avg hub height + avg rotor diameter (best-effort)

    Parameters:
      jitter_km: if >0, spatially jitter pseudo-turbines around plant centroid (useful for maps).
    """
    nve = _fetch_no_nve_operational(timeout=min(timeout, 120))
    meta_raw = _fetch_no_enrichment_csv(timeout=timeout)
    enrich = _build_no_enrichment(meta_raw)

    no = _normalise_no_pseudoturbines(nve, enrich, jitter_km=jitter_km, seed=seed)
    return no

def clean_and_impute_turbine_metadata(
    df: pd.DataFrame,
    *,
    country: str,
    drop_if_no_capacity: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Cleans turbine metadata by:
    - imputing missing capacity / diameter / height
    - converting capacity from MW -> kW (FINAL OUTPUT)
    - optionally dropping rows still missing capacity

    Assumes INPUT capacity is in MW.
    Works for BE and NO outputs.
    """

    df = df.copy()

    # -------------------------------------------------
    # 0) Enforce numeric + clarify units (MW internally)
    # -------------------------------------------------
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")  # MW
    df["diameter"] = pd.to_numeric(df.get("diameter"), errors="coerce")
    df["height"] = pd.to_numeric(df.get("height"), errors="coerce")

    # -----------------------
    # 1) Capacity imputation (MW)
    # -----------------------
    if df["capacity"].isna().any():
        medians = (
            df.groupby("type")["capacity"]
            .median()
            .dropna()
        )

        def fill_capacity(row):
            if not pd.isna(row["capacity"]):
                return row["capacity"]
            return medians.get(row["type"], np.nan)

        df["capacity"] = df.apply(fill_capacity, axis=1)

    # -----------------------
    # 2) Diameter imputation
    # -----------------------
    # Empirical scaling: D ≈ 35 * sqrt(P_MW)
    mask_d = df["diameter"].isna() & df["capacity"].notna()
    df.loc[mask_d, "diameter"] = (
        35.0 * np.sqrt(df.loc[mask_d, "capacity"])
    )

    # Clamp to realistic bounds
    df["diameter"] = df["diameter"].clip(lower=40, upper=260)

    # -----------------------
    # 3) Hub height imputation
    # -----------------------
    mask_h = df["height"].isna() & df["diameter"].notna()

    df.loc[mask_h & (df["type"] == "onshore"), "height"] = (
        1.1 * df.loc[mask_h & (df["type"] == "onshore"), "diameter"]
    )

    df.loc[mask_h & (df["type"] == "offshore"), "height"] = (
        0.9 * df.loc[mask_h & (df["type"] == "offshore"), "diameter"]
    )

    # -----------------------
    # 4) Final drop (MW stage)
    # -----------------------
    if drop_if_no_capacity:
        before = len(df)
        df = df.dropna(subset=["capacity"])
        after = len(df)
        if verbose and before != after:
            print(f"[{country}] Dropped {before-after} rows with no capacity")

    # -----------------------
    # 5) Convert capacity MW → kW (FINAL OUTPUT)
    # -----------------------
    df["capacity"] = df["capacity"] * 1000.0  # MW → kW

    # -----------------------
    # 6) Sanity report
    # -----------------------
    if verbose:
        print(f"[{country}] Final turbine metadata (capacity in kW):")
        print(" rows:", len(df))
        print(" missing capacity:", int(df["capacity"].isna().sum()))
        print(" missing diameter:", int(df["diameter"].isna().sum()))
        print(" missing height:", int(df["height"].isna().sum()))
        print(
            " capacity range [MW]:",
            (df["capacity"] / 1000).min(),
            "–",
            (df["capacity"] / 1000).max(),
        )

    return df
# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    be = create_turbine_metadata_be(include_osm_wallonia_brussels=True, dedup_km=0.25)
    be = clean_and_impute_turbine_metadata(be, country="BE")
    be.to_csv("be_md.csv", index=False)
    print("BE saved:", be.shape)

    no = create_turbine_metadata_no(jitter_km=0.0, seed=0)
    no = clean_and_impute_turbine_metadata(no, country="NO")
    no.to_csv("no_md.csv", index=False)
    print("NO saved:", no.shape)