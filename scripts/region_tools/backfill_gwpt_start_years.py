"""Backfill start years for tracker records that carry none, from national registers.

The per-year country grids (``weight_country_grid_points.py --per-year``) place
each Global Wind Power Tracker project in time by its start year, and keep a
project without one in every year. In the February 2026 release that is 3,621
MW of France's operating onshore fleet and 296 MW of Norway's (after the curated
exclusions), and the records are not old sites: every undated Norwegian
project came online in 2018-21, and more than half of the French capacity this
script dates was built in 2015-21. Counted from the first year of a training
window, they flatten exactly the growth a per-year grid exists to show.

Two national registers date them:

- **Norway: NVE's register of wind plants in operation** (``api.nve.no``,
  NLOD licence), one record per plant with its turbines and their dates. A
  project is matched by name, uniquely, and dated by the capacity-weighted
  median commissioning year of the plant's turbines in service, so a
  repowering phase takes its new turbines' year rather than the site's first.
- **France: the ODRE national register of production installations**
  (``odre.opendatasoft.com``, Licence Ouverte), filtered to wind. Most names
  are confidential, so a project is matched by commune and capacity, with
  commune centres from ``geo.api.gouv.fr``. Three rules, each accepting only
  capacity within 5% and a single commissioning year: records at the project's
  commune or named after it; the one record within 5 km; the records of the
  nearest commune within 5 km, summed. Where more than one rule fires, they
  must agree to within a year, and the earlier year is taken.

Each rule is run on the records the tracker does date, and its agreement with
the tracker's year is printed before anything is written. That is the only
evidence of its accuracy. On the February 2026 release the French rules
together agree to within a year on 92% of dated projects and miss by more
than two on 7%, the misses being mostly repowerings (the register gives the
new date) and dense clusters. A looser rule, every record within 2 km sharing
a year, reached 44% of the undated capacity instead of 29% but was less
accurate on dated projects, and is not used.

Nothing here replaces a year the tracker gives. The output,
``configs/curation/gwpt_start_years.csv``, names the register records each
year rests on and the sha256 of the register file.

The registers are local and git-ignored, under
``<input-root>/reference/registers/``: ``nve_wind_in_operation_<date>.json``,
``odre_registre_eolien_<date>.csv`` and ``fr_communes_centre_<date>.json``.
See ``docs/guides/data-sources.md``.

Usage:
    PYTHONPATH=src python scripts/region_tools/backfill_gwpt_start_years.py --dry-run
    PYTHONPATH=src python scripts/region_tools/backfill_gwpt_start_years.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyvwf.config import PyVWFPaths  # noqa: E402
from pyvwf.datasets.gwpt import COUNTRY_NAME, load_exclusions, load_gwpt  # noqa: E402
from pyvwf.datasets.gwpt import default_path as gwpt_default_path  # noqa: E402

REGISTERS = PyVWFPaths.INPUT_ROOT / "reference" / "registers"
NVE = REGISTERS / "nve_wind_in_operation_2026-09-25.json"
ODRE = REGISTERS / "odre_registre_eolien_2026-09-25.csv"
COMMUNES = REGISTERS / "fr_communes_centre_2026-09-25.json"
EXCLUSIONS_PATH = REPO_ROOT / "configs" / "curation" / "gwpt_exclusions.csv"
OUT_PATH = REPO_ROOT / "configs" / "curation" / "gwpt_start_years.csv"

#: Capacity agreement a match needs, as a share of the tracker's capacity.
CAPACITY_TOL = 0.05
#: NVE plants hold every phase of a site, including a repowered one's older
#: turbines, so the Norwegian capacity check is looser.
NVE_CAPACITY_TOL = 0.10
RADIUS_KM = 5.0

COLUMNS = [
    "gem_phase_id",
    "country",
    "project_name",
    "phase_name",
    "capacity_mw",
    "start_year",
    "rule",
    "register",
    "register_ids",
]


def norm(text: object) -> str:
    """Upper-case ASCII words, for matching names and communes across sources."""
    if not isinstance(text, str):
        return ""
    for a, b in (("æ", "ae"), ("Æ", "AE"), ("ø", "o"), ("Ø", "O")):
        text = text.replace(a, b)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().upper()
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    text = re.sub(r"\bSAINTE\b", "STE", re.sub(r"\bSAINT\b", "ST", text))
    return " ".join(text.split())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; broadcasts."""
    p = np.deg2rad
    a = (
        np.sin(p(lat2 - lat1) / 2) ** 2
        + np.cos(p(lat1)) * np.cos(p(lat2)) * np.sin(p(lon2 - lon1) / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def tracker_rows(gwpt: pd.DataFrame, code: str, excluded: set[str]) -> pd.DataFrame:
    """Operating records with coordinates, less the curated exclusions."""
    rows = gwpt[gwpt["Country/Area"] == COUNTRY_NAME[code]]
    rows = rows[rows["Status"].astype(str).str.lower() == "operating"]
    rows = rows.dropna(subset=["Latitude", "Longitude", "Capacity (MW)"])
    rows = rows[~rows["GEM phase ID"].astype(str).isin(excluded)].copy()
    rows["tracker_year"] = pd.to_numeric(rows["Start year"], errors="coerce")
    return rows.reset_index(drop=True)


def agreement(found: pd.Series, tracker: pd.Series) -> str:
    """How a rule's years compare with the tracker's, on records it dates."""
    both = found.notna() & tracker.notna()
    d = found[both] - tracker[both]
    if not len(d):
        return "n=0"
    return (
        f"n={len(d)}, exact {(d == 0).mean():.0%}, within 1 y {(d.abs() <= 1).mean():.0%}, "
        f"more than 2 y off {(d.abs() > 2).mean():.0%}"
    )


# --------------------------------------------------------------------------
# Norway
# --------------------------------------------------------------------------


def nve_start_year(turbines: list[dict]) -> int | None:
    """Capacity-weighted median commissioning year of a plant's turbines in service."""
    rows = [
        (pd.Timestamp(t["DatoIdriftsatt"]).year, t["AntallTurbiner"] * t["TurbinStorrelse_kW"])
        for t in turbines
        if t.get("DatoUtavdrift") is None and t.get("DatoIdriftsatt")
    ]
    if not rows:
        return None
    years = sorted(rows)
    total = sum(kw for _, kw in years)
    running = 0.0
    for year, kw in years:
        running += kw
        if running >= total / 2:
            return int(year)
    return int(years[-1][0])


def date_norway(rows: pd.DataFrame, nve: pd.DataFrame) -> pd.DataFrame:
    """One NVE plant per tracker record, by name, at 10% of the plant's in-service capacity."""
    nve = nve.assign(key=nve["Navn"].map(norm))
    out = []
    for _, r in rows.iterrows():
        key = norm(str(r["Project Name"]).replace(" wind farm", ""))
        hit = nve[(nve["key"] == key) | nve["key"].str.startswith(key + " ")]
        year = plant = None
        if len(hit) == 1:
            p = hit.iloc[0]
            in_service_mw = (
                sum(
                    t["AntallTurbiner"] * t["TurbinStorrelse_kW"]
                    for t in p["Turbiner"]
                    if t.get("DatoUtavdrift") is None
                )
                / 1000
            )
            if abs(in_service_mw - r["Capacity (MW)"]) <= NVE_CAPACITY_TOL * r["Capacity (MW)"]:
                year, plant = nve_start_year(p["Turbiner"]), int(p["VindkraftAnleggId"])
        out.append({"year": year, "ids": f"nve:{plant}" if plant is not None else ""})
    return pd.DataFrame(out, index=rows.index)


# --------------------------------------------------------------------------
# France
# --------------------------------------------------------------------------


def load_odre(path: Path, communes: Path) -> pd.DataFrame:
    """Onshore wind records with a date, a capacity and, where known, a commune centre."""
    o = pd.read_csv(path, sep=";", dtype={"codeinseecommune": str})
    o["mw"] = o["puismaxinstallee"] / 1000
    o["year"] = pd.to_datetime(o["datemiseenservice"], format="%d/%m/%Y", errors="coerce").dt.year
    o = o[(o["codetechnologie"] == "TERRE") & o["year"].notna() & (o["mw"] > 0)]
    centres = pd.DataFrame(
        [(c["code"], *c["centre"]["coordinates"]) for c in json.loads(communes.read_text())],
        columns=["code", "lon", "lat"],
    ).set_index("code")
    o = o.join(centres, on="codeinseecommune").reset_index(drop=True)
    o["commune_n"] = o["commune"].map(norm)
    o["name_n"] = o["nominstallation"].map(norm)
    o["rid"] = [
        f"eic:{e}" if isinstance(e, str) and e else f"insee:{c}/{d}/{kw:g}kW"
        for e, c, d, kw in zip(
            o["codeeicresourceobject"],
            o["codeinseecommune"],
            o["datemiseenservice"],
            o["puismaxinstallee"],
        )
    ]
    return o


def one_year(recs: pd.DataFrame) -> int | None:
    years = recs["year"].unique()
    return int(years[0]) if len(years) == 1 else None


def date_france(rows: pd.DataFrame, o: pd.DataFrame) -> pd.DataFrame:
    """The three French rules per tracker record, and their combination."""
    dist = km(
        rows["Latitude"].to_numpy()[:, None],
        rows["Longitude"].to_numpy()[:, None],
        o["lat"].to_numpy()[None, :],
        o["lon"].to_numpy()[None, :],
    )
    dist = np.where(np.isnan(dist), np.inf, dist)
    mw = rows["Capacity (MW)"].to_numpy()
    close_cap = np.abs(o["mw"].to_numpy()[None, :] - mw[:, None]) <= CAPACITY_TOL * mw[:, None]

    def fits(total: float, i: int) -> bool:
        return abs(total - mw[i]) <= CAPACITY_TOL * mw[i]

    out = []
    for i, r in rows.iterrows():
        found: dict[str, tuple[int, list[str]]] = {}

        name = norm(str(r["Project Name"]).replace(" wind farm", ""))
        keys = {k for k in (name, norm(r.get("City"))) if k}
        cand = o[
            o["commune_n"].isin(keys)
            | o["name_n"].map(lambda n, name=name: len(name) > 3 and name in n)
        ]
        close = cand[(cand["mw"] - mw[i]).abs() <= CAPACITY_TOL * mw[i]]
        if len(close) == 1 or (len(close) > 1 and one_year(close) is not None):
            found["name"] = (int(close["year"].iloc[0]), close["rid"].tolist())
        elif len(cand) and fits(cand["mw"].sum(), i) and one_year(cand) is not None:
            found["name"] = (one_year(cand), cand["rid"].tolist())  # type: ignore[assignment]

        hit = np.flatnonzero((dist[i] <= RADIUS_KM) & close_cap[i])
        if len(hit) == 1:
            found["geo"] = (int(o.at[hit[0], "year"]), [o.at[hit[0], "rid"]])

        near = np.flatnonzero(dist[i] <= RADIUS_KM)
        if len(near):
            commune = o.at[near[np.argmin(dist[i, near])], "codeinseecommune"]
            recs = o[o["codeinseecommune"] == commune]
            if fits(recs["mw"].sum(), i) and one_year(recs) is not None:
                found["commune_sum"] = (one_year(recs), recs["rid"].tolist())  # type: ignore[assignment]

        years = [y for y, _ in found.values()]
        ok = bool(years) and max(years) - min(years) <= 1
        out.append(
            {
                **{k: found.get(k, (None, []))[0] for k in ("name", "geo", "commune_sum")},
                "year": min(years) if ok else None,
                "rule": "+".join(found) if ok else "",
                "ids": "|".join(sorted({x for _, ids in found.values() for x in ids}))
                if ok
                else "",
            }
        )
    return pd.DataFrame(out, index=rows.index)


# --------------------------------------------------------------------------


def main(out: Path = OUT_PATH, dry_run: bool = False) -> pd.DataFrame:
    gwpt = load_gwpt(gwpt_default_path())
    excluded = load_exclusions(EXCLUSIONS_PATH)
    tables = []

    no = tracker_rows(gwpt, "NO", excluded)
    nve = pd.DataFrame(json.loads(NVE.read_text()))
    got = date_norway(no, nve)
    print(
        f"NO, NVE rule on records the tracker dates: {agreement(got['year'], no['tracker_year'])}"
    )
    undated = no["tracker_year"].isna()
    print(
        f"NO undated: {int(undated.sum())} records, {no.loc[undated, 'Capacity (MW)'].sum():.0f} MW; "
        f"dated here {int((undated & got['year'].notna()).sum())}"
    )
    tables.append((no[undated], got[undated], "name", f"{NVE.name} sha256:{sha256(NVE)[:12]}"))

    fr = tracker_rows(gwpt, "FR", excluded)
    fr = fr[fr["Installation Type"] == "Onshore"].reset_index(drop=True)
    odre = load_odre(ODRE, COMMUNES)
    got = date_france(fr, odre)
    for rule in ("name", "geo", "commune_sum", "year"):
        label = "combined" if rule == "year" else rule
        print(
            f"FR, {label} on records the tracker dates: {agreement(got[rule], fr['tracker_year'])}"
        )
    undated = fr["tracker_year"].isna()
    dated_here = undated & got["year"].notna()
    print(
        f"FR undated: {int(undated.sum())} records, {fr.loc[undated, 'Capacity (MW)'].sum():.0f} MW; "
        f"dated here {int(dated_here.sum())}, {fr.loc[dated_here, 'Capacity (MW)'].sum():.0f} MW"
    )
    register = (
        f"{ODRE.name} sha256:{sha256(ODRE)[:12]}; {COMMUNES.name} sha256:{sha256(COMMUNES)[:12]}"
    )
    tables.append((fr[undated], got[undated], None, register))

    rows = []
    for tracker, found, rule, register in tables:
        for i in tracker.index[found.loc[tracker.index, "year"].notna()]:
            r = tracker.loc[i]
            rows.append(
                {
                    "gem_phase_id": r["GEM phase ID"],
                    "country": next(c for c, n in COUNTRY_NAME.items() if n == r["Country/Area"]),
                    "project_name": r["Project Name"],
                    "phase_name": r["Phase Name"],
                    "capacity_mw": float(r["Capacity (MW)"]),
                    "start_year": int(found.at[i, "year"]),
                    "rule": rule or found.at[i, "rule"],
                    "register": register,
                    "register_ids": found.at[i, "ids"],
                }
            )
    table = pd.DataFrame(rows, columns=COLUMNS).sort_values(["country", "gem_phase_id"])
    by_year = table.groupby(["country", "start_year"])["capacity_mw"].sum().round(0)
    print("\nDated capacity by country and start year (MW):")
    print(by_year.to_string())
    if dry_run:
        print("\ndry run, nothing written")
    else:
        table.to_csv(out, index=False)
        print(f"\nwrote {len(table)} rows to {out.relative_to(REPO_ROOT)}")
    return table


def cli(argv: list[str] | None = None) -> None:
    """Parse ``[--dry-run] [--out PATH]`` and run :func:`main`."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="report only")
    parser.add_argument("--out", type=Path, default=OUT_PATH, help="the curation table to write")
    args = parser.parse_args(argv)
    main(out=args.out, dry_run=args.dry_run)


if __name__ == "__main__":
    cli()
