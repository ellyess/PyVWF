"""Match the German turbine register to MaStR, for per-turbine coordinates.

The DE fleet the caches carry sits at postcode centroids: 579 distinct points
behind 4,288 turbines in the fitted fleet, 1,161 behind the 10,889 the
register holds. The Marktstammdatenregister (MaStR) gives a coordinate and a
hub height per unit. This matches one to the other so the caches can be
rebuilt on real locations, as
`docs/findings/method-physics-informed-turbine-prereg.md` anticipated under
"Germany's locations".

**The two sides do not share a key.** The register's ID carries the postcode
(`01454 A01`) and the file gives manufacturer, rating, rotor diameter, tower
height and a commissioning date; MaStR gives postcode, rating, commissioning
date, rotor diameter, hub height, a manufacturer code and a type designation.
The registration names postcode, capacity, commissioning date and model. The
register has **no model designation** (`scorecard.md` records this), so rotor
diameter and tower height stand in for it, and the match is scored on all
four of postcode, capacity, date and rotor, with hub height as a tie-break.

A match is accepted only inside one postcode, and one MaStR unit is used at
most once: the assignment is greedy over the best-scoring pairs, so a
register row takes a MaStR unit only if no better pairing for that unit
exists. Rows that fail any tolerance stay unmatched and keep their postcode
centroid.

    PYTHONPATH=src python scripts/studies/physics-informed/mastr_match.py

Writes `mastr_units.csv` (the parsed register), `de_match.csv` (one row per
register turbine, matched or not), `de_fleet_override.csv` (the matched units
as `ID`, `lon`, `lat`, `height`, which `build_cache.py --fleet-override`
reads) and `de_match_summary.json` under `--out`.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pyvwf.cli.common import make_parser  # noqa: E402
from pyvwf.config import PyVWFPaths  # noqa: E402

ARCHIVE = Path("input/raw/mastr/Gesamtdatenexport_20260918_26.1.zip")
MEMBER = "EinheitenWind.xml"
OUT = Path("output/pinn_turbine_2026-09-18/mastr")

#: Fields taken from each `EinheitWind` record.
FIELDS = (
    "EinheitMastrNummer",
    "Postleitzahl",
    "Laengengrad",
    "Breitengrad",
    "Inbetriebnahmedatum",
    "Bruttoleistung",
    "Nabenhoehe",
    "Rotordurchmesser",
    "Typenbezeichnung",
    "Hersteller",
    "WindAnLandOderAufSee",
    "EinheitBetriebsstatus",
    "NameWindpark",
)

#: Tolerances a pair must clear, all four together.
CAPACITY_TOLERANCE = 0.02  # relative, on kW
ROTOR_TOLERANCE_M = 2.0
DATE_TOLERANCE_DAYS = 400
#: Weights turning the four distances into one score, so the assignment is
#: deterministic. Capacity and rotor identify the machine; the date separates
#: two identical machines commissioned years apart in one postcode.
SCORE_WEIGHTS = {"capacity": 1.0, "rotor": 0.5, "days": 0.002, "height": 0.05}


def parse_units(archive: Path, member: str = MEMBER) -> pd.DataFrame:
    """Every wind unit in the MaStR export, as a frame of ``FIELDS``.

    The export is UTF-16 XML of about 200 MB, so it is streamed from inside
    the zip and each record is cleared as it closes.
    """
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(archive) as zf, zf.open(member) as stream:
        for event, element in ElementTree.iterparse(stream, events=("end",)):
            if element.tag != "EinheitWind":
                continue
            row = {}
            for child in element:
                if child.tag in FIELDS:
                    row[child.tag] = child.text
            rows.append(row)
            element.clear()
    units = pd.DataFrame(rows)
    for column in (
        "Bruttoleistung",
        "Nabenhoehe",
        "Rotordurchmesser",
        "Laengengrad",
        "Breitengrad",
    ):
        units[column] = pd.to_numeric(units.get(column), errors="coerce")
    units["postcode"] = pd.to_numeric(units.get("Postleitzahl"), errors="coerce")
    units["commissioned"] = pd.to_datetime(units.get("Inbetriebnahmedatum"), errors="coerce")
    return units


def register_fleet() -> pd.DataFrame:
    """The German register as the loader reads it, plus the columns it drops.

    ``load_turbine_metadata`` returns no commissioning date, and the match
    needs one, so the same file is read here: ID, manufacturer, rating, rotor,
    tower and `StartDate`.
    """
    path = PyVWFPaths.TURBINE_DATA / "DE" / "DE_md.csv"
    frame = pd.read_csv(
        path, usecols=["V1", "Manufacturer", "kW", "Rotor..m.", "Tower..m.", "StartDate"]
    )
    frame.columns = ["ID", "manufacturer", "capacity", "diameter", "height", "start"]
    frame["postcode"] = pd.to_numeric(frame["ID"].astype(str).str[:5], errors="coerce")
    frame["commissioned"] = pd.to_datetime(frame["start"], format="%d-%m-%Y", errors="coerce")
    return frame


def _pair_score(row: pd.Series, candidates: pd.DataFrame) -> pd.Series:
    """The weighted distance from one register turbine to each candidate."""
    capacity = (candidates["Bruttoleistung"] - row["capacity"]).abs() / max(row["capacity"], 1.0)
    rotor = (candidates["Rotordurchmesser"] - row["diameter"]).abs()
    days = (candidates["commissioned"] - row["commissioned"]).dt.days.abs()
    height = (candidates["Nabenhoehe"] - row["height"]).abs()
    within = (
        (capacity <= CAPACITY_TOLERANCE)
        & (rotor.fillna(np.inf) <= ROTOR_TOLERANCE_M)
        & (days.fillna(np.inf) <= DATE_TOLERANCE_DAYS)
    )
    score = (
        SCORE_WEIGHTS["capacity"] * capacity
        + SCORE_WEIGHTS["rotor"] * rotor.fillna(ROTOR_TOLERANCE_M)
        + SCORE_WEIGHTS["days"] * days.fillna(DATE_TOLERANCE_DAYS)
        + SCORE_WEIGHTS["height"] * height.fillna(0.0)
    )
    return score.where(within)


def match(fleet: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """One row per register turbine, with its MaStR unit where one matched.

    Greedy over the best pair in each postcode, so a MaStR unit goes to the
    register row that fits it best and is then unavailable.
    """
    with_coords = units.dropna(subset=["Laengengrad", "Breitengrad", "postcode"])
    by_postcode = {code: group for code, group in with_coords.groupby("postcode")}

    results = []
    for postcode, rows in fleet.groupby("postcode"):
        candidates = by_postcode.get(postcode)
        if candidates is None or candidates.empty:
            for _, row in rows.iterrows():
                results.append(
                    {**row.to_dict(), "matched": False, "reason": "no MaStR unit in postcode"}
                )
            continue
        scores = {}
        for index, row in rows.iterrows():
            scores[index] = _pair_score(row, candidates)
        taken: set = set()
        pairs: list[tuple[float, int, str]] = []
        for index, series in scores.items():
            for unit_id, value in series.dropna().items():
                pairs.append((float(value), index, unit_id))
        pairs.sort()
        assigned: dict[int, str] = {}
        for _value, index, unit_id in pairs:
            if index in assigned or unit_id in taken:
                continue
            assigned[index] = unit_id
            taken.add(unit_id)
        for index, row in rows.iterrows():
            unit_id = assigned.get(index)
            if unit_id is None:
                reason = "no candidate within tolerance"
                results.append({**row.to_dict(), "matched": False, "reason": reason})
                continue
            unit = candidates.loc[unit_id]
            results.append(
                {
                    **row.to_dict(),
                    "matched": True,
                    "reason": "",
                    "mastr_id": unit["EinheitMastrNummer"],
                    "lon": float(unit["Laengengrad"]),
                    "lat": float(unit["Breitengrad"]),
                    "mastr_height": float(unit["Nabenhoehe"])
                    if pd.notna(unit["Nabenhoehe"])
                    else np.nan,
                    "mastr_capacity": float(unit["Bruttoleistung"]),
                    "mastr_rotor": float(unit["Rotordurchmesser"])
                    if pd.notna(unit["Rotordurchmesser"])
                    else np.nan,
                    "mastr_commissioned": unit["commissioned"],
                    "mastr_type": unit.get("Typenbezeichnung"),
                    "windpark": unit.get("NameWindpark"),
                }
            )
    return pd.DataFrame(results)


def main(archive: Path = ARCHIVE, out: Path = OUT) -> int:
    """Parse, match, and write the tables and the summary."""
    print(f"parsing {archive}")
    units = parse_units(archive)
    print(f"  {len(units)} wind units; {int(units['Laengengrad'].notna().sum())} with a coordinate")

    fleet = register_fleet()
    print(f"register: {len(fleet)} turbines, {fleet['postcode'].nunique()} postcodes")

    matched = match(fleet, units)
    rate = float(matched["matched"].mean())
    capacity_rate = float(
        matched.loc[matched["matched"], "capacity"].sum() / matched["capacity"].sum()
    )
    print(f"\nmatched {int(matched['matched'].sum())} of {len(matched)} turbines ({rate:.1%})")
    print(f"matched share of register capacity: {capacity_rate:.1%}")
    print("\nunmatched by reason:")
    print(matched.loc[~matched["matched"], "reason"].value_counts().to_string())

    unmatched_postcodes = (
        matched[~matched["matched"]].groupby("postcode").size().sort_values(ascending=False)
    )
    print(f"\npostcodes with an unmatched turbine: {len(unmatched_postcodes)}")
    print(unmatched_postcodes.head(10).to_string())

    moved = matched[matched["matched"]].copy()
    print(
        f"\ndistinct coordinates: {len(moved[['lon', 'lat']].drop_duplicates())} matched, "
        f"against 1161 postcode centroids for the whole register"
    )
    if len(moved):
        print(
            "hub height, register against MaStR: "
            f"median {moved['height'].median():.1f} m and {moved['mastr_height'].median():.1f} m"
        )

    out.mkdir(parents=True, exist_ok=True)
    units.to_csv(out / "mastr_units.csv", index=False)
    matched.to_csv(out / "de_match.csv", index=False)
    # The override a cache rebuild reads: matched units only, so an unmatched
    # turbine keeps the postcode centroid it has today rather than being
    # dropped from a fleet its observations cover.
    override = moved.loc[moved["mastr_height"].notna(), ["ID", "lon", "lat", "mastr_height"]]
    override = override.rename(columns={"mastr_height": "height"})
    override.to_csv(out / "de_fleet_override.csv", index=False)
    summary = {
        "archive": str(archive),
        "mastr_units": int(len(units)),
        "mastr_units_with_coordinates": int(units["Laengengrad"].notna().sum()),
        "register_turbines": int(len(fleet)),
        "matched": int(matched["matched"].sum()),
        "match_rate": rate,
        "matched_capacity_share": capacity_rate,
        "distinct_coordinates_matched": int(len(moved[["lon", "lat"]].drop_duplicates())),
        "override_units": int(len(moved.loc[moved["mastr_height"].notna()])),
        "tolerances": {
            "capacity_relative": CAPACITY_TOLERANCE,
            "rotor_m": ROTOR_TOLERANCE_M,
            "days": DATE_TOLERANCE_DAYS,
        },
    }
    (out / "de_match_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"\nwrote {out}/de_match.csv and three more")
    return 0


def cli(argv: list[str] | None = None) -> int:
    """Parse the recorded command line, which has no arguments, and run :func:`main`."""
    parser = make_parser(__doc__)
    parser.add_argument(
        "--archive", type=Path, default=ARCHIVE, help=f"MaStR export (default: {ARCHIVE})"
    )
    parser.add_argument("--out", type=Path, default=OUT, help=f"Output directory (default: {OUT})")
    args = parser.parse_args(argv)
    return main(archive=args.archive, out=args.out)


if __name__ == "__main__":
    raise SystemExit(cli())
