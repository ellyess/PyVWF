"""Build the curve library study's override tables, and report what they reach.

Registered in ``docs/findings/method-curve-library-prereg.md``. Each condition
reassigns model keys, and none of them runs until this has said how much of a
fleet it reaches and how far it moves what it touches. Coverage before the
gate is fixed is what made T1's 50.2% and 53.3% trustworthy; C2 and T2 are held
to the same standard.

**Distance matters as much as coverage for C2 and T2.** Both substitute the
nearest in-band model, and "nearest" can still be far: the open library's
closest in-band match for a V90-3.0 is 350 W/m2 against 472. A null from a
condition whose substitutes happened to sit close reads as "the library is
adequate" when it means "these substitutes were nearly the same curve". So
every replacement is reported with its distance in specific power and in
rating, before the condition runs, and a result is read against them.

Per `findings-doc`, each condition also reports what its rule does to the
smallest and the largest unit in the fleet, since that is where a rule written
against the typical unit stops behaving as its author intended.

Read-only. It writes override tables and a report under ``<out_dir>``, and
runs nothing.

Usage, from the repository root:

    PYVWF_INPUT=input/combined PYTHONPATH=src:scripts/analysis python \\
        scripts/analysis/curve_library_tables.py <out_dir>
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curve_library_assign as t2rule  # noqa: E402
import curve_library_match as matcher  # noqa: E402
import curve_match_audit as audit  # noqa: E402

COUNTRY_ROWS = ("BE", "ES", "FR", "IE", "IT", "NO", "PT", "SE")
#: T1 needs a register designation; DE records none, so it is not in this list
#: (the finding is in the pre-registration). The US is built and reported, and
#: ungated by rule.
T1_ROWS = ("DK", "UK", "US")
T2_ROWS = ("DE", "DK", "UK", "US")

OPEN_MODELS = Path("input/reference/models.csv")
COMBINED_MODELS = Path("input/combined/reference/models_with_library.csv")
RERUN = Path("output/eu_rerun_2026-09-12/new")
REFRESH = Path("output/validation/refresh_2026-08-24")


def fleet_of(code: str) -> pd.DataFrame:
    """The training fleet a row fits, from the run standing in the scorecard."""
    root = RERUN if (RERUN / code).is_dir() else REFRESH
    files = sorted((root / code).glob("train-*/train_turb_info_*.csv"))
    if not files:
        raise SystemExit(f"{code}: no training fleet under {root}")
    return pd.read_csv(files[0])


def per_turbine_rating(fleet: pd.DataFrame) -> pd.Series:
    """Rating of one machine, which is not ``capacity`` for a fleet of plants."""
    capacity = pd.to_numeric(fleet["capacity"], errors="coerce")
    if "n_turbines" in fleet.columns:
        n = pd.to_numeric(fleet["n_turbines"], errors="coerce").replace(0, pd.NA)
        return capacity / n.fillna(1)
    return capacity


def nearest_in_band(key: str, catalogue: pd.DataFrame, source: pd.Series) -> dict:
    """The in-band model nearest ``key`` in specific power, with the distance."""
    kw = float(source["capacity"])
    band = catalogue[catalogue["capacity"].between(*[kw * b for b in t2rule.SCALE_BAND])]
    band = band[band["model"] != key]
    if band.empty:
        return {"substitute": None, "d_specific_power": None, "d_rating_kw": None}
    nearest = (band["p_density"] - float(source["p_density"])).abs().idxmin()
    row = band.loc[nearest]
    return {"substitute": str(row["model"]),
            "d_specific_power": float(row["p_density"]) - float(source["p_density"]),
            "d_rating_kw": float(row["capacity"]) - kw}


def extremes(fleet: pd.DataFrame, applied: pd.Series) -> list[dict]:
    """What the rule did to the smallest and largest unit, by capacity."""
    cap = pd.to_numeric(fleet["capacity"], errors="coerce")
    out = []
    for label, i in (("smallest", cap.idxmin()), ("largest", cap.idxmax())):
        out.append({"end": label, "ID": str(fleet.loc[i, "ID"]),
                    "capacity": float(cap.loc[i]), "from": str(fleet.loc[i, "model"]),
                    "to": str(applied.get(str(fleet.loc[i, "ID"]), fleet.loc[i, "model"]))})
    return out


def share(fleet: pd.DataFrame, ids) -> float:
    cap = pd.to_numeric(fleet["capacity"], errors="coerce").fillna(0.0)
    total = float(cap.sum())
    hit = fleet["ID"].astype(str).isin({str(i) for i in ids})
    return float(cap[hit].sum()) / total if total else 0.0


def build_c2(out_dir: Path) -> list[dict]:
    """C2: each country grid key replaced by the nearest in-band open model."""
    combined = pd.read_csv(COMBINED_MODELS)
    open_lib = combined[combined["library"] == "open"].reset_index(drop=True)
    catalogue = combined.set_index("model")
    report = []
    for code in COUNTRY_ROWS:
        fleet = fleet_of(code)
        mapping = {}
        for key in sorted(fleet["model"].astype(str).unique()):
            if key not in catalogue.index:
                report.append({"condition": "C2", "region": code, "key": key,
                               "substitute": None, "note": "key not in the catalogue"})
                continue
            found = nearest_in_band(key, open_lib, catalogue.loc[key])
            mapping[key] = found["substitute"]
            units = fleet["ID"][fleet["model"].astype(str) == key]
            report.append({"condition": "C2", "region": code, "key": key,
                           "own_specific_power": float(catalogue.loc[key, "p_density"]),
                           "own_rating_kw": float(catalogue.loc[key, "capacity"]),
                           **found, "capacity_share": share(fleet, units)})
        table = pd.DataFrame({"ID": fleet["ID"].astype(str),
                              "model": fleet["model"].astype(str).map(mapping)})
        table = table.dropna(subset=["model"])
        table.to_csv(out_dir / f"C2_{code}.csv", index=False)
        for row in extremes(fleet, table.set_index("ID")["model"]):
            report.append({"condition": "C2", "region": code, **row})
    return report


def designation_fields(code: str, fleet: pd.DataFrame) -> tuple[pd.Series | None, pd.Series]:
    """The maker and the machine designation, as each register writes them.

    T1 needs the designation, and the three registers keep it in three shapes:
    Denmark in its own ``model`` column beside a ``manufacturer`` one, the
    United Kingdom packed into the manufacturer field ("Vestas V90 3000"), and
    the United States in ``uswtdb_model``. Reading the manufacturer for all
    three, as the curve-match audit does, matches a maker against itself and
    finds nothing: that is what produced a coverage of 0.0% for Denmark on the
    first run of this builder, against 50.2% measured directly.
    """
    ids = fleet["ID"].astype(str)
    if code == "DK":
        # Read from the processed register, not through load_turbine_metadata:
        # its column allowlist for DK keeps ID, manufacturer, capacity,
        # diameter, height, lon, lat and location_type, and drops `model`. So
        # the Danish designation never reaches the pipeline at all, one layer
        # earlier than add_models not reading one.
        from vwf.config import PyVWFPaths
        md = pd.read_csv(PyVWFPaths.TURBINE_DATA / "DK/dk_md.csv")
        lut = md.assign(ID=md["ID"].astype(str)).drop_duplicates("ID").set_index("ID")
        return ids.map(lut["manufacturer"]), ids.map(lut["model"])
    if code == "UK":
        from vwf.loaders import load_turbine_metadata
        md = load_turbine_metadata("UK").assign(ID=lambda d: d["ID"].astype(str))
        lut = md.drop_duplicates("ID").set_index("ID")["manufacturer"]
        return None, ids.map(lut)          # packed: the field holds both
    if code == "US":
        return None, fleet["uswtdb_model"]
    raise SystemExit(f"{code}: no designation field is known for this register")


def build_t1(out_dir: Path) -> list[dict]:
    """T1: the brand-and-spec match, for the rows whose register names machines."""
    combined = pd.read_csv(COMBINED_MODELS)
    licensed = combined[combined["library"] == "real"]
    index = matcher.build_index(licensed)
    report = []
    for code in T1_ROWS:
        fleet = fleet_of(code)
        maker, machine = designation_fields(code, fleet)
        if maker is None:
            maker = pd.Series([None] * len(fleet), index=fleet.index)
        keys = [matcher.match(a, b, index) for a, b in zip(maker, machine)]
        table = pd.DataFrame({"ID": fleet["ID"].astype(str), "model": keys}).dropna()
        table.to_csv(out_dir / f"T1_{code}.csv", index=False)
        report.append({"condition": "T1", "region": code,
                       "designation": "model column" if code == "DK"
                       else ("packed into manufacturer" if code == "UK" else "uswtdb_model"),
                       "units_matched": int(len(table)), "units": int(len(fleet)),
                       "capacity_share": share(fleet, table["ID"]),
                       "distinct_keys": int(table["model"].nunique())})
        for row in extremes(fleet, table.set_index("ID")["model"]):
            report.append({"condition": "T1", "region": code, **row})
    return report


def build_t2(out_dir: Path) -> list[dict]:
    """T2: same-brand units moved to the nearest in-band other-brand model."""
    combined = pd.read_csv(COMBINED_MODELS)
    catalogue = combined.set_index("model")
    report = []
    for code in T2_ROWS:
        fleet = fleet_of(code)
        own, source = audit.own_manufacturer(code, fleet)
        rating = per_turbine_rating(fleet)
        got = t2rule.other_brand_assignment(fleet, own, combined, rating_kw=rating)
        moved = got["t2_reason"] == "moved"
        table = pd.DataFrame({"ID": fleet["ID"].astype(str)[moved],
                              "model": got["t2_model"][moved]})
        table.to_csv(out_dir / f"T2_{code}.csv", index=False)
        distances = []
        for old, new in zip(fleet["model"].astype(str)[moved], got["t2_model"][moved]):
            if old in catalogue.index and new in catalogue.index:
                distances.append((float(catalogue.loc[new, "p_density"])
                                  - float(catalogue.loc[old, "p_density"]),
                                  float(catalogue.loc[new, "capacity"])
                                  - float(catalogue.loc[old, "capacity"])))
        d = pd.DataFrame(distances, columns=["d_specific_power", "d_rating_kw"])
        report.append({"condition": "T2", "region": code, "own_manufacturer": source,
                       "units_moved": int(moved.sum()), "units": int(len(fleet)),
                       "capacity_share": t2rule.moved_share(fleet, got),
                       **{f"reason_{k}": int(v) for k, v in
                          got["t2_reason"].value_counts().items()},
                       "median_d_specific_power": float(d["d_specific_power"].abs().median())
                       if len(d) else None,
                       "max_d_specific_power": float(d["d_specific_power"].abs().max())
                       if len(d) else None,
                       "max_d_rating_kw": float(d["d_rating_kw"].abs().max())
                       if len(d) else None})
        for row in extremes(fleet, table.set_index("ID")["model"]):
            report.append({"condition": "T2", "region": code, **row})
    return report


def build_c1() -> list[dict]:
    """C1 has no override table: it changes the library, not the keys.

    What it needs reporting is whether the keys the grids name resolve in the
    combined library, since that is the whole condition: under the open library
    they do not, and every unit falls back.
    """
    combined = set(pd.read_csv(COMBINED_MODELS)["model"].astype(str))
    open_lib = set(pd.read_csv(OPEN_MODELS)["model"].astype(str))
    report = []
    for code in COUNTRY_ROWS:
        fleet = fleet_of(code)
        keys = fleet["model"].astype(str)
        report.append({"condition": "C1", "region": code,
                       "keys": ", ".join(sorted(keys.unique())),
                       "capacity_share_resolving_combined":
                           share(fleet, fleet["ID"][keys.isin(combined)]),
                       "capacity_share_resolving_open":
                           share(fleet, fleet["ID"][keys.isin(open_lib)])})
    return report


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_c1() + build_c2(out_dir) + build_t1(out_dir) + build_t2(out_dir)
    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "override_report.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for condition in ("C1", "C2", "T1", "T2"):
            part = report[report["condition"] == condition].dropna(axis=1, how="all")
            print(f"\n=== {condition}")
            print(part.to_string(index=False))


if __name__ == "__main__":
    main()
