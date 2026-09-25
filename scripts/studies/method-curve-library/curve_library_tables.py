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

**Both fleets, since 2026-09-13.** A table was first built from the training
fleet alone, which is not the set a condition has to reach: a run fits the
training fleet and is scored on the test fleet, and the two differ by every
unit installed after the training window and every unit gone before the test
year. Denmark and the United Kingdom passed that construction because their
training fleets are strict subsets of their test fleets, so the table happened
to apply in full to both; Germany, whose fleet also shrinks, was refused, and
the United States passed with two units to spare. The table now covers the
union of the two fleets, and declares per unit which fleet holds it, so the
driver's check stays as strong as it was. Evidence for the subset claim is in
``docs/findings/method-curve-library-prereg.md``.

Read-only. It writes override tables and a report under ``<out_dir>``, and
runs nothing.

Usage, from the repository root:

    PYVWF_INPUT=input/combined PYTHONPATH=src:scripts/analysis python \\
        scripts/studies/method-curve-library/curve_library_tables.py <out_dir>
"""

import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parents[1] / "analysis")]  # siblings, then the tools
import baseline_bootstrap as bb  # noqa: E402
from pyvwf.harness.driver import load_obs_and_fleet  # noqa: E402
import curve_library_assign as t2rule  # noqa: E402
import curve_library_match as matcher  # noqa: E402
import curve_match_audit as audit  # noqa: E402
from pyvwf.cli.common import make_parser  # noqa: E402
from pyvwf.harness.regions import load_region  # noqa: E402

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


class Inputs(NamedTuple):
    """The files the tables are built from, each a flag defaulting to the recorded path."""

    open_models: Path = OPEN_MODELS
    combined_models: Path = COMBINED_MODELS
    rerun: Path = RERUN
    refresh: Path = REFRESH


INPUTS = Inputs()


#: The fleet fields each condition's rule reads, and therefore the fields a
#: unit in both fleets has to agree on. Disagreement would give that unit two
#: keys, one per phase, which is not a condition anyone registered. The lists
#: are per condition because they differ: C2 maps a model key to a substitute
#: and reads nothing else, while T2 reads the rating and the rotor to pick a
#: band and a specific power.
#:
#: **The per-condition list is load-bearing, and the country rows are where it
#: bears.** A country grid point records capacity per year, so the same point
#: is a different object in the two fleets: 14 Belgian points hold 0 MW in the
#: training fleet and 11 to 18 MW in the test one. The union takes the training
#: row, so a condition that read capacity there would read the wrong one. C2
#: reads no capacity, so the union is safe for C2 by the condition's own
#: shape and not by anything the country grids guarantee. **A later condition
#: that reads capacity at country level will be refused here, and that is the
#: correct behaviour**: the refusal says the two fleets disagree about the
#: object, which is true, and whoever adds that condition has to decide which
#: year's capacity it means rather than inheriting a silent choice. Nothing is
#: fixed in anticipation, since there is no way to know which answer a
#: condition that does not exist would want.
#:
#: Checked rather than assumed, and the check is what found the Belgian case.
RULE_FIELDS = {
    "C1": ("model",),
    "C2": ("model",),
    "T1": ("uswtdb_model",),
    "T2": ("model", "capacity", "diameter", "n_turbines", "uswtdb_model"),
}


def train_fleet_of(code: str, inputs: Inputs = INPUTS) -> pd.DataFrame:
    """The training fleet a row fits, from the run standing in the scorecard."""
    rerun, refresh = Path(inputs.rerun), Path(inputs.refresh)
    root = rerun if (rerun / code).is_dir() else refresh
    runs = sorted((root / code).glob("train-*"))
    # One training run per region, or the fleet would be pooled across runs
    # that need not agree; a second run beside the first is refused, not mixed.
    if len(runs) != 1:
        raise SystemExit(f"{code}: expected one training run under {root / code}, found {runs}")
    files = sorted(runs[0].glob("train_turb_info_*.csv"))
    if not files:
        raise SystemExit(f"{code}: no training fleet under {runs[0]}")
    return audit.training_fleet(files)


def test_fleet_of(code: str) -> pd.DataFrame:
    """The fleet the row is scored on, by the route ``run_evaluate`` takes.

    ``pyvwf.harness.driver.load_obs_and_fleet`` is the part of ``val_set`` that
    loads no ERA5, and ``val_set`` calls it: the same ``prep_country`` call and
    the same narrowing to the units the test year observes. Reusing it keeps
    one definition of the test fleet rather than a second one written here that
    could drift from the first.

    A country row's grid names its own model keys, so
    ``prepare_country_fleet`` never reaches for a default curve and the fleet
    does not depend on which input root is loaded. The agreement check in
    :func:`load_fleets` tests that independently: the training fleet on disk
    was written by a run under the row's own root.
    """
    spec = load_region(Path("configs/regions/scorecard") / f"{bb.CONFIGS[code]}.toml")
    _, fleet = load_obs_and_fleet(spec, int(spec.test_years[0]))
    return fleet


class Fleets(NamedTuple):
    """Both fleets of one row, and the union a condition assigns over.

    ``union`` carries the training row for a unit in both, which is the frame
    the rules read. A per-fleet quantity is taken from ``train`` or ``test``
    instead: the country grids record capacity per year, so a unit's share of
    its fleet is not the same number on the two sides.
    """

    union: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame


def load_fleets(code: str, condition: str, inputs: Inputs = INPUTS) -> Fleets:
    """Both fleets of a row, with the union the condition's rule assigns over.

    Raises:
        SystemExit: if a unit in both fleets carries different values for a
            field this condition's rule reads. The table would then owe that
            unit two keys, and which one applied would depend on the phase.
    """
    train, test = train_fleet_of(code, inputs), test_fleet_of(code)
    train_ids = set(train["ID"].astype(str))
    test_ids = set(test["ID"].astype(str))

    a = train.assign(ID=train["ID"].astype(str)).drop_duplicates("ID").set_index("ID")
    b = test.assign(ID=test["ID"].astype(str)).drop_duplicates("ID").set_index("ID")
    shared = sorted(train_ids & test_ids)
    for field in RULE_FIELDS[condition]:
        if field not in a.columns or field not in b.columns:
            continue
        x, y = a.loc[shared, field], b.loc[shared, field]
        if x.dtype.kind in "fciu" or y.dtype.kind in "fciu":
            xn, yn = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
            differ = ~((xn - yn).abs() < 1e-9) & ~(xn.isna() & yn.isna())
        else:
            differ = x.astype(str) != y.astype(str)
        if bool(differ.any()):
            examples = [(i, x[i], y[i]) for i in list(differ[differ].index)[:3]]
            raise SystemExit(
                f"{code}: {int(differ.sum())} units carry a different {field} in the "
                f"training and test fleets, for example {examples}. {condition} reads "
                "this field, so these units have no single key."
            )

    extra = test[~test["ID"].astype(str).isin(train_ids)]
    union = pd.concat([train, extra], ignore_index=True)
    ids = union["ID"].astype(str)
    union["in_train"] = ids.isin(train_ids)
    union["in_test"] = ids.isin(test_ids)
    return Fleets(union, train, test)


def write_table(out_dir: Path, name: str, fleet: pd.DataFrame, keys: pd.Series) -> pd.DataFrame:
    """Write one override table, carrying which fleet holds each unit."""
    table = pd.DataFrame(
        {
            "ID": fleet["ID"].astype(str),
            "model": keys,
            "in_train": fleet["in_train"].to_numpy(),
            "in_test": fleet["in_test"].to_numpy(),
        }
    ).dropna(subset=["model"])
    table.to_csv(out_dir / f"{name}.csv", index=False)
    return table


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
    return {
        "substitute": str(row["model"]),
        "d_specific_power": float(row["p_density"]) - float(source["p_density"]),
        "d_rating_kw": float(row["capacity"]) - kw,
    }


def extremes(fleet: pd.DataFrame, applied: pd.Series) -> list[dict]:
    """What the rule did to the smallest and largest unit, by capacity."""
    cap = pd.to_numeric(fleet["capacity"], errors="coerce")
    out = []
    for label, i in (("smallest", cap.idxmin()), ("largest", cap.idxmax())):
        out.append(
            {
                "end": label,
                "ID": str(fleet.loc[i, "ID"]),
                "capacity": float(cap.loc[i]),
                "from": str(fleet.loc[i, "model"]),
                "to": str(applied.get(str(fleet.loc[i, "ID"]), fleet.loc[i, "model"])),
            }
        )
    return out


def share(fleet: pd.DataFrame, ids) -> float:
    """The capacity share the ids hold of ``fleet``, by that fleet's capacity."""
    cap = pd.to_numeric(fleet["capacity"], errors="coerce").fillna(0.0)
    total = float(cap.sum())
    hit = fleet["ID"].astype(str).isin({str(i) for i in ids})
    return float(cap[hit].sum()) / total if total else 0.0


def coverage(fleets: Fleets, table: pd.DataFrame) -> dict:
    """Units and capacity a table reaches, in each fleet separately.

    A condition reaches a different share of the fleet it is fitted on and the
    fleet it is scored on, and one number for both hides exactly the gap that
    made the single-fleet construction look adequate.
    """
    ids = set(table["ID"].astype(str))
    return {
        "units_union": int(len(fleets.union)),
        "units_train": int(len(fleets.train)),
        "units_test": int(len(fleets.test)),
        "reached_train": int(fleets.train["ID"].astype(str).isin(ids).sum()),
        "reached_test": int(fleets.test["ID"].astype(str).isin(ids).sum()),
        "capacity_share_train": share(fleets.train, ids),
        "capacity_share_test": share(fleets.test, ids),
    }


def build_c2(out_dir: Path, inputs: Inputs = INPUTS) -> list[dict]:
    """C2: each country grid key replaced by the nearest in-band open model."""
    combined = pd.read_csv(inputs.combined_models)
    open_lib = combined[combined["library"] == "open"].reset_index(drop=True)
    catalogue = combined.set_index("model")
    report = []
    for code in COUNTRY_ROWS:
        fleets = load_fleets(code, "C2", inputs)
        fleet = fleets.union
        mapping = {}
        for key in sorted(fleet["model"].astype(str).unique()):
            if key not in catalogue.index:
                report.append(
                    {
                        "condition": "C2",
                        "region": code,
                        "key": key,
                        "substitute": None,
                        "note": "key not in the catalogue",
                    }
                )
                continue
            found = nearest_in_band(key, open_lib, catalogue.loc[key])
            mapping[key] = found["substitute"]
            units = fleet["ID"][fleet["model"].astype(str) == key]
            report.append(
                {
                    "condition": "C2",
                    "region": code,
                    "key": key,
                    "own_specific_power": float(catalogue.loc[key, "p_density"]),
                    "own_rating_kw": float(catalogue.loc[key, "capacity"]),
                    **found,
                    "capacity_share_train": share(fleets.train, units),
                    "capacity_share_test": share(fleets.test, units),
                }
            )
        table = write_table(out_dir, f"C2_{code}", fleet, fleet["model"].astype(str).map(mapping))
        report.append({"condition": "C2", "region": code, **coverage(fleets, table)})
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
        from pyvwf.config import PyVWFPaths

        md = pd.read_csv(PyVWFPaths.TURBINE_DATA / "DK/dk_md.csv")
        lut = md.assign(ID=md["ID"].astype(str)).drop_duplicates("ID").set_index("ID")
        return ids.map(lut["manufacturer"]), ids.map(lut["model"])
    if code == "UK":
        from pyvwf.loaders import load_turbine_metadata

        md = load_turbine_metadata("UK").assign(ID=lambda d: d["ID"].astype(str))
        lut = md.drop_duplicates("ID").set_index("ID")["manufacturer"]
        return None, ids.map(lut)  # packed: the field holds both
    if code == "US":
        return None, fleet["uswtdb_model"]
    raise SystemExit(f"{code}: no designation field is known for this register")


def build_t1(out_dir: Path, inputs: Inputs = INPUTS) -> list[dict]:
    """T1: the brand-and-spec match, for the rows whose register names machines."""
    combined = pd.read_csv(inputs.combined_models)
    licensed = combined[combined["library"] == "real"]
    index = matcher.build_index(licensed)
    report = []
    for code in T1_ROWS:
        fleets = load_fleets(code, "T1", inputs)
        fleet = fleets.union
        maker, machine = designation_fields(code, fleet)
        if maker is None:
            maker = pd.Series([None] * len(fleet), index=fleet.index)
        keys = pd.Series(
            [matcher.match(a, b, index) for a, b in zip(maker, machine)], index=fleet.index
        )
        table = write_table(out_dir, f"T1_{code}", fleet, keys)
        report.append(
            {
                "condition": "T1",
                "region": code,
                "designation": "model column"
                if code == "DK"
                else ("packed into manufacturer" if code == "UK" else "uswtdb_model"),
                "units_matched": int(len(table)),
                **coverage(fleets, table),
                "distinct_keys": int(table["model"].nunique()),
            }
        )
        for row in extremes(fleet, table.set_index("ID")["model"]):
            report.append({"condition": "T1", "region": code, **row})
    return report


def build_t2(out_dir: Path, inputs: Inputs = INPUTS) -> list[dict]:
    """T2: same-brand units moved to the nearest in-band other-brand model."""
    combined = pd.read_csv(inputs.combined_models)
    catalogue = combined.set_index("model")
    report = []
    for code in T2_ROWS:
        fleets = load_fleets(code, "T2", inputs)
        fleet = fleets.union
        own, source = audit.own_manufacturer(code, fleet)
        rating = per_turbine_rating(fleet)
        got = t2rule.other_brand_assignment(fleet, own, combined, rating_kw=rating)
        moved = got["t2_reason"] == "moved"
        table = write_table(out_dir, f"T2_{code}", fleet[moved], got["t2_model"][moved])
        distances = []
        for old, new in zip(fleet["model"].astype(str)[moved], got["t2_model"][moved]):
            if old in catalogue.index and new in catalogue.index:
                distances.append(
                    (
                        float(catalogue.loc[new, "p_density"])
                        - float(catalogue.loc[old, "p_density"]),
                        float(catalogue.loc[new, "capacity"])
                        - float(catalogue.loc[old, "capacity"]),
                    )
                )
        d = pd.DataFrame(distances, columns=["d_specific_power", "d_rating_kw"])
        report.append(
            {
                "condition": "T2",
                "region": code,
                "own_manufacturer": source,
                "units_moved": int(moved.sum()),
                **coverage(fleets, table),
                "moved_share_union": t2rule.moved_share(fleet, got),
                **{f"reason_{k}": int(v) for k, v in got["t2_reason"].value_counts().items()},
                "median_d_specific_power": float(d["d_specific_power"].abs().median())
                if len(d)
                else None,
                "max_d_specific_power": float(d["d_specific_power"].abs().max())
                if len(d)
                else None,
                "max_d_rating_kw": float(d["d_rating_kw"].abs().max()) if len(d) else None,
            }
        )
        for row in extremes(fleet, table.set_index("ID")["model"]):
            report.append({"condition": "T2", "region": code, **row})
    return report


def build_c1(inputs: Inputs = INPUTS) -> list[dict]:
    """C1 has no override table: it changes the library, not the keys.

    What it needs reporting is whether the keys the grids name resolve in the
    combined library, since that is the whole condition: under the open library
    they do not, and every unit falls back.
    """
    combined = set(pd.read_csv(inputs.combined_models)["model"].astype(str))
    open_lib = set(pd.read_csv(inputs.open_models)["model"].astype(str))
    report = []
    for code in COUNTRY_ROWS:
        fleets = load_fleets(code, "C1", inputs)
        fleet = fleets.union
        keys = fleet["model"].astype(str)
        report.append(
            {
                "condition": "C1",
                "region": code,
                "keys": ", ".join(sorted(keys.unique())),
                "units_train": int(len(fleets.train)),
                "units_test": int(len(fleets.test)),
                "capacity_share_resolving_combined": share(
                    fleets.test, fleet["ID"][keys.isin(combined)]
                ),
                "capacity_share_resolving_open": share(
                    fleets.test, fleet["ID"][keys.isin(open_lib)]
                ),
            }
        )
    return report


def main(out_dir: str | Path, inputs: Inputs = INPUTS) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = (
        build_c1(inputs)
        + build_c2(out_dir, inputs)
        + build_t1(out_dir, inputs)
        + build_t2(out_dir, inputs)
    )
    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "override_report.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for condition in ("C1", "C2", "T1", "T2"):
            part = report[report["condition"] == condition].dropna(axis=1, how="all")
            print(f"\n=== {condition}")
            print(part.to_string(index=False))


def cli(argv: list[str] | None = None) -> None:
    """Parse the recorded command line, ``<out_dir>``, and run :func:`main`."""
    parser = make_parser(__doc__)
    parser.add_argument("out_dir", help="Directory for the tables and report, under output/")
    for name, default, text in (
        ("--open-models", OPEN_MODELS, "The open library's models.csv"),
        ("--combined-models", COMBINED_MODELS, "The combined library's model catalogue"),
        ("--rerun", RERUN, "The European re-run, whose training fleets stand first"),
        ("--refresh", REFRESH, "The refresh runs, for rows the re-run does not hold"),
    ):
        parser.add_argument(name, type=Path, default=default, help=f"{text} (default: {default})")
    args = parser.parse_args(argv)
    main(args.out_dir, Inputs(args.open_models, args.combined_models, args.rerun, args.refresh))


if __name__ == "__main__":
    cli()
