"""Cross-manufacturer curve matching, per fitted fleet.

A unit can resolve to a curve its ``power_curves.csv`` contains and still be
simulated on the wrong machine: ``vwf.data.add_models`` and ``assign_curves_from_library`` match
on specific power, and the manufacturer tier of ``add_models`` is fuzzy enough
to cross brands. This compares each unit's own manufacturer with the
manufacturer of the curve it was assigned, over the training fleet a run fitted.

Classes. Pre-registered on 2026-09-11, before any share was computed:

- ``same``: a brand word of the assigned model's manufacturer is a word of the
  unit's own manufacturer string (lower-case alphanumeric words, aliases below);
- ``different``: the assigned model's manufacturer is not the unit's own;
- ``unverifiable``: the unit's own manufacturer is missing, blank or "unknown".

``different`` is reported in two parts, ``different-brand`` (another
manufacturer's model) and ``different-reference`` (a research reference,
generic or composite curve, which is never the unit's own machine); that split
was not pre-registered (see below).

Capacity share and unit count are reported separately for each class, since a
mismatch concentrated in a few large units is a different problem from one
spread across many small ones.

The unit's own manufacturer is read per region from what records it: the raw
register for DK, DE and UK (``load_turbine_metadata``), ``true_model`` for NZ,
``uswtdb_model`` for US, and the curated spec tables for AU-NEM, CL and AR.
Nothing records it for BR, so BR is wholly unverifiable. The source used is
written to the output. The assigned model's manufacturer comes from the
``models.csv`` the run used, found by the sha256 in its manifest. That is a
different table from ``power_curves.csv``: a key can have a curve in
``power_curves.csv`` and no row in ``models.csv`` (the open library's
normalized composites are curve-only by design). A key whose manufacturer is
missing or "Unknown" in ``models.csv``, or that is absent from it, is a
reference curve when the open library's ``power_curves_provenance.csv``
names it, and unverifiable otherwise.

The rule is strict by design and names brands, not lineages: a Bonus turbine on
a Siemens curve counts as ``different-brand``, and a GE plant on the DOE
reference curve of a GE 1.5 MW machine (``DOE_GE_1.5MW_77``, 6.6% of US
capacity) as ``different-reference``, because a reference design is not the
unit's own curve. Read the ``top_different_brand_pairs`` column before drawing
conclusions from a share.

What was decided when, beyond the pre-registered classes:

- **Settled after looking at samples of the strings, before any share was
  computed:** the per-region sources above, whole-word matching, and the split
  of ``different`` into brand and reference.
- **Fixed after the first results:** a missing own manufacturer was being read
  as the string "na" and classed as different rather than unverifiable.
- **Added after review, on the same day, after the results were in:**
  ``unverifiable`` also covers the curve side. A curve whose manufacturer is
  missing or "Unknown" in ``models.csv``, or whose key is absent from
  ``models.csv``, and which the open library's provenance file does not name,
  now classes as unverifiable instead of as a reference curve. The reason is
  symmetry: an unidentified curve is as uncheckable as an unrecorded own
  manufacturer, and the first rule would have counted it as a mismatch.
  Re-running every audited fleet under the new rule changed no share or count
  (maximum difference 0.0), because no audited unit sits on a curve that
  neither ``models.csv`` nor the open provenance identifies. The rule was
  changed because it gave the right answer for the wrong reason, not to move a
  number.

It reads the run tree under the git-ignored ``output/`` and the local input
tree, so it is not runnable by a third party; it is committed so the reported
shares can be regenerated from the runs that produced them.

Usage (from the repository root, ``src`` on the path):
    python scripts/analysis/curve_match_audit.py --out <csv>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

REFRESH = Path("output/validation/refresh_2026-08-24")
#: The fleet each audited row fitted: every turbine-level scorecard row, plus
#: the two runs behind the DK and NZ correlations in method-country-level.md
#: section 7.
RUNS = {
    **{
        c: REFRESH / c / "train-refresh"
        for c in ["DE", "DK", "UK", "US", "BR", "AU-NEM", "NZ", "CL", "AR"]
    },
    "DK (train-ppopen)": Path("output/validation/DK/train-ppopen"),
    "NZ (train-k147)": Path("output/validation/NZ/train-k147"),
}

#: Manufacturer strings in the curve libraries that are not a commercial brand:
#: research reference designs, national-lab composites and placeholders.
REFERENCE = {
    "bar",
    "cf",
    "doe",
    "dtu",
    "iea",
    "leanwind",
    "nps",
    "nrel",
    "sd",
    "swift",
    "reference",
}
#: Curve-side label for a model no source identifies.
UNRECORDED = "unrecorded"
ALIASES = {"vestasv": "vestas", "anbonus": "bonus"}
#: Words that never identify a brand on their own.
STOP = {"neg", "wind", "systems", "energy", "power", "as", "a", "s", "gmbh", "ag"}
UNKNOWN = {"", "unknown", "nan", "none"}


def words(text: object) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", str(text).lower())
    return [ALIASES.get(t, t) for t in toks]


def classify(own: object, model_manufacturer: object) -> str:
    if own is None or (not isinstance(own, str) and pd.isna(own)):
        return "unverifiable"
    own_words = [w for w in words(own) if w not in STOP]
    if not own_words or " ".join(own_words) in UNKNOWN:
        return "unverifiable"
    if model_manufacturer == UNRECORDED:
        return "unverifiable"
    brand = [w for w in words(model_manufacturer) if w not in STOP]
    if not brand or any(w in REFERENCE for w in brand):
        return "different-reference"
    return "same" if any(b in own_words for b in brand) else "different-brand"


def own_manufacturer(region: str, fleet: pd.DataFrame) -> tuple[pd.Series, str]:
    """The unit's own manufacturer string, and where it was read from."""
    ids = fleet["ID"].astype(str)
    base = region.split(" ")[0]
    if base in {"DK", "DE", "UK"}:
        from vwf.loaders import load_turbine_metadata

        md = load_turbine_metadata(base)
        lut = md.assign(ID=md["ID"].astype(str)).set_index("ID")["manufacturer"]
        return ids.map(lut), f"load_turbine_metadata('{base}').manufacturer"
    if base == "NZ":
        return fleet["true_model"], "fleet true_model"
    if base == "US":
        return fleet["uswtdb_model"], "fleet uswtdb_model"
    spec = {
        "AU-NEM": "au_turbine_models.csv",
        "CL": "cl_turbine_specs.csv",
        "AR": "ar_turbine_specs.csv",
    }.get(base)
    if spec:
        s = pd.read_csv(Path("configs/curation") / spec)
        lut = s.assign(ID=s["ID"].astype(str)).drop_duplicates("ID").set_index("ID")["manufacturer"]
        return ids.map(lut), f"configs/curation/{spec}.manufacturer"
    return pd.Series(pd.NA, index=fleet.index), "none available"


def curve_side_manufacturer(keys: pd.Series, lut: pd.Series) -> pd.Series:
    """The assigned curve's manufacturer, or ``reference`` / ``unrecorded``.

    A missing or "Unknown" manufacturer, or a key absent from ``models.csv``,
    is a reference curve only if the open library's provenance file names the
    key; otherwise nothing identifies the curve and it cannot be checked.
    """
    from importlib import resources

    prov = pd.read_csv(str(resources.files("vwf.resources") / "power_curves_provenance.csv"))
    man = keys.map(lut)
    unknown = man.isna() | man.astype(str).str.strip().str.lower().isin(UNKNOWN)
    identified = keys.isin(set(prov["column"]))
    fallback = pd.Series(UNRECORDED, index=keys.index).where(~identified, "reference")
    return man.where(~unknown, fallback)


def _models_file(lib: dict) -> Path:
    """The models table a run used, found by its recorded sha256.

    Manifest paths predate the input/ reorganisation of 2026-07-23 for the older
    runs, so the hash, not the path, identifies the file.
    """
    candidates = [
        Path(lib["models_path"]),
        Path("input/reference/models.csv"),
        Path("input/combined/reference/models.csv"),
        Path("src/vwf/resources/models.csv"),
    ]
    for c in candidates:
        if c.is_file() and hashlib.sha256(c.read_bytes()).hexdigest() == lib["models_sha256"]:
            return c
    raise FileNotFoundError(f"no models.csv on disk matches sha256 {lib['models_sha256']}")


def audit(region: str, run_dir: Path) -> dict:
    fleet_file = sorted(run_dir.glob("train_turb_info_*.csv"))[0]
    fleet = pd.read_csv(fleet_file, low_memory=False)
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    lib = manifest["curve_library"]
    models_file = _models_file(lib)
    models = pd.read_csv(models_file)
    lut = models.drop_duplicates("model").set_index("model")["manufacturer"]
    model_manufacturer = curve_side_manufacturer(fleet["model"], lut)

    own, source = own_manufacturer(region, fleet)
    cls = pd.Series([classify(o, m) for o, m in zip(own, model_manufacturer)], index=fleet.index)
    cap = pd.to_numeric(fleet["capacity"], errors="coerce").fillna(0.0)
    total = cap.sum()

    row = {
        "region": region,
        "fleet_file": str(fleet_file),
        "library": lib["library"],
        "models_file": str(models_file),
        "own_manufacturer_source": source,
        "n_units": len(fleet),
    }
    row["curve_unrecorded_units"] = int((model_manufacturer == UNRECORDED).sum())
    for c in ["same", "different-brand", "different-reference", "unverifiable"]:
        m = cls == c
        row[f"{c}_cap_share"] = float(cap[m].sum() / total) if total else float("nan")
        row[f"{c}_units"] = int(m.sum())
    pairs = (
        pd.DataFrame(
            {
                "own": own.astype(str),
                "curve_manufacturer": model_manufacturer,
                "model": fleet["model"],
                "cap": cap,
                "cls": cls,
            }
        )
        .query("cls == 'different-brand'")
        .groupby(["own", "curve_manufacturer"])["cap"]
        .sum()
        .sort_values(ascending=False)
    )
    row["top_different_brand_pairs"] = "; ".join(
        f"{o} -> {m} ({c / total:.1%})" for (o, m), c in pairs.head(4).items()
    )
    row["different_cap_share"] = (
        row["different-brand_cap_share"] + row["different-reference_cap_share"]
    )
    row["different_units"] = row["different-brand_units"] + row["different-reference_units"]
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    table = pd.DataFrame([audit(r, d) for r, d in RUNS.items()])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    cols = [
        "region",
        "library",
        "n_units",
        "same_cap_share",
        "same_units",
        "different_cap_share",
        "different_units",
        "different-brand_cap_share",
        "different-brand_units",
        "different-reference_cap_share",
        "different-reference_units",
        "unverifiable_cap_share",
        "unverifiable_units",
    ]
    with pd.option_context(
        "display.width", 250, "display.max_columns", 20, "display.float_format", "{:.3f}".format
    ):
        print(table[cols].to_string(index=False))


if __name__ == "__main__":
    main()
