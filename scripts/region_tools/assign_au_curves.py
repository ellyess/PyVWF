#!/usr/bin/env python
"""Assign power-curve models to the AU fleet, per library (D2 dual-library run).

Two strategies, per the D2 rulings:

REAL library (PRIMARY, the gate):
    The method-consistent, D1-validated path: farms with a sourced rotor
    diameter go through the SAME vwf.data.add_models logic as every other
    region (fuzzy manufacturer + nearest p_density, global p_density
    fallback), fed with per-turbine capacity. Manufacturer-only farms match
    within manufacturer by nearest per-turbine RATED CAPACITY; fallback
    farms take the capacity-nearest model across the catalog, marked.

OPEN library (SECOND, robustness rider):
    Manufacturer-first matching is ACTIVELY HARMFUL here (its only Vestas
    machines are V27–V82; a V150 farm would pair with a V82 curve), so
    farms with a p_density match into the MW-scale onshore
    references/composites by nearest p_density; farms without one take the
    IEC Class 2 normalized industry composite, marked. The proxy-matching
    caveat travels with every open-stack result.

Writes au_nem_md_real.csv / au_nem_md_open.csv next to au_nem_md.csv, each
being the AEMONemSource metadata contract with `model` + `model_source`.

Run with PYVWF_INPUT pointing at the staging dir holding the REAL
models.csv (add_models resolves the catalog through PyVWFPaths).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-csv", default="configs/curation/au_turbine_models.csv")
    ap.add_argument("--md", default="input/observations/turbine/AU_NEM/au_nem_md.csv")
    ap.add_argument("--open-models", required=True, help="models_open.csv path")
    ap.add_argument("--open-curves", required=True, help="open power-curve CSV path")
    args = ap.parse_args()

    from vwf.data import add_models  # resolves models.csv via PYVWF_INPUT

    md = pd.read_csv(args.md, parse_dates=["commissioning_date"])
    tm = pd.read_csv(args.models_csv)
    open_models = pd.read_csv(args.open_models)
    open_curve_cols = pd.read_csv(args.open_curves, nrows=0).columns[1:]

    tm["rotor_diameter_m"] = pd.to_numeric(tm["rotor_diameter_m"], errors="coerce")
    tm["unit_mw_eff"] = pd.to_numeric(tm["unit_mw_eff"], errors="coerce")
    merged = md.drop(columns=["model"]).merge(
        tm[["ID", "manufacturer", "model_string", "rotor_diameter_m",
            "unit_mw_eff", "p_density_wm2", "confidence"]],
        on="ID", how="left",
    )

    # ---------------- REAL library ----------------
    with_diam = merged[merged["rotor_diameter_m"].notna()].copy()
    am_input = pd.DataFrame(
        {
            "ID": with_diam["ID"],
            "capacity": with_diam["unit_mw_eff"] * 1000.0,  # per-turbine kW
            "diameter": with_diam["rotor_diameter_m"],
            "height": with_diam["height"],
            "lon": with_diam["lon"],
            "lat": with_diam["lat"],
            "manufacturer": with_diam["manufacturer"],
            "type": with_diam["type"],
        }
    )
    matched = add_models(am_input)[["ID", "model"]]

    real = merged.merge(matched, on="ID", how="left")
    real["model_source"] = np.where(real["model"].notna(), "add_models", None)

    from vwf.config import PyVWFPaths
    cat = pd.read_csv(PyVWFPaths.reference_file("models.csv"))
    cat["mk"] = cat["manufacturer"].astype(str).str.lower()

    def real_fill(row):
        if pd.notna(row["model"]):
            return row["model"], row["model_source"]
        unit_kw = (row["unit_mw_eff"] * 1000.0) if pd.notna(row["unit_mw_eff"]) else row["capacity"] / 50
        pool = cat
        src = "fallback-capacity-nearest"
        if pd.notna(row["manufacturer"]):
            mpool = cat[cat["mk"].str.contains(str(row["manufacturer"]).lower().split()[0], na=False)]
            if len(mpool):
                pool, src = mpool, "manufacturer-capacity-nearest"
        pick = pool.iloc[(pool["capacity"] - unit_kw).abs().argmin()]
        return pick["model"], src

    fills = real.apply(real_fill, axis=1, result_type="expand")
    real["model"], real["model_source"] = fills[0], fills[1]

    # ---------------- OPEN library ----------------
    mw = open_models[(open_models["capacity"] >= 1500) & (open_models["offshore"] == "no")]
    generic = "IEC_Class2_Normalized_Industry_Composite"
    assert generic in open_curve_cols, "IEC Class 2 composite missing from open curves"

    def open_pick(row):
        pdens = row["p_density_wm2"]
        if pd.notna(pdens):
            pick = mw.iloc[(mw["p_density"] - pdens).abs().argmin()]
            return pick["model"], "pdensity-class"
        return generic, "iec-class2-generic"

    open_md = merged.copy()
    picks = open_md.apply(open_pick, axis=1, result_type="expand")
    open_md["model"], open_md["model_source"] = picks[0], picks[1]

    base_cols = ["ID", "site_name", "region", "lon", "lat", "height", "capacity",
                 "model", "model_source", "type", "commissioning_date"]
    out_dir = Path(args.md).parent
    real[base_cols].to_csv(out_dir / "au_nem_md_real.csv", index=False)
    open_md[base_cols].to_csv(out_dir / "au_nem_md_open.csv", index=False)

    for name, frame in (("REAL", real), ("OPEN", open_md)):
        print(f"\n=== {name} assignment ===")
        print(frame["model_source"].value_counts().to_string())
        print("top models:", frame["model"].value_counts().head(8).to_dict())
        missing = frame["model"].isna().sum()
        assert missing == 0, f"{name}: {missing} farms without a model"
    print(f"\nwritten: {out_dir/'au_nem_md_real.csv'} and au_nem_md_open.csv")


if __name__ == "__main__":
    main()
