"""Power curves and curve assignment: which curve each unit is simulated on.

Moved from ``vwf.data`` so that the adapters under ``vwf.sources``, which
assign curves at load time, can import it without importing the data
orchestration above them. ``vwf.data`` imports these names back for its own
use.
"""

from __future__ import annotations

import difflib

import numpy as np
import pandas as pd

import vwf.wind as wind
from vwf.config import PyVWFPaths


def _default_power_curve(power_curves: pd.DataFrame) -> str:
    """Pick a default turbine model from a power curve table.

    The same curve the simulation falls back to for a missing model
    (:func:`vwf.wind.default_curve_key`), so the two cannot drift apart.
    """
    key = wind.default_curve_key(power_curves)
    if key is None:
        raise ValueError("power_curves has no turbine model columns.")
    return key


def load_power_curves():
    """Load turbine power curves.

    Reads ``power_curves.csv`` from the configured input root, falling back to
    the open curve library bundled with the package (with a warning). See
    :meth:`vwf.config.PyVWFPaths.reference_file`.

    Returns:
        DataFrame with wind speed in the ``data$speed`` column and one
        capacity-factor column per turbine model, on a 0 to 40 m/s grid.
    """
    return pd.read_csv(PyVWFPaths.reference_file("power_curves.csv"))


def add_models(df: pd.DataFrame) -> pd.DataFrame:
    """Assign turbine model names based on metadata.

    Args:
        df: Turbine metadata.

    Returns:
        DataFrame with a ``model`` column added, and ``model_match`` naming how
        each turbine was matched: ``"fuzzy-manufacturer+specific-power"`` (a model
        within 1 W/m2 of the turbine's specific power from a manufacturer whose
        name fuzzily matches, at a difflib cutoff of 0.3) or
        ``"specific-power-only"`` (the nearest specific power across all models,
        within 100 W/m2). Turbines with neither are dropped. The fuzzy match is
        loose: "ewt" scores 0.4 against "vestasv", so a Vestas V27 (392.98 W/m2)
        can be matched to the EWT DW54 at the same specific power. The first tier
        therefore does not guarantee the turbine's own manufacturer.
    """
    models = pd.read_csv(PyVWFPaths.reference_file("models.csv"))
    models["model"] = models["model"].astype("string")
    models["manufacturer"] = models["manufacturer"].astype("string").str.lower().fillna("")
    models = models.sort_values("p_density").reset_index(drop=True)

    df = df.copy()

    # --- Ensure required columns exist ---
    required = ["ID", "capacity", "diameter", "height", "lon", "lat"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"add_models: missing required columns: {missing}")

    # --- Coerce numerics safely ---
    for c in ["capacity", "diameter", "height", "lon", "lat"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows missing core numeric fields
    df = df.dropna(subset=["ID", "capacity", "diameter", "height", "lon", "lat"]).reset_index(
        drop=True
    )

    # Remove unrealistic turbines
    df = df.loc[df["height"] >= 1].reset_index(drop=True)

    # --- Manufacturer cleanup (difflib cannot handle pd.NA) ---
    if "manufacturer" in df.columns:
        df["manufacturer"] = df["manufacturer"].astype("string").str.lower().fillna("")
    else:
        df["manufacturer"] = ""

    # Default type
    if "type" not in df.columns:
        df["type"] = "onshore"
    else:
        df["type"] = df["type"].astype("string").fillna("onshore")

    df["ID"] = df["ID"].astype(str)

    # Compute power density (NOTE: your capacity is in kW; convert to W for density)
    df["p_density"] = (df["capacity"] * 1000.0) / (np.pi * (df["diameter"] / 2.0) ** 2)

    # --- Fuzzy match manufacturer against model manufacturers ---
    # Create candidate pairs by manufacturer similarity, then pick closest p_density
    # (This keeps your original logic but makes it robust.)
    cand = models.assign(
        match=models["manufacturer"].apply(
            lambda x: difflib.get_close_matches(x, df["manufacturer"].tolist(), cutoff=0.3, n=50)
        )
    ).explode("match")

    if cand["match"].isna().all():
        # If no manufacturer matches at all, fall back to nearest p_density later
        df["model"] = pd.NA
    else:
        merged = df.merge(
            cand.drop_duplicates(subset=["manufacturer", "match", "model", "p_density"]),
            left_on="manufacturer",
            right_on="match",
            how="left",
            suffixes=("", "_m"),
        )

        # choose closest p_density among matched manufacturer candidates
        merged["closest"] = (merged["p_density"] - merged["p_density_m"]).abs()
        merged = merged.sort_values(["ID", "closest"])
        merged = merged.drop_duplicates(subset=["ID"], keep="first")

        # accept manufacturer-based match only if close enough
        merged["model"] = merged["model"].where(merged["closest"] < 1, pd.NA)

        df = merged[
            ["ID", "type", "capacity", "diameter", "height", "lon", "lat", "p_density", "model"]
        ].copy()

    # --- Final fallback: nearest p_density across all models ---
    # merge_asof requires sorted keys
    df = df.sort_values("p_density").reset_index(drop=True)
    fallback = pd.merge_asof(
        df[["p_density"]],
        models[["p_density", "model"]],
        on="p_density",
        direction="nearest",
        tolerance=100,
    )["model"]

    # Record which tier matched each turbine, so a run can say how its fleet
    # got its curves (vwf.provenance.curve_resolution reads this).
    # A specific-power-only match is a different machine chosen for its rotor
    # loading, not the turbine's own.
    matched_by_manufacturer = df["model"].notna()
    df["model"] = df["model"].fillna(fallback)
    df["model_match"] = np.where(
        matched_by_manufacturer, "fuzzy-manufacturer+specific-power", "specific-power-only"
    )

    # Drop if still no model
    df = df.dropna(subset=["model"]).reset_index(drop=True)

    # Keep types clean
    df["capacity"] = df["capacity"].astype(float)
    df["diameter"] = df["diameter"].astype(float)
    df["height"] = df["height"].astype(float)

    df = df.sort_values("ID").reset_index(drop=True)
    return df
