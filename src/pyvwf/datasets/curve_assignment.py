"""Curve assignment from a curve library, by scale and specific power.

Shared by the adapters whose registers give a unit's rating and rotor but no
curve: the US, New Zealand, Chile and Argentina. It lived in
``pyvwf.datasets.eia_us``, so New Zealand's processing imported a US module to
reach it (#85); ``eia_us`` still re-exports both names for the scripts that
import them from there.
"""

from __future__ import annotations

import pandas as pd

#: Rated-capacity band a candidate curve must fall in, as a multiple of the
#: plant's per-turbine kW. Matching on specific power alone assigns a 1 kW
#: micro turbine to a 4 MW machine (the bundled open library spans both), so
#: scale is constrained before specific power decides.
SCALE_BAND = (0.5, 2.0)


def assign_curves_from_library(metadata: pd.DataFrame, *, fallback_model: str) -> pd.DataFrame:
    """Match each plant to a real power curve by specific power.

    The US fleet spans roughly 200-390 W/m2 of specific power, so a single
    uniform curve mis-specifies where most plants reach rated power.

    This deliberately does NOT call :func:`pyvwf.curves.add_models`. That matches
    on specific power alone, which is safe against a utility-only catalogue
    (as AU-NEM's licensed library is) but not against the BUNDLED OPEN
    library, which also carries distributed machines down to 1 kW. A 1.5 kW
    Pika sits near 212 W/m2, indistinguishable from a modern utility turbine
    on specific power, so plain add_models assigned sub-100 kW curves to 28%
    of the US fleet - once pairing a 1 kW SWIFT with 4.2 MW machines. Scale is
    therefore constrained first (see ``SCALE_BAND``), specific power second.

    ``capacity`` in the metadata is the PLANT total (kW), but ``add_models``
    expects PER-TURBINE kW - the units trap ``scripts/region_tools/assign_au_curves.py``
    documents - so it is divided by ``n_turbines`` here.

    Plants without a USWTDB rotor diameter or turbine count keep
    ``fallback_model``; ``model_source`` records which of the two applied.

    Args:
        metadata: Plant metadata with ``diameter``, ``n_turbines``, ``capacity``.
        fallback_model: Uniform curve key for plants that cannot be matched.

    Returns:
        Metadata with ``model``/``model_source`` reassigned.
    """
    import numpy as np

    from pyvwf.config import PyVWFPaths

    out = metadata.copy()
    matchable = (
        out["diameter"].notna()
        & (out["diameter"] > 0)
        & out["n_turbines"].notna()
        & (out["n_turbines"] > 0)
        & out["capacity"].notna()
    )
    if not matchable.any():
        # Same contract as the per-plant path below: unmatchable plants take
        # the fallback explicitly, rather than silently keeping whatever the
        # caller had already put in ``model``.
        out["model"] = fallback_model
        out["model_source"] = "default-uniform"
        return out

    catalog = pd.read_csv(PyVWFPaths.reference_file("models.csv"))
    catalog = catalog[
        (catalog["offshore"].astype(str).str.lower() == "no")
        & catalog["capacity"].gt(0)
        & catalog["p_density"].gt(0)
    ]

    per_turbine_kw = out["capacity"] / out["n_turbines"]
    swept = np.pi * (out["diameter"] / 2.0) ** 2
    plant_p_density = (per_turbine_kw * 1000.0) / swept

    models, sources = [], []
    for idx in out.index:
        if not matchable[idx]:
            models.append(fallback_model)
            sources.append("default-uniform")
            continue
        kw, pd_target = per_turbine_kw[idx], plant_p_density[idx]
        # SCALE GUARD. Matching on specific power ALONE picks a 1 kW micro
        # turbine for a 4 MW machine, because the bundled open library mixes
        # distributed and utility turbines at overlapping W/m2. Restrict to
        # curves of a comparable rating first, then take nearest specific
        # power inside that band.
        band = catalog[catalog["capacity"].between(kw * SCALE_BAND[0], kw * SCALE_BAND[1])]
        if band.empty:
            models.append(fallback_model)
            sources.append("default-uniform-no-curve-at-scale")
            continue
        pick = band.iloc[(band["p_density"] - pd_target).abs().argmin()]
        models.append(pick["model"])
        sources.append("matched-scale-and-specific-power")

    out["model"] = models
    out["model_source"] = sources
    return out
