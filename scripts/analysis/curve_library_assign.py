"""The T2 other-brand assignment for the curve library study.

T2 asks what happens when a unit that is currently on its own maker's curve is
moved to another maker's, at the same scale and the same specific power. If
held-out skill survives that, specific power is carrying the curve's shape and
the brand is not; if it does not, curve assignment matters more than the
specific-power match assumes.

The rule, fixed before any T2 run:

- only units the curve-match audit classes as **same brand** move. A unit
  already on another brand's curve, on a reference design, or whose maker
  cannot be identified keeps its T0 assignment, since there is nothing to
  change or no way to know;
- a mover goes to the model of a **different brand** whose specific power is
  nearest its own, within the same rating band `assign_curves_from_library`
  uses, 0.5 to 2 times the unit's per-turbine rating;
- a reference design is not a brand, so it is never a destination. The audit's
  own classification decides that, rather than a second list kept here;
- a unit with no candidate in its band keeps T0, and the share that happens to
  is reported.

The classification is imported from `curve_match_audit`, not restated, so that
T2 moves exactly the units the scorecard's Other brand column counts.

Read-only and importable: the study driver and the tests use the same rules.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curve_match_audit as audit  # noqa: E402

#: The rating band a candidate must fall in, as a multiple of the unit's own
#: per-turbine rating. Taken from ``vwf.datasets.eia_us.SCALE_BAND`` so that T2
#: and the US assignment agree about what "the same scale" means.
SCALE_BAND = (0.5, 2.0)


def specific_power(capacity_kw, diameter_m):
    """W/m2 from a per-turbine rating in kW and a rotor diameter in m."""
    capacity = pd.to_numeric(capacity_kw, errors="coerce")
    diameter = pd.to_numeric(diameter_m, errors="coerce")
    area = np.pi * (diameter / 2.0) ** 2
    return (capacity * 1000.0) / area.where(area > 0)


def other_brand_assignment(
    fleet: pd.DataFrame, own: pd.Series, models: pd.DataFrame, *, rating_kw: pd.Series,
) -> pd.DataFrame:
    """The T2 key for every unit, and why it is what it is.

    Args:
        fleet: units with ``ID``, ``model`` (their T0 key) and ``diameter``.
        own: each unit's own manufacturer, as ``curve_match_audit`` reads it.
        models: the curve catalogue, with ``manufacturer``, ``model``,
            ``capacity`` and ``p_density``.
        rating_kw: per-turbine rating, which is not ``capacity`` for a fleet
            whose rows are plants.

    Returns:
        A frame indexed like ``fleet`` with ``t2_model`` and ``t2_reason``.
        ``t2_reason`` is one of ``moved``, ``not-same-brand``, ``unverifiable``
        or ``no-candidate-in-band``.
    """
    lut = models.assign(model=models["model"].astype(str)).set_index("model")["manufacturer"]
    t0 = fleet["model"].astype(str)
    curve_maker = audit.curve_side_manufacturer(t0, lut)
    classes = [audit.classify(o, m) for o, m in zip(own, curve_maker)]

    unit_sp = specific_power(rating_kw, fleet["diameter"])
    rating = pd.to_numeric(rating_kw, errors="coerce")
    catalogue = models.dropna(subset=["p_density", "capacity"]).reset_index(drop=True)
    # Candidates are read through the audit's own reading of a curve's maker,
    # exactly as the T0 side is. Taking the catalogue's manufacturer column
    # raw would make "Unknown" a brand, and a distributed reference curve a
    # legitimate destination, which it is not.
    catalogue = catalogue.assign(curve_manufacturer=audit.curve_side_manufacturer(
        catalogue["model"].astype(str), lut))

    keys, reasons = [], []
    for i, (klass, own_maker) in enumerate(zip(classes, own)):
        if klass == "unverifiable":
            keys.append(t0.iloc[i]), reasons.append("unverifiable")
            continue
        if klass != "same":
            keys.append(t0.iloc[i]), reasons.append("not-same-brand")
            continue
        kw, sp = rating.iloc[i], unit_sp.iloc[i]
        band = catalogue[catalogue["capacity"].between(kw * SCALE_BAND[0], kw * SCALE_BAND[1])] \
            if pd.notna(kw) else catalogue.iloc[0:0]
        other = band[[audit.classify(own_maker, m) == "different-brand"
                      for m in band["curve_manufacturer"]]]
        if other.empty or pd.isna(sp):
            keys.append(t0.iloc[i]), reasons.append("no-candidate-in-band")
            continue
        nearest = (other["p_density"] - sp).abs().idxmin()
        keys.append(str(other.loc[nearest, "model"])), reasons.append("moved")

    return pd.DataFrame({"t2_model": keys, "t2_reason": reasons}, index=fleet.index)


def moved_share(fleet: pd.DataFrame, assignment: pd.DataFrame, weight: str = "capacity") -> float:
    """The share of capacity T2 actually moves, which is what it can test."""
    w = pd.to_numeric(fleet[weight], errors="coerce").fillna(0.0)
    total = float(w.sum())
    return float(w[assignment["t2_reason"] == "moved"].sum()) / total if total else 0.0
