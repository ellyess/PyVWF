"""The T2 other-brand assignment's rules (scripts/studies/method-curve-library/curve_library_assign.py).

T2 moves a unit from its own maker's curve to another maker's at the same scale
and specific power, so what it can test is the share of capacity it actually
moves. These tests pin which units move, where they go, and which are left
alone on purpose, on a catalogue small enough to check by hand.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "curve_library_assign",
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "studies"
    / "method-curve-library"
    / "curve_library_assign.py",
)
assign = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(assign)

#: Four utility machines at 2 MW and one distributed machine far below the band.
MODELS = pd.DataFrame(
    {
        "manufacturer": ["Vestas", "Gamesa", "Enercon", "Vestas", "Unknown"],
        "model": [
            "Vestas.V90.2000",
            "Gamesa.G90.2000",
            "Enercon.E82.2000",
            "Vestas.V80.2000",
            "2019COE_DW100_100kW_27.6",
        ],
        "capacity": [2000, 2000, 2000, 2000, 100],
        "p_density": [314.0, 314.4, 379.0, 398.0, 167.1],
    }
)


def fleet(model, diameter=90.0, capacity=2000.0, unit_id="a"):
    return pd.DataFrame(
        {"ID": [unit_id], "model": [model], "diameter": [diameter], "capacity": [capacity]}
    )


def assignment(model, own, **kwargs):
    f = fleet(model, **kwargs)
    return assign.other_brand_assignment(f, pd.Series([own]), MODELS, rating_kw=f["capacity"])


def test_a_unit_on_its_own_makers_curve_moves_to_another_maker():
    got = assignment("Vestas.V90.2000", "Vestas Wind Systems A/S")
    assert got["t2_reason"].iloc[0] == "moved"
    key = got["t2_model"].iloc[0]
    assert key != "Vestas.V90.2000"
    assert MODELS.set_index("model").loc[key, "manufacturer"] != "Vestas"


def test_it_moves_to_the_nearest_specific_power_not_the_first_candidate():
    """The unit is 314 W/m2. Gamesa at 314.4 is nearer than Enercon at 379."""
    got = assignment("Vestas.V90.2000", "Vestas")
    assert got["t2_model"].iloc[0] == "Gamesa.G90.2000"


def test_a_unit_already_on_another_brand_is_left_alone():
    """T2 asks what moving a same-brand unit does. One already moved has
    nothing to change, and moving it again would test something else."""
    got = assignment("Gamesa.G90.2000", "Vestas")
    assert got["t2_reason"].iloc[0] == "not-same-brand"
    assert got["t2_model"].iloc[0] == "Gamesa.G90.2000"


def test_a_unit_whose_maker_is_unknown_is_left_alone():
    for own in ("Unknown", None, ""):
        got = assignment("Vestas.V90.2000", own)
        assert got["t2_reason"].iloc[0] == "unverifiable"
        assert got["t2_model"].iloc[0] == "Vestas.V90.2000"


def test_a_reference_design_is_never_a_destination():
    """A composite is not a brand. The only candidate outside the band here is
    the distributed reference curve, and the band excludes it anyway; this
    pins that a unit with no real alternative keeps T0 rather than taking it."""
    got = assignment("Vestas.V90.2000", "Vestas", capacity=100.0, diameter=27.6)
    assert got["t2_reason"].iloc[0] == "no-candidate-in-band"
    assert got["t2_model"].iloc[0] == "Vestas.V90.2000"


def test_the_rating_band_is_half_to_double():
    """A 5 MW unit has no candidate among 2 MW machines: 5000 x 0.5 is 2500."""
    got = assignment("Vestas.V90.2000", "Vestas", capacity=5000.0, diameter=126.0)
    assert got["t2_reason"].iloc[0] == "no-candidate-in-band"
    got = assignment("Vestas.V90.2000", "Vestas", capacity=3500.0, diameter=110.0)
    assert got["t2_reason"].iloc[0] == "moved"  # 3500 x 0.5 = 1750, so 2000 is in


def test_the_moved_share_is_by_capacity_not_by_count():
    f = pd.DataFrame(
        {
            "ID": ["big", "small"],
            "model": ["Vestas.V90.2000", "Gamesa.G90.2000"],
            "diameter": [90.0, 90.0],
            "capacity": [9000.0, 1000.0],
        }
    )
    got = assign.other_brand_assignment(
        f, pd.Series(["Vestas", "Vestas"]), MODELS, rating_kw=pd.Series([2000.0, 2000.0])
    )
    assert list(got["t2_reason"]) == ["moved", "not-same-brand"]
    assert assign.moved_share(f, got) == pytest.approx(0.9)


def test_specific_power_is_watts_per_swept_area():
    assert assign.specific_power(pd.Series([2000.0]), pd.Series([90.0])).iloc[0] == pytest.approx(
        2000 * 1000 / (3.14159265 * 45**2), rel=1e-6
    )
    assert pd.isna(assign.specific_power(pd.Series([2000.0]), pd.Series([0.0])).iloc[0])
