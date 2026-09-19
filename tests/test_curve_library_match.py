"""The brand-and-spec matcher's rules (scripts/studies/method-curve-library/curve_library_match.py).

T1 of the curve library study simulates a unit on the curve of the machine its
register names, so the matcher decides which units the condition can reach at
all: its coverage is what makes G2a gateable for a region. The rules are fixed
before any T1 run and pinned here on strings taken from the registers
themselves, not invented for the test.

The cases that matter are the ones where the register and the library disagree
about how to write the same machine, and the ones where they look like they
agree and do not.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "curve_library_match",
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "studies"
    / "method-curve-library"
    / "curve_library_match.py",
)
matcher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(matcher)

#: A slice of the licensed library, shaped as models.csv is.
LIBRARY = pd.DataFrame(
    {
        "manufacturer": [
            "Vestas",
            "Vestas",
            "Vestas",
            "Bonus",
            "REpower",
            "Siemens",
            "GE",
            "GE",
            "Nordex",
            "Neg Micon",
        ],
        "model": [
            "Vestas.V47.660",
            "Vestas.V90.3000",
            "Vestas.V110.2000",
            "Bonus.B44.600",
            "REpower.MM82.2000",
            "Siemens.SWT.2.3.93",
            "GE.1.5sle",
            "GE.1.5s",
            "Nordex.N90.2500",
            "NegMicon.NM48.750",
        ],
    }
)


@pytest.fixture(scope="module")
def index():
    return matcher.build_index(LIBRARY)


def test_normalise_drops_everything_that_is_not_a_letter_or_digit():
    assert matcher.normalise("V 47-660") == "v47660"
    assert matcher.normalise("Vestas Wind Systems A/S") == "vestaswindsystemsas"
    assert matcher.normalise(None) == ""


def test_a_packed_register_string_matches(index):
    """The British register writes the maker and the machine in one field."""
    assert matcher.match(None, "Vestas V90 3000", index) == "Vestas.V90.3000"
    assert matcher.match(None, "Bonus B44 600", index) == "Bonus.B44.600"


def test_split_register_fields_match(index):
    """The Danish register writes them apart, with a corporate suffix."""
    assert matcher.match("Vestas Wind Systems A/S", "V 47-660", index) == "Vestas.V47.660"
    assert matcher.match("NEG Micon", "NM 48/750", index) == "NegMicon.NM48.750"


def test_a_rating_in_megawatts_matches_a_key_in_kilowatts(index):
    """``Vestas V110-2.0`` and ``Vestas.V110.2000`` are the same machine. This
    is the rule that moved US coverage from 9.7% to 23.5% of capacity."""
    assert matcher.match(None, "Vestas V110-2.0", index) == "Vestas.V110.2000"
    assert matcher.match("Vestas", "V110-2.0 MW", index) == "Vestas.V110.2000"


def test_stripping_a_rating_never_eats_the_model_number(index):
    """``V47`` must not become ``V``: only a rating after a separator goes."""
    assert matcher.without_trailing_rating("V47") == "v47"
    assert matcher.without_trailing_rating("SWP10-14TG20") == "swp1014tg20"
    assert matcher.match("Vestas Wind Systems A/S", "V 47", index) == "Vestas.V47.660"


def test_a_maker_alone_is_not_a_machine(index):
    """Half the British fleet records only ``Siemens`` or ``Vestas``. A brand
    cannot name a curve, and guessing one is what T0 already does."""
    assert matcher.match(None, "Siemens", index) is None
    assert matcher.match("Vestas", None, index) is None


def test_an_unknown_register_row_matches_nothing(index):
    assert matcher.match("Unknown", "Unknown", index) is None
    assert matcher.match(None, None, index) is None
    assert matcher.match("", "  ", index) is None


def test_a_machine_the_library_does_not_carry_matches_nothing(index):
    """The Danish fleet's Solid Wind and Gaia machines are not in the library,
    and the American register names GE variants the library distinguishes and
    it does not: ``GE Wind GE1.5-77`` could be ``GE.1.5s`` or ``GE.1.5sle``."""
    assert matcher.match("Solid Wind Power A/S", "SWP10-14TG20", index) is None
    assert matcher.match(None, "GE Wind GE1.5-77", index) is None


def test_an_ambiguous_form_is_dropped_rather_than_resolved():
    """Two keys sharing a form without their ratings is not an exact match."""
    ambiguous = pd.DataFrame(
        {
            "manufacturer": ["Vestas", "Vestas"],
            "model": ["Vestas.V80.1800", "Vestas.V80.2000"],
        }
    )
    index = matcher.build_index(ambiguous)
    assert matcher.match("Vestas", "V80", index) is None  # which one?
    assert matcher.match("Vestas", "V80-2.0", index) == "Vestas.V80.2000"


def test_the_manufacturer_aliases_are_applied(index):
    assert matcher.canonical_manufacturer("Vestas Wind Systems A/S") == "vestas"
    assert matcher.canonical_manufacturer("GE Wind") == "ge"
    assert matcher.canonical_manufacturer("Nordex") == "nordex"  # already agrees
