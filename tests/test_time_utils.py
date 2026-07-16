"""Tests for temporal resolution parsing and column assignment."""
import pandas as pd
import pytest

from vwf.time_utils import (
    parse_time_slice,
    add_time_resolution_columns,
    TIME_SLICE_MAPPING,
)


@pytest.mark.parametrize("slice_str,expected", [
    ("winter", [1, 2, 12]),
    ("spring", [3, 4, 5]),
    ("summer", [6, 7, 8]),
    ("autumn", [9, 10, 11]),
    ("1/6", [1, 2]),
    ("1/1", list(range(1, 13))),
    ("3", [3]),
    ("11", [11]),
])
def test_parse_time_slice(slice_str, expected):
    assert parse_time_slice(slice_str) == expected


def test_every_named_slice_covers_valid_months():
    for months in TIME_SLICE_MAPPING.values():
        assert all(1 <= m <= 12 for m in months)


def test_bimonth_slices_partition_the_year():
    months = []
    for key in ["1/6", "2/6", "3/6", "4/6", "5/6", "6/6"]:
        months.extend(parse_time_slice(key))
    assert sorted(months) == list(range(1, 13))


def test_add_time_resolution_columns():
    df = pd.DataFrame({"month": [1, 4, 7, 10]})
    out = add_time_resolution_columns(df)
    assert list(out["bimonth"]) == ["1/6", "2/6", "4/6", "5/6"]
    assert list(out["season"]) == ["winter", "spring", "summer", "autumn"]
    assert (out["fixed"] == "1/1").all()


def test_add_time_resolution_columns_all_months_assigned():
    df = pd.DataFrame({"month": list(range(1, 13))})
    out = add_time_resolution_columns(df)
    assert out["season"].notna().all()
    assert out["bimonth"].notna().all()


# ---------------------------------------------------------------------------
# Explicit season definitions. Default None must stay
# bit-identical to the legacy NH behaviour; explicit SH definitions must
# actually change the answer (must-distinguish).
# ---------------------------------------------------------------------------

SH_SEASONS = {
    "summer": [12, 1, 2],
    "autumn": [3, 4, 5],
    "winter": [6, 7, 8],
    "spring": [9, 10, 11],
}


def test_parse_time_slice_explicit_seasons():
    assert parse_time_slice("winter", seasons=SH_SEASONS) == [6, 7, 8]
    assert parse_time_slice("winter") == [1, 2, 12]
    # Hemisphere-neutral slices are unaffected by the seasons argument.
    assert parse_time_slice("1/6", seasons=SH_SEASONS) == [1, 2]
    assert parse_time_slice("7", seasons=SH_SEASONS) == [7]


def test_add_time_resolution_columns_explicit_seasons():
    df = pd.DataFrame({"month": [1, 7]})
    nh = add_time_resolution_columns(df.copy())
    sh = add_time_resolution_columns(df.copy(), seasons=SH_SEASONS)
    assert list(nh["season"]) == ["winter", "summer"]
    assert list(sh["season"]) == ["summer", "winter"]  # inverted, as it must be
    # bimonth and fixed are hemisphere-neutral: identical either way.
    assert list(sh["bimonth"]) == list(nh["bimonth"])
    assert (sh["fixed"] == "1/1").all()


def test_explicit_nh_seasons_match_default_exactly():
    """Passing the NH mapping explicitly is a no-op relative to the default —
    the guarantee that lets Europe configs carry explicit months without
    changing legacy results."""
    nh_explicit = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "autumn": [9, 10, 11],
    }
    df = pd.DataFrame({"month": list(range(1, 13))})
    default = add_time_resolution_columns(df.copy())
    explicit = add_time_resolution_columns(df.copy(), seasons=nh_explicit)
    pd.testing.assert_frame_equal(default, explicit)
    for name in nh_explicit:
        assert set(parse_time_slice(name, seasons=nh_explicit)) == set(
            parse_time_slice(name)
        )
