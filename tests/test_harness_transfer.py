"""Transfer semantics (design §7): collapse, pair enforcement, name matching.

The mirrored-hemisphere test is normative for §7.3 and must-distinguish by
construction: it computes the transferred output under season-NAME matching
and under month matching, asserts the two DIFFER, and asserts the
name-matched one is correct against planted ground truth. A fixture on which
both modes agree would pass vacuously and is explicitly not acceptable
(design §8, test 8).
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from vwf.harness.driver import check_transfer_pair, collapse_factors
from vwf.wind import correct_wind_speed

NH_SEASONS = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
}
SH_SEASONS = {
    "summer": [12, 1, 2],
    "autumn": [3, 4, 5],
    "winter": [6, 7, 8],
    "spring": [9, 10, 11],
}


# ---------------------------------------------------------------------------
# §7.1: capacity-weighted collapse
# ---------------------------------------------------------------------------

def test_collapse_is_capacity_weighted_not_unweighted():
    factors = pd.DataFrame(
        {
            "cluster": [0, 1],
            "fixed": ["1/1", "1/1"],
            "scalar": [0.8, 1.2],
            "offset": [-1.0, 1.0],
        }
    )
    capacity = pd.Series({0: 90.0, 1: 10.0})
    collapsed = collapse_factors(factors, capacity, "fixed")

    assert len(collapsed) == 1
    row = collapsed.iloc[0]
    assert row["cluster"] == 0
    # Weighted: 0.8*0.9 + 1.2*0.1 = 0.84. Unweighted would be 1.0, so the
    # fixture makes the two modes distinguishable.
    assert row["scalar"] == pytest.approx(0.84)
    assert row["scalar"] != pytest.approx((0.8 + 1.2) / 2)
    assert row["offset"] == pytest.approx(-0.8)


def test_collapse_keeps_slices_separate():
    factors = pd.DataFrame(
        {
            "cluster": [0, 1, 0, 1],
            "season": ["winter", "winter", "summer", "summer"],
            "scalar": [0.5, 0.7, 1.0, 1.0],
            "offset": [0.0, 0.0, 0.0, 0.0],
        }
    )
    capacity = pd.Series({0: 50.0, 1: 50.0})
    collapsed = collapse_factors(factors, capacity, "season")
    assert len(collapsed) == 2
    winter = collapsed[collapsed["season"] == "winter"].iloc[0]
    assert winter["scalar"] == pytest.approx(0.6)


def test_collapse_refuses_unknown_clusters():
    factors = pd.DataFrame(
        {"cluster": [0, 7], "fixed": ["1/1", "1/1"], "scalar": [1.0, 1.0], "offset": [0.0, 0.0]}
    )
    with pytest.raises(ValueError, match=r"cluster\(s\) \[7\]"):
        collapse_factors(factors, pd.Series({0: 1.0}), "fixed")


# ---------------------------------------------------------------------------
# Approved pair set
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source,target", [("AU-NEM", "UK"), ("DK", "AU-NEM")])
def test_transfer_pairs_with_au_on_one_side_pass(source, target):
    check_transfer_pair(source, target)  # must not raise


@pytest.mark.parametrize(
    "source,target", [("DK", "UK"), ("UK", "DE"), ("AU-NEM", "AU-NEM")]
)
def test_transfer_pairs_outside_the_approved_set_fail(source, target):
    with pytest.raises(ValueError):
        check_transfer_pair(source, target)


# ---------------------------------------------------------------------------
# §7.3: the mirrored-hemisphere fixture (normative)
# ---------------------------------------------------------------------------

def test_mirrored_hemisphere_transfer_matches_by_season_name():
    """Source (NH) learned: winter winds are over-blown 2x (scalar 0.5).
    Target (SH) has the same physics: ITS winter (JJA) is over-blown 2x.
    Name matching corrects JJA; month matching corrects DJF. They must
    differ, and only name matching recovers the planted truth."""
    times = pd.date_range("2019-01-01", "2019-12-31", freq="D")
    n = len(times)
    reanalysis_ws = xr.DataArray(
        np.full((n, 2), 10.0),
        coords={"time": times, "turbine": [0, 1]},
        dims=["time", "turbine"],
    )
    # Planted truth in the TARGET region: true winds are 5 m/s in JJA
    # (the target's winter), 10 m/s otherwise.
    months = times.month.to_numpy()
    true_ws = np.where(np.isin(months, [6, 7, 8]), 5.0, 10.0)[:, None] * np.ones((1, 2))

    turb_info = pd.DataFrame(
        {"ID": ["a", "b"], "cluster": [0, 0], "model": ["M", "M"], "capacity": [1.0, 1.0]}
    )
    # Source-trained factors, already collapsed: labelled by season NAME.
    factors = pd.DataFrame(
        {
            "cluster": [0, 0, 0, 0],
            "season": ["winter", "spring", "summer", "autumn"],
            "scalar": [0.5, 1.0, 1.0, 1.0],
            "offset": [0.0, 0.0, 0.0, 0.0],
        }
    )

    # Name matching (correct, design §7.3): target's season definitions.
    cor_name = correct_wind_speed(
        reanalysis_ws, "season", factors, turb_info, seasons=SH_SEASONS
    ).transpose("time", "turbine").to_numpy()
    # Month matching (the bug this design eliminates): source's definitions.
    cor_month = correct_wind_speed(
        reanalysis_ws, "season", factors, turb_info, seasons=NH_SEASONS
    ).transpose("time", "turbine").to_numpy()

    rmse_name = float(np.sqrt(np.mean((cor_name - true_ws) ** 2)))
    rmse_month = float(np.sqrt(np.mean((cor_month - true_ws) ** 2)))

    # 1. The two modes are distinguishable on this fixture.
    assert not np.allclose(cor_name, cor_month)
    # 2. Name matching recovers the planted truth exactly.
    assert rmse_name == pytest.approx(0.0, abs=1e-12)
    # 3. Month matching is not just different, it is WORSE than doing nothing:
    #    it corrects the wrong months and leaves the biased ones alone.
    rmse_uncorrected = float(np.sqrt(np.mean((10.0 - true_ws) ** 2)))
    assert rmse_month > rmse_uncorrected > rmse_name
