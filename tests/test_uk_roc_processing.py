"""UK (Ofgem ROC + REPD) transforms: banding, cert->MWh, pseudo-replication.

The observations path cannot be verified against a live source (per-station
ROC issuance is a gated Ofgem RER export), so it is pinned here with synthetic
fixtures that reproduce the committed ukobs.csv arithmetic exactly.
"""
import numpy as np
import pandas as pd
import pytest

from vwf.datasets.uk_roc import (
    band_for,
    osgb_to_wgs84,
    pseudo_replicate_metadata,
    pseudo_replicate_observations,
    repd_wind_metadata,
    roc_issuance_to_station_monthly,
)


# --------------------------------------------------------------- banding
def test_banding_onshore_grandfathered():
    assert band_for("Wind Onshore", 2015) == 0.9
    assert band_for("onshore", 2010) == 1.0        # pre-2013 vintage
    assert band_for("Wind Onshore", None) == 0.9   # default


def test_banding_offshore_vintages():
    assert band_for("Wind Offshore", 2013) == 2.0
    assert band_for("Wind Offshore", 2015) == 1.9
    assert band_for("Wind Offshore", 2017) == 1.8
    assert band_for("offshore", None) == 2.0


# --------------------------------------------------------------- reprojection
def test_osgb_reprojects_into_uk_bbox():
    # A central-England grid point maps to sane UK degrees, not the raw metres.
    c = osgb_to_wgs84(pd.Series([400000]), pd.Series([300000]))
    assert -6 < c["lon"].iloc[0] < 2
    assert 50 < c["lat"].iloc[0] < 56


# --------------------------------------------------------------- ROC -> MWh
def test_roc_to_mwh_uses_banding():
    roc = pd.DataFrame({
        "AccreditationNumber": ["R00116SQSC", "R00200RPEN"],   # onshore, offshore
        "OutputPeriod": ["2015-01-01", "2015-01-01"],
        "Certificates": [1538.1, 3800.0],
    })
    sm = roc_issuance_to_station_monthly(
        roc, station_col="AccreditationNumber",
        period_col="OutputPeriod", certs_col="Certificates")
    onshore = sm[sm.ID == "R00116SQSC"].iloc[0]
    offshore = sm[sm.ID == "R00200RPEN"].iloc[0]
    assert onshore["mwh"] == pytest.approx(1538.1 / 0.9)     # SQ -> onshore /0.9
    assert offshore["mwh"] == pytest.approx(3800.0 / 2.0)    # RP -> offshore /2.0


def test_roc_technology_from_accreditation_code():
    # SP = offshore Scotland; RQ = onshore England — decoded from the id.
    roc = pd.DataFrame({
        "AccreditationNumber": ["R00001SPSC", "R00002RQEN"],
        "OutputPeriod": ["2016-06-01", "2016-06-01"],
        "Certificates": [190.0, 90.0],
    })
    sm = roc_issuance_to_station_monthly(
        roc, station_col="AccreditationNumber",
        period_col="OutputPeriod", certs_col="Certificates")
    assert sm[sm.ID == "R00001SPSC"].iloc[0]["mwh"] == pytest.approx(190 / 2.0)  # offshore
    assert sm[sm.ID == "R00002RQEN"].iloc[0]["mwh"] == pytest.approx(90 / 0.9)   # onshore


# ----------------------------------------------- pseudo-replication (obs)
def test_pseudo_replicate_equal_split_reproduces_committed_arithmetic():
    """A 3-turbine station's monthly MWh is split into 3 identical kWh rows —
    the exact committed convention (R00116SQSC-1 Jan 2015 = 569666.67 kWh)."""
    sm = pd.DataFrame({"ID": ["R00116SQSC"], "year": [2015], "month": [1],
                       "mwh": [1709.0]})
    obs = pseudo_replicate_observations(sm, pd.Series({"R00116SQSC": 3}))
    assert len(obs) == 3
    assert set(obs["ID"]) == {"R00116SQSC-1", "R00116SQSC-2", "R00116SQSC-3"}
    assert obs["1"].nunique() == 1                       # identical across rows
    assert obs["1"].iloc[0] == pytest.approx(1709.0 * 1000 / 3)  # 569666.67 kWh


def test_pseudo_replicate_missing_month_is_zero():
    sm = pd.DataFrame({"ID": ["S"], "year": [2015], "month": [3], "mwh": [100.0]})
    obs = pseudo_replicate_observations(sm, pd.Series({"S": 1}))
    assert obs["3"].iloc[0] == pytest.approx(100_000.0)
    assert obs["1"].iloc[0] == 0.0                       # no January data -> 0


# --------------------------------------------------------------- REPD metadata
def _repd_fixture():
    return pd.DataFrame({
        "Ref ID": ["A", "B", "C"],
        "Site Name": ["Alpha", "Beta", "Gamma"],
        "Technology Type": ["Wind Onshore", "Wind Offshore", "Solar Photovoltaics"],
        "Development Status (short)": ["Operational", "Operational", "Operational"],
        "Installed Capacity (MWelec)": [30.0, 100.0, 5.0],
        "Turbine Capacity (MW)": [3.0, 5.0, np.nan],
        "No. of Turbines": [10, 20, np.nan],
        "Height of Turbines (m)": [80.0, np.nan, np.nan],
        "X-coordinate": [400000, 350000, 410000],
        "Y-coordinate": [300000, 800000, 305000],
        "Operational": ["01/06/2015", "15/09/2016", "01/01/2018"],
    })


def test_repd_metadata_keeps_wind_and_fills_defaults():
    md = repd_wind_metadata(_repd_fixture(), height_default=100.0, model="M")
    assert set(md["station"]) == {"A", "B"}              # solar dropped
    a = md[md.station == "A"].iloc[0]
    assert a["type"] == "onshore" and a["capacity"] == pytest.approx(3000.0)
    assert a["height"] == 80.0 and a["height_source"] == "repd-tip-height"
    b = md[md.station == "B"].iloc[0]
    assert b["type"] == "offshore" and b["height"] == 100.0  # REPD blank -> default
    assert b["height_source"] == "default-uniform"
    assert (md["model"] == "M").all()
    assert -6 < a["lon"] < 2 and 50 < a["lat"] < 62         # reprojected


def test_pseudo_replicate_metadata_expands_turbines():
    md = repd_wind_metadata(_repd_fixture(), model="M")
    turb = pseudo_replicate_metadata(md)
    assert (turb["ID"] == "A-1").any() and (turb["ID"] == "A-10").any()
    assert len(turb[turb.ID.str.startswith("A-")]) == 10
