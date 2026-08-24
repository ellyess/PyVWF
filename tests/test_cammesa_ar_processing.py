"""Argentina (CAMMESA) processing: GWh->CF, capacity join guards, masks.

All fixtures are synthetic; real CAMMESA acquisition is user-executed. The key
things pinned: monthly GWh becomes CF against a JOINED capacity (CAMMESA has
none), a too-small capacity is caught by the median-CF guard, and a plant
without coordinate OR capacity fails loudly.
"""
import numpy as np
import pandas as pd
import pytest

from vwf.config import PyVWFPaths
from vwf.datasets.cammesa_ar import (
    build_ar_metadata,
    capacity_suspect_ids,
    monthly_cf_from_gwh,
    strip_commissioning_prefix,
)
from vwf.sources.cammesa_ar import CAMMESAArgentinaSource


def _gwh(rows):
    return pd.DataFrame(rows, columns=["ID", "year", "month", "gwh"])


# --------------------------------------------------------------- GWh -> CF
def test_cf_is_energy_over_capacity_times_hours():
    # 100 MW plant at exactly CF 0.5 in a 30-day month: MWh = 100*0.5*30*24
    gwh = _gwh([["P", 2024, 4, 100 * 0.5 * 30 * 24 / 1000.0]])  # April = 30 days
    cap = pd.DataFrame({"ID": ["P"], "capacity_mw": [100.0]})
    wide = monthly_cf_from_gwh(gwh, cap, 2024, 2024)
    assert wide.iloc[0]["obs_4"] == pytest.approx(0.5, abs=1e-9)


def test_february_uses_28_days():
    gwh = _gwh([["P", 2023, 2, 100 * 0.4 * 28 * 24 / 1000.0]])  # 2023 Feb = 28 d
    cap = pd.DataFrame({"ID": ["P"], "capacity_mw": [100.0]})
    wide = monthly_cf_from_gwh(gwh, cap, 2023, 2023)
    assert wide.iloc[0]["obs_2"] == pytest.approx(0.4, abs=1e-9)


def test_plant_without_capacity_is_dropped():
    gwh = _gwh([["P", 2024, 1, 50.0]])
    cap = pd.DataFrame({"ID": ["OTHER"], "capacity_mw": [100.0]})
    wide = monthly_cf_from_gwh(gwh, cap, 2024, 2024)
    assert len(wide) == 0 or wide["ID"].tolist() == []


# --------------------------------------------------------- capacity guard
def test_capacity_suspect_flags_too_small_denominator():
    """A too-small capacity inflates CF above what wind delivers. A plant
    whose median CF is implausibly high is flagged; a normal one is not: a
    check that distinguishes the two, not one that always passes."""
    oc = {f"obs_{m}": [0.40, 1.10] for m in range(1, 13)}  # P good, Q inflated
    wide = pd.DataFrame({"ID": ["P", "Q"], "year": [2024, 2024], **oc})
    assert capacity_suspect_ids(wide) == ["Q"]


# ----------------------------------------------- commissioning strip
def test_strip_commissioning_prefix_only_leads():
    wide = pd.DataFrame({
        "ID": ["1"], "year": [2024],
        **{f"obs_{m}": [v] for m, v in enumerate(
            [0.0, 0.30, 0.28, 0.005, 0.31, 0.29, 0.30, 0.28, 0.31, 0.30, 0.29, 0.33], 1)},
    })
    out = strip_commissioning_prefix(wide, threshold=0.02)
    r = out.iloc[0]
    assert np.isnan(r["obs_1"])              # leading zero stripped
    assert r["obs_2"] == pytest.approx(0.30)  # first operational month kept
    assert r["obs_4"] == pytest.approx(0.005)  # later low month KEPT (not a lead)


# --------------------------------------------------------------- metadata
def _fleet():
    return pd.DataFrame({
        "ID": ["A", "B"], "site_name": ["PE A", "PE B"],
        "region": ["PATAGONIA", "PAMPA"], "provincia": ["CHUBUT", "BUENOS AIRES"],
    })


def test_build_metadata_joins_coords_and_capacity():
    join = pd.DataFrame({"ID": ["A", "B"], "lon": [-67.0, -62.0],
                         "lat": [-45.0, -38.0], "capacity_mw": [99.0, 50.0]})
    md = build_ar_metadata(_fleet(), join, height=100.0, model="M")
    assert set(md["ID"]) == {"A", "B"}
    assert md.loc[md.ID == "A", "capacity"].iloc[0] == pytest.approx(99_000.0)


def test_build_metadata_fails_loudly_on_missing_capacity():
    # B has a coordinate but no capacity -> no CF denominator -> must raise.
    join = pd.DataFrame({"ID": ["A", "B"], "lon": [-67.0, -62.0],
                         "lat": [-45.0, -38.0], "capacity_mw": [99.0, np.nan]})
    with pytest.raises(ValueError, match="coordinate or capacity"):
        build_ar_metadata(_fleet(), join, height=100.0, model="M")


def test_build_metadata_exclude_allows_missing():
    join = pd.DataFrame({"ID": ["A"], "lon": [-67.0], "lat": [-45.0],
                         "capacity_mw": [99.0]})
    md = build_ar_metadata(_fleet(), join, height=100.0, model="M", exclude=("B",))
    assert set(md["ID"]) == {"A"}


# --------------------------------------------------------------- adapter
def test_source_resolves_and_loads(monkeypatch, tmp_path):
    from vwf.sources import resolve
    ar = tmp_path / "AR"
    ar.mkdir()
    pd.DataFrame({
        "ID": ["PE A"], "lon": [-67.0], "lat": [-45.0], "height": [100.0],
        "capacity": [99_000.0], "model": ["2019COE_Market_Average_2.6MW_121"],
        "type": ["onshore"],
    }).to_csv(ar / "ar_md.csv", index=False)
    pd.DataFrame({"ID": ["PE A", "PE A"], "year": [2021, 2024],
                  **{f"obs_{m}": [0.4, 0.45] for m in range(1, 13)}}).to_csv(
        ar / "ar_obs.csv", index=False)

    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path)
    src = resolve("AR", "turbine")
    assert isinstance(src, CAMMESAArgentinaSource)
    assert src.load_metadata().iloc[0]["ID"] == "PE A"
    obs = src.load_observations(2021, 2023)
    assert obs["year"].tolist() == [2021]


def test_missing_files_raise_with_runbook_pointer(monkeypatch, tmp_path):
    (tmp_path / "AR").mkdir()
    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path)
    src = CAMMESAArgentinaSource("AR")
    with pytest.raises(FileNotFoundError, match="runbooks/ar"):
        src.load_metadata()
    with pytest.raises(FileNotFoundError, match="runbooks/ar"):
        src.load_observations(2021, 2023)
