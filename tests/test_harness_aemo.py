"""AEMO NEM source: AEST→UTC binning, CF arithmetic, masks (design §2).

All fixtures are synthetic; real AEMO acquisition is Phase 2. The fixtures
carry AEST-labelled timestamps on purpose: the UTC conversion is part of
the contract under test, not an implementation detail.
"""
from calendar import monthrange

import numpy as np
import pandas as pd
import pytest

from vwf.config import PyVWFPaths
from vwf.sources import available_sources, get_source, resolve
from vwf.sources.aemo import AEMONemSource, aest_to_utc, scada_to_monthly_cf


def five_minute_scada(duid, start_aest, end_aest, mw):
    """Constant-output SCADA rows on the 5-minute dispatch grid, AEST-labelled."""
    times = pd.date_range(start_aest, end_aest, freq="5min", inclusive="left")
    return pd.DataFrame({"timestamp": times, "ID": duid, "mw": float(mw)})


METADATA = pd.DataFrame(
    {
        "ID": ["FARM1"],
        "lon": [148.0],
        "lat": [-35.0],
        "height": [100.0],
        "capacity": [100_000.0],  # kW == 100 MW, per the source contract
        "model": ["M"],
        "type": ["onshore"],
    }
)


def test_aest_to_utc_is_a_fixed_minus_10h_shift():
    ts = pd.Series(pd.to_datetime(["2019-07-01 10:00", "2019-01-01 05:00"]))
    utc = aest_to_utc(ts)
    assert list(utc) == list(pd.to_datetime(["2019-07-01 00:00", "2018-12-31 19:00"]))
    with pytest.raises(ValueError, match="naive market time"):
        aest_to_utc(pd.Series(pd.to_datetime(["2019-07-01 10:00+10:00"])))


def test_constant_half_power_gives_cf_half():
    """50 MW all month from a 100 MW farm -> CF 0.5, whatever the timezone."""
    scada = five_minute_scada("FARM1", "2019-03-01", "2019-04-01", mw=50.0)
    wide = scada_to_monthly_cf(scada, METADATA, 2019, 2019, min_coverage=0.0)
    row = wide[wide["year"] == 2019].iloc[0]
    # March in AEST leaks its first 10 h into February UTC, so March coverage
    # is not exactly full; CF must still be ~0.5 where data exists.
    assert row["obs_3"] == pytest.approx(0.5, abs=0.01)


def test_monthly_bins_are_utc_not_market_time():
    """The decided convention (design §2): generation at 00:00–09:55 AEST on
    1 July is 30 June UTC and must land in JUNE's bin. Month matching in
    market time would put it in July, so the two conventions are
    distinguishable on this fixture."""
    scada = five_minute_scada(
        "FARM1", "2019-07-01 00:00", "2019-07-01 10:00", mw=100.0
    )
    wide = scada_to_monthly_cf(scada, METADATA, 2019, 2019, min_coverage=0.0)
    row = wide.iloc[0]

    june_hours = monthrange(2019, 6)[1] * 24.0
    expected_june_cf = (100.0 * 10.0) / (100.0 * june_hours)  # 10 h at full power
    assert row["obs_6"] == pytest.approx(expected_june_cf, rel=1e-6)
    assert np.isnan(row["obs_7"])  # nothing lands in July UTC


def test_low_coverage_months_are_nan_not_biased_low():
    # Half of March present: at the default 0.9 coverage floor this must be
    # NaN; with the floor disabled it computes (and would read ~0.25).
    scada = five_minute_scada("FARM1", "2019-03-01", "2019-03-16 12:00", mw=50.0)
    strict = scada_to_monthly_cf(scada, METADATA, 2019, 2019)
    loose = scada_to_monthly_cf(scada, METADATA, 2019, 2019, min_coverage=0.0)
    assert np.isnan(strict.iloc[0]["obs_3"])
    assert loose.iloc[0]["obs_3"] == pytest.approx(0.25, abs=0.01)


def test_commissioning_mask():
    """Months starting before commissioning are NaN; the first full month
    after commissioning is the first valid one."""
    meta = METADATA.assign(commissioning_date="2019-03-15")
    scada = five_minute_scada("FARM1", "2019-02-01", "2019-06-01", mw=50.0)
    wide = scada_to_monthly_cf(scada, meta, 2019, 2019, min_coverage=0.0)
    row = wide.iloc[0]
    assert np.isnan(row["obs_2"])  # pre-commissioning
    assert np.isnan(row["obs_3"])  # month containing the commissioning date
    assert row["obs_4"] == pytest.approx(0.5, abs=0.01)


def test_registry_resolution():
    assert "aemo-nem" in available_sources()
    src = get_source("aemo-nem", "AU-NEM")
    assert isinstance(src, AEMONemSource)
    assert isinstance(resolve("AU-NEM", "turbine"), AEMONemSource)
    with pytest.raises(ValueError, match="supports"):
        AEMONemSource("NZ")


def test_missing_files_fail_with_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path)
    src = AEMONemSource()
    with pytest.raises(FileNotFoundError, match="Phase 2"):
        src.load_metadata()
    with pytest.raises(FileNotFoundError, match="Phase 2"):
        src.load_observations(2019, 2019)


def test_source_end_to_end_from_files(tmp_path, monkeypatch):
    """The adapter reads the documented on-disk layout and returns the wide
    obs contract (ID, year, obs_1..obs_12)."""
    data_dir = tmp_path / "AU_NEM"
    data_dir.mkdir(parents=True)
    METADATA.to_csv(data_dir / "au_nem_md.csv", index=False)
    five_minute_scada("FARM1", "2019-01-01", "2020-01-01", mw=25.0).to_csv(
        data_dir / "au_nem_scada.csv", index=False
    )
    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path)

    src = AEMONemSource()
    obs = src.load_observations(2019, 2019)
    assert list(obs.columns) == ["ID", "year"] + [f"obs_{m}" for m in range(1, 13)]
    # Middle months are fully covered: CF = 25/100.
    assert obs.iloc[0]["obs_6"] == pytest.approx(0.25, abs=0.005)
