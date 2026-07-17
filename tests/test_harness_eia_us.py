"""EIA-US source: net-generation CF arithmetic, annual-respondent +
commissioning screens, registry wiring.

All fixtures are synthetic; real EIA acquisition is Phase 2.
"""
import numpy as np
import pandas as pd
import pytest

from vwf.config import PyVWFPaths
from vwf.sources import available_sources, get_source, resolve
from vwf.sources.eia_us import EIAUSSource, netgen_to_monthly_cf

METADATA = pd.DataFrame(
    {
        "ID": ["100"],
        "lon": [-104.2],
        "lat": [41.5],
        "height": [100.0],
        "capacity": [100_000.0],  # kW == 100 MW, per the source contract
        "model": ["M"],
        "type": ["onshore"],
    }
)


def monthly_netgen(plant_id, year, mwh, freq="M"):
    """Twelve monthly rows of constant net generation for one plant-year."""
    return pd.DataFrame(
        {
            "ID": [str(plant_id)] * 12,
            "year": [year] * 12,
            "month": list(range(1, 13)),
            "net_gen_mwh": [float(mwh)] * 12,
            "respondent_frequency": [freq] * 12,
        }
    )


def test_constant_generation_gives_expected_cf():
    """36,000 MWh in June from a 100 MW plant -> CF 0.5 (720 h month)."""
    netgen = monthly_netgen("100", 2020, mwh=0.0)
    netgen.loc[netgen["month"] == 6, "net_gen_mwh"] = 36_000.0
    wide = netgen_to_monthly_cf(netgen, METADATA, 2020, 2020)
    row = wide.iloc[0]
    assert row["obs_6"] == pytest.approx(0.5, rel=1e-6)
    # A different-length month scales by its hours: 31-day January (744 h).
    netgen.loc[netgen["month"] == 1, "net_gen_mwh"] = 37_200.0
    wide = netgen_to_monthly_cf(netgen, METADATA, 2020, 2020)
    assert wide.iloc[0]["obs_1"] == pytest.approx(37_200_000.0 / (744 * 100_000), rel=1e-6)


def test_annual_respondents_dropped_by_default():
    """Must-distinguish: an annual-flagged plant-year is NaN by default and
    computes only when the screen is turned off."""
    netgen = monthly_netgen("100", 2020, mwh=36_000.0, freq="A")
    dropped = netgen_to_monthly_cf(netgen, METADATA, 2020, 2020)
    assert dropped.iloc[0][[f"obs_{m}" for m in range(1, 13)]].isna().all()

    kept = netgen_to_monthly_cf(
        netgen, METADATA, 2020, 2020, drop_annual_respondents=False
    )
    assert kept.iloc[0]["obs_6"] == pytest.approx(0.5, rel=1e-6)


def test_monthly_respondents_are_kept():
    netgen = monthly_netgen("100", 2020, mwh=36_000.0, freq="M")
    wide = netgen_to_monthly_cf(netgen, METADATA, 2020, 2020)
    assert wide.iloc[0]["obs_6"] == pytest.approx(0.5, rel=1e-6)


def test_negative_net_generation_kept_not_clipped():
    """A calm month can be net-negative (parasitic load); the signal is kept."""
    netgen = monthly_netgen("100", 2020, mwh=0.0)
    netgen.loc[netgen["month"] == 2, "net_gen_mwh"] = -720.0  # small negative
    wide = netgen_to_monthly_cf(netgen, METADATA, 2020, 2020)
    assert wide.iloc[0]["obs_2"] < 0


def test_commissioning_mask():
    """Months starting before commissioning are NaN; the first full month
    after commissioning is the first valid one."""
    meta = METADATA.assign(commissioning_date="2020-03-15")
    netgen = monthly_netgen("100", 2020, mwh=36_000.0)
    wide = netgen_to_monthly_cf(netgen, meta, 2020, 2020)
    row = wide.iloc[0]
    assert np.isnan(row["obs_2"])  # pre-commissioning
    assert np.isnan(row["obs_3"])  # month containing the commissioning date
    assert not np.isnan(row["obs_4"])  # first full post-commissioning month


def test_year_window_filter():
    two_years = pd.concat(
        [monthly_netgen("100", 2019, 36_000.0), monthly_netgen("100", 2020, 36_000.0)],
        ignore_index=True,
    )
    wide = netgen_to_monthly_cf(two_years, METADATA, 2020, 2020)
    assert wide["year"].tolist() == [2020]


def test_registry_resolution():
    assert "eia-us" in available_sources()
    src = get_source("eia-us", "US")
    assert isinstance(src, EIAUSSource)
    assert isinstance(resolve("US", "turbine"), EIAUSSource)
    assert isinstance(resolve("USA", "turbine"), EIAUSSource)
    with pytest.raises(ValueError, match="supports"):
        EIAUSSource("NZ")


def test_missing_files_fail_with_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path)
    src = EIAUSSource()
    with pytest.raises(FileNotFoundError, match="Phase 2"):
        src.load_metadata()
    with pytest.raises(FileNotFoundError, match="Phase 2"):
        src.load_observations(2020, 2020)


def test_source_end_to_end_from_files(tmp_path, monkeypatch):
    """The adapter reads the documented on-disk layout and returns the wide
    obs contract (ID, year, obs_1..obs_12)."""
    data_dir = tmp_path / "US"
    data_dir.mkdir(parents=True)
    METADATA.to_csv(data_dir / "us_md.csv", index=False)
    monthly_netgen("100", 2020, mwh=25_000.0).to_csv(
        data_dir / "us_eia923_netgen.csv", index=False
    )
    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path)

    src = EIAUSSource()
    obs = src.load_observations(2020, 2020)
    assert list(obs.columns) == ["ID", "year"] + [f"obs_{m}" for m in range(1, 13)]
    # June: 25,000 MWh / (720 h * 100 MW) = 0.347...
    assert obs.iloc[0]["obs_6"] == pytest.approx(25_000_000.0 / (720 * 100_000), rel=1e-6)


def test_default_train_years_used_when_unspecified(tmp_path, monkeypatch):
    data_dir = tmp_path / "US"
    data_dir.mkdir(parents=True)
    METADATA.to_csv(data_dir / "us_md.csv", index=False)
    # Rows inside and outside the default window; only in-window survives.
    inside = monthly_netgen("100", EIAUSSource().default_train_years[0], 36_000.0)
    outside = monthly_netgen("100", EIAUSSource().default_train_years[1] + 5, 36_000.0)
    pd.concat([inside, outside], ignore_index=True).to_csv(
        data_dir / "us_eia923_netgen.csv", index=False
    )
    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path)

    obs = EIAUSSource().load_observations()
    lo, hi = EIAUSSource().default_train_years
    assert obs["year"].min() >= lo and obs["year"].max() <= hi
