"""New Zealand (EMI) processing: trading-period mapping, monthly CF, masks.

All fixtures are synthetic; real EMI acquisition is user-executed (Phase 2).
The trading-period timestamps are NZ-civil-time-labelled on purpose: the
DST-aware UTC conversion is part of the contract under test, not an
implementation detail. NZ DST anchors used (from the zone database, not
hardcoded in the code under test): 2023-04-02 fall-back (50 periods),
2023-09-24 spring-forward (46 periods).
"""
import numpy as np
import pandas as pd
import pytest

# Module-level on purpose: this binds the same PyVWFPaths class object the
# adapter closed over at import, so the monkeypatch hits the right class even
# after test_config.py reloads vwf.config (which mints a new class object).
from vwf.config import PyVWFPaths
from vwf.datasets.emi_nz import (
    below_final_build_mask,
    capacity_history_from_register,
    day_start_utc,
    expected_trading_periods,
    half_hourly_from_generation_md,
    monthly_cf,
    trading_period_start_utc,
)
from vwf.sources.emi_nz import EMINewZealandSource, apply_month_mask


# ---------------------------------------------------------------------------
# Trading-period -> UTC mapping
# ---------------------------------------------------------------------------

def test_day_start_utc_winter_and_summer_offsets():
    # NZST (winter) is UTC+12; NZDT (summer) is UTC+13.
    starts = day_start_utc(pd.Series(["2023-06-15", "2023-01-15"]))
    assert starts.iloc[0] == pd.Timestamp("2023-06-14 12:00")
    assert starts.iloc[1] == pd.Timestamp("2023-01-14 11:00")


def test_expected_trading_periods_dst_days():
    n = expected_trading_periods(
        pd.Series(["2023-06-15", "2023-04-02", "2023-09-24"])
    )
    assert n.tolist() == [48, 50, 46]


def test_trading_periods_are_contiguous_across_fall_back():
    """TP50 of the fall-back day must end exactly where the next day's TP1
    starts: sequential elapsed-time mapping, no gap and no double-count. A
    local-clock mapping (midnight + (n-1)*30min of wall time) would fail here."""
    end_of_long_day = trading_period_start_utc(
        pd.Series(["2023-04-02"]), pd.Series([50])
    ).iloc[0] + pd.Timedelta(minutes=30)
    next_day_tp1 = trading_period_start_utc(
        pd.Series(["2023-04-03"]), pd.Series([1])
    ).iloc[0]
    assert end_of_long_day == next_day_tp1


def test_tp_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        trading_period_start_utc(pd.Series(["2023-06-15"]), pd.Series([51]))


# ---------------------------------------------------------------------------
# Generation_MD melt
# ---------------------------------------------------------------------------

def gen_md_day(plant, date, kwh, *, n_periods=48):
    """One wide Generation_MD row: constant kWh in each existing period."""
    row = {"Gen_Code": plant, "Trading_Date": date}
    for tp in range(1, 51):
        row[f"TP{tp}"] = float(kwh) if tp <= n_periods else np.nan
    return pd.DataFrame([row])


def test_melt_produces_one_row_per_period():
    gen = gen_md_day("wf_a", "2023-06-15", 500.0)
    hh = half_hourly_from_generation_md(gen, id_col="Gen_Code")
    assert len(hh) == 48
    assert hh["timestamp"].min() == pd.Timestamp("2023-06-14 12:00")
    assert hh["timestamp"].max() == pd.Timestamp("2023-06-15 11:30")
    assert (hh["kwh"] == 500.0).all()


def test_melt_rejects_generation_beyond_day_length():
    """A value in TP47 on the 46-period spring-forward day means the TP-time
    mapping assumption is broken; it must raise, not silently mis-place it."""
    gen = gen_md_day("wf_a", "2023-09-24", 500.0, n_periods=47)
    with pytest.raises(ValueError, match="beyond the day's length"):
        half_hourly_from_generation_md(gen, id_col="Gen_Code")


def test_melt_accepts_full_fall_back_day():
    gen = gen_md_day("wf_a", "2023-04-02", 500.0, n_periods=50)
    hh = half_hourly_from_generation_md(gen, id_col="Gen_Code")
    assert len(hh) == 50


# ---------------------------------------------------------------------------
# Capacity history and monthly CF
# ---------------------------------------------------------------------------

def _register_two_stage():
    """Two units of one farm: 50 MW from 2019, +50 MW from 1 July 2020."""
    return pd.DataFrame(
        {
            "UnitCode": ["u1", "u2"],
            "NameplateMegawatts": [50.0, 50.0],
            "EffectiveFrom": ["2019-01-01", "2020-07-01"],
            "EffectiveTo": [np.nan, np.nan],
        }
    )


def _history():
    return capacity_history_from_register(
        _register_two_stage(),
        pd.Series({"u1": "wf_a", "u2": "wf_a"}),
        key_col="UnitCode",
        capacity_col="NameplateMegawatts",
        start_col="EffectiveFrom",
        end_col="EffectiveTo",
    )


def test_capacity_history_steps_at_second_unit():
    hist = _history()
    assert hist["capacity"].tolist() == [50_000.0, 100_000.0]
    assert hist["effective_from"].tolist() == [
        pd.Timestamp("2019-01-01"),
        pd.Timestamp("2020-07-01"),
    ]


def test_monthly_cf_uses_then_current_capacity():
    """Half-build months must divide by the then-current 50 MW, not the final
    100 MW: a static-nameplate denominator would halve the CF and fail here."""
    times = pd.date_range("2020-03-01", "2020-04-01", freq="30min",
                          inclusive="left")
    # CF 0.4 against 50 MW: kwh = 50_000 kW * 0.5 h * 0.4
    hh = pd.DataFrame({"ID": "wf_a", "timestamp": times,
                       "kwh": 50_000 * 0.5 * 0.4})
    wide = monthly_cf(hh, _history(), 2020, 2020, min_coverage=0.0)
    assert wide.iloc[0]["obs_3"] == pytest.approx(0.4, abs=1e-9)


def test_monthly_bins_are_utc_not_nz_local():
    """Energy in TP1-24 of trading date 1 July (local) is 30 June 12:00-24:00
    UTC and must land in JUNE. Local binning would put it in July,
    distinguishable on this fixture (mirrors the AU/BR market-time tests)."""
    gen = gen_md_day("wf_a", "2023-07-01", 1000.0)
    hh = half_hourly_from_generation_md(gen, id_col="Gen_Code")
    hh = hh.rename(columns={"Gen_Code": "ID"})
    hist = pd.DataFrame(
        {"ID": ["wf_a"], "effective_from": [pd.Timestamp("2019-01-01")],
         "capacity": [50_000.0]}
    )
    wide = monthly_cf(hh, hist, 2023, 2023, min_coverage=0.0)
    row = wide[wide["year"] == 2023].iloc[0]
    assert not np.isnan(row["obs_6"])  # first 24 periods are June UTC
    assert not np.isnan(row["obs_7"])  # rest are July UTC


def test_low_coverage_month_is_nan():
    times = pd.date_range("2020-03-01", "2020-03-10", freq="30min",
                          inclusive="left")
    hh = pd.DataFrame({"ID": "wf_a", "timestamp": times, "kwh": 1000.0})
    hist = pd.DataFrame(
        {"ID": ["wf_a"], "effective_from": [pd.Timestamp("2019-01-01")],
         "capacity": [50_000.0]}
    )
    strict = monthly_cf(hh, hist, 2020, 2020)
    loose = monthly_cf(hh, hist, 2020, 2020, min_coverage=0.0)
    assert np.isnan(strict.iloc[0]["obs_3"])
    assert not np.isnan(loose.iloc[0]["obs_3"])


def test_pre_registration_generation_is_dropped():
    times = pd.date_range("2018-06-01", "2018-06-02", freq="30min",
                          inclusive="left")
    hh = pd.DataFrame({"ID": "wf_a", "timestamp": times, "kwh": 1000.0})
    wide = monthly_cf(hh, _history(), 2018, 2018, min_coverage=0.0)
    assert len(wide) == 0 or np.isnan(wide.iloc[0]["obs_6"])


def test_below_final_build_mask_lists_ramp_months_only():
    mask = below_final_build_mask(_history(), 2020, 2020)
    masked = set(zip(mask["year"], mask["month"]))
    assert (2020, 6) in masked      # still 50 of 100 MW
    assert (2020, 7) not in masked  # full build from 1 July
    assert (2020, 12) not in masked


# ---------------------------------------------------------------------------
# Source finalisation and registry
# ---------------------------------------------------------------------------

def _wide_obs():
    return pd.DataFrame(
        {"ID": ["wf_a", "wf_a"], "year": [2020, 2021],
         **{f"obs_{m}": [0.4, 0.5] for m in range(1, 13)}}
    )


def test_apply_month_mask_nans_listed_cells_only():
    mask = pd.DataFrame({"ID": ["wf_a"], "year": [2020], "month": [6]})
    out = apply_month_mask(_wide_obs(), mask)
    assert np.isnan(out.loc[out["year"] == 2020, "obs_6"].iloc[0])
    assert out.loc[out["year"] == 2020, "obs_5"].iloc[0] == pytest.approx(0.4)
    assert out.loc[out["year"] == 2021, "obs_6"].iloc[0] == pytest.approx(0.5)


def test_source_resolves_for_nz(monkeypatch, tmp_path):
    from vwf.sources import resolve

    nz_dir = tmp_path / "NZ"
    nz_dir.mkdir()
    md = pd.DataFrame(
        {"ID": ["wf_a"], "lon": [175.0], "lat": [-40.0], "height": [80.0],
         "capacity": [100_000.0], "model": ["Vestas_V90_3000_90"],
         "type": ["onshore"]}
    )
    md.to_csv(nz_dir / "nz_md.csv", index=False)
    _wide_obs().to_csv(nz_dir / "nz_obs.csv", index=False)

    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path)
    src = resolve("NZ", "turbine")
    assert isinstance(src, EMINewZealandSource)

    meta = src.load_metadata()
    assert meta.iloc[0]["ID"] == "wf_a"

    obs = src.load_observations(2020, 2020)
    assert obs["year"].tolist() == [2020]
    assert obs.iloc[0]["obs_1"] == pytest.approx(0.4)


def test_source_applies_build_mask_when_present(monkeypatch, tmp_path):
    nz_dir = tmp_path / "NZ"
    nz_dir.mkdir()
    _wide_obs().to_csv(nz_dir / "nz_obs.csv", index=False)
    pd.DataFrame({"ID": ["wf_a"], "year": [2020], "month": [2]}).to_csv(
        nz_dir / "nz_build_mask.csv", index=False
    )

    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path)
    obs = EMINewZealandSource("NZ").load_observations(2020, 2021)
    assert np.isnan(obs.loc[obs["year"] == 2020, "obs_2"].iloc[0])
    assert obs.loc[obs["year"] == 2021, "obs_2"].iloc[0] == pytest.approx(0.5)


def test_missing_files_raise_with_runbook_pointer(monkeypatch, tmp_path):
    (tmp_path / "NZ").mkdir()
    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path)
    src = EMINewZealandSource("NZ")
    with pytest.raises(FileNotFoundError, match="runbooks/NZ"):
        src.load_metadata()
    with pytest.raises(FileNotFoundError, match="runbooks/NZ"):
        src.load_observations(2020, 2020)
