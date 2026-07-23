"""Chile (CEN) processing: wind filter, fixed-offset UTC binning, CF, masks.

All fixtures are synthetic; real CEN acquisition is user-executed. The
timestamps are Chilean-standard-labelled on purpose — the fixed UTC-4 shift
(no DST) is part of the contract under test, not an implementation detail.
"""
import numpy as np
import pandas as pd
import pytest

# Module-level so the monkeypatch hits the class the adapter closed over,
# even after another test reloads vwf.config (the NZ-adapter reload gotcha).
from vwf.config import PyVWFPaths
from vwf.datasets.cen_cl import (
    build_cl_metadata,
    monthly_cf_from_generation,
    strip_commissioning_prefix,
    wind_fleet_from_generation,
    wind_rows,
)
from vwf.sources.cen_cl import CENChileSource


def gen_hours(id_central, start, end, gen_mw, *, cap=100.0, tipo="Eólica",
              name=None, prop="OWNER", subtipo="Onshore"):
    """Hourly CEN generation rows on [start, end), Chilean-standard-labelled."""
    times = pd.date_range(start, end, freq="h", inclusive="left")
    return pd.DataFrame({
        "id_central": id_central,
        "central": name or f"PE {id_central}",
        "tipo_tecnologia": tipo,
        "fecha_hora": times.strftime("%Y-%m-%d %H:%M:%S"),
        "gen_real_mw": float(gen_mw),
        "potencia_maxima": float(cap),
        "propietario": prop,
        "subtipo_tecnologia": subtipo,
    })


# --------------------------------------------------------------- wind filter
def test_wind_rows_excludes_hidraulica_not_by_substring():
    gen = pd.concat([
        gen_hours(1, "2024-06-01", "2024-06-02", 40, tipo="Eólica"),
        gen_hours(2, "2024-06-01", "2024-06-02", 90, tipo="Hidráulica"),
        gen_hours(3, "2024-06-01", "2024-06-02", 50, tipo="Solar"),
    ])
    w = wind_rows(gen)
    assert set(w["id_central"]) == {1}  # 'lica' substring must NOT catch Hidráulica


# ------------------------------------------------------------- monthly CF
def test_monthly_cf_is_gen_over_capacity():
    gen = gen_hours(1, "2024-06-01", "2024-07-01", 40, cap=100.0)  # CF 0.4
    wide = monthly_cf_from_generation(gen, 2024, 2024, min_coverage=0.0)
    assert wide.iloc[0]["obs_6"] == pytest.approx(0.4, abs=1e-6)


def test_monthly_bins_are_utc_not_local():
    """Fixed UTC-4: 21:00 Chilean-standard on 30 Jun is 01:00 UTC on 1 Jul, so
    it must land in JULY. Local binning would put it in June — distinguishable
    on this single-hour fixture (mirrors the AU/BR/NZ market-time tests)."""
    gen = gen_hours(1, "2024-06-30 21:00", "2024-06-30 22:00", 100, cap=100.0)
    wide = monthly_cf_from_generation(gen, 2024, 2024, min_coverage=0.0)
    row = wide.iloc[0]
    assert np.isnan(row["obs_6"])            # nothing lands in June UTC
    assert row["obs_7"] == pytest.approx(1.0)  # the hour is July UTC


def test_no_dst_fixed_offset_all_days_24h():
    """Every day is a clean 24 h (CEN applies no DST). A whole September at
    constant CF returns that CF with no spring-forward gap."""
    gen = gen_hours(1, "2024-09-01", "2024-10-01", 30, cap=100.0)
    wide = monthly_cf_from_generation(gen, 2024, 2024, min_coverage=0.9)
    assert wide.iloc[0]["obs_9"] == pytest.approx(0.3, abs=1e-6)


def test_low_coverage_month_is_nan():
    gen = gen_hours(1, "2024-06-01", "2024-06-10", 40, cap=100.0)  # ~1/3 of June
    strict = monthly_cf_from_generation(gen, 2024, 2024)
    loose = monthly_cf_from_generation(gen, 2024, 2024, min_coverage=0.0)
    assert np.isnan(strict.iloc[0]["obs_6"])
    assert not np.isnan(loose.iloc[0]["obs_6"])


def test_zero_capacity_rows_dropped():
    gen = gen_hours(1, "2024-06-01", "2024-07-01", 40, cap=0.0)
    wide = monthly_cf_from_generation(gen, 2024, 2024, min_coverage=0.0)
    assert len(wide) == 0 or np.isnan(wide.iloc[0]["obs_6"])


# --------------------------------------------------------------- fleet
def test_fleet_capacity_is_max_potencia_maxima():
    gen = pd.concat([
        gen_hours(1, "2024-06-01", "2024-06-02", 40, cap=100.0, name="PE UNO"),
        gen_hours(1, "2024-08-01", "2024-08-02", 50, cap=100.0, name="PE UNO"),
    ])
    fleet = wind_fleet_from_generation(gen)
    assert fleet.iloc[0]["capacity_mw"] == pytest.approx(100.0)
    assert fleet.iloc[0]["site_name"] == "PE UNO"


# ------------------------------------------------- commissioning strip
def test_strip_commissioning_prefix_only_leads():
    wide = pd.DataFrame({
        "ID": ["1"], "year": [2024],
        **{f"obs_{m}": [v] for m, v in enumerate(
            [0.0, 0.001, 0.30, 0.28, 0.005, 0.31, 0.29, 0.30, 0.28, 0.31, 0.30, 0.29], 1)},
    })
    out = strip_commissioning_prefix(wide, threshold=0.02)
    r = out.iloc[0]
    assert np.isnan(r["obs_1"]) and np.isnan(r["obs_2"])  # leading near-zeros stripped
    assert r["obs_3"] == pytest.approx(0.30)              # first operational month kept
    assert r["obs_5"] == pytest.approx(0.005)             # a LATER low month is KEPT (outage)


# --------------------------------------------------------------- metadata
def _fleet():
    return pd.DataFrame({
        "ID": ["1", "2"], "site_name": ["PE UNO", "PE DOS"],
        "propietario": ["A", "B"], "capacity_mw": [100.0, 50.0],
        "subtipo": ["Onshore", "Onshore"],
    })


def test_build_metadata_joins_coords_and_converts_kw():
    coords = pd.DataFrame({"ID": ["1", "2"], "lon": [-70.0, -71.0], "lat": [-24.0, -38.0]})
    md = build_cl_metadata(_fleet(), coords, height=100.0, model="M")
    assert set(md["ID"]) == {"1", "2"}
    assert md.loc[md.ID == "1", "capacity"].iloc[0] == pytest.approx(100_000.0)
    assert (md["type"] == "onshore").all()


def test_build_metadata_fails_loudly_on_missing_coord():
    coords = pd.DataFrame({"ID": ["1"], "lon": [-70.0], "lat": [-24.0]})  # 2 missing
    with pytest.raises(ValueError, match="no coordinate"):
        build_cl_metadata(_fleet(), coords, height=100.0, model="M")


def test_build_metadata_exclude_allows_missing_coord():
    coords = pd.DataFrame({"ID": ["1"], "lon": [-70.0], "lat": [-24.0]})
    md = build_cl_metadata(_fleet(), coords, height=100.0, model="M", exclude=("2",))
    assert set(md["ID"]) == {"1"}  # excluded plant dropped, no error


# --------------------------------------------------------------- adapter
def test_source_resolves_and_loads(monkeypatch, tmp_path):
    from vwf.sources import resolve
    cl = tmp_path / "CL"; cl.mkdir()
    pd.DataFrame({
        "ID": ["1"], "lon": [-70.0], "lat": [-24.0], "height": [100.0],
        "capacity": [100_000.0], "model": ["2019COE_Market_Average_2.6MW_121"],
        "type": ["onshore"],
    }).to_csv(cl / "cl_md.csv", index=False)
    pd.DataFrame({"ID": ["1", "1"], "year": [2021, 2024],
                  **{f"obs_{m}": [0.3, 0.35] for m in range(1, 13)}}).to_csv(
        cl / "cl_obs.csv", index=False)

    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path)
    src = resolve("CL", "turbine")
    assert isinstance(src, CENChileSource)
    assert src.load_metadata().iloc[0]["ID"] == "1"
    obs = src.load_observations(2021, 2023)
    assert obs["year"].tolist() == [2021]  # 2024 held out of the train window


def test_missing_files_raise_with_runbook_pointer(monkeypatch, tmp_path):
    (tmp_path / "CL").mkdir()
    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path)
    src = CENChileSource("CL")
    with pytest.raises(FileNotFoundError, match="RUNBOOK_CL"):
        src.load_metadata()
    with pytest.raises(FileNotFoundError, match="RUNBOOK_CL"):
        src.load_observations(2021, 2023)
