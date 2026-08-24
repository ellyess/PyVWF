"""ONS-BR source: FC->monthly CF, curtailment mask, registry wiring.

All fixtures are synthetic; real ONS acquisition is Phase 2.
"""
import numpy as np
import pandas as pd
import pytest

from vwf.config import PyVWFPaths
from vwf.sources import available_sources, get_source, resolve
from vwf.sources.ons_br import ONSBrazilSource, fc_to_monthly_cf


def fc_hours(id_ons, start, end, cf):
    times = pd.date_range(start, end, freq="h", inclusive="left")
    return pd.DataFrame({
        "id_ons": id_ons,
        "nom_tipousina": "Eólica",
        "din_instante": times,
        "val_fatorcapacidade": float(cf),
    })


METADATA = pd.DataFrame({
    "ID": ["CJU_A"],
    "lon": [-40.0],
    "lat": [-5.0],
    "height": [100.0],
    "capacity": [100_000.0],
    "model": ["M"],
    "type": ["onshore"],
})


def test_fc_to_monthly_cf_basic():
    fc = fc_hours("CJU_A", "2023-03-01", "2023-04-01", 0.42)
    wide = fc_to_monthly_cf(fc, 2023, 2023)
    assert list(wide.columns) == ["ID", "year"] + [f"obs_{m}" for m in range(1, 13)]
    assert wide.iloc[0]["obs_3"] == pytest.approx(0.42, abs=1e-6)


def test_curtailment_mask_flows_through():
    """Must-distinguish: with the mask the curtailed month is NaN; without it,
    the same month carries the (curtailment-contaminated) value."""
    fc = fc_hours("CJU_A", "2023-05-01", "2023-06-01", 0.30)
    unmasked = fc_to_monthly_cf(fc, 2023, 2023)
    assert unmasked.iloc[0]["obs_5"] == pytest.approx(0.30, abs=1e-6)

    mask = pd.DataFrame({"ID": ["CJU_A"], "year": [2023], "month": [5]})
    masked = fc_to_monthly_cf(fc, 2023, 2023, curtailment_mask=mask)
    assert np.isnan(masked.iloc[0]["obs_5"])


def test_registry_resolution():
    assert "ons-br" in available_sources()
    src = get_source("ons-br", "BR")
    assert isinstance(src, ONSBrazilSource)
    assert isinstance(resolve("BR", "turbine"), ONSBrazilSource)
    assert isinstance(resolve("BRA", "turbine"), ONSBrazilSource)
    with pytest.raises(ValueError, match="supports"):
        ONSBrazilSource("NZ")


def test_missing_files_fail_with_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path)
    src = ONSBrazilSource()
    with pytest.raises(FileNotFoundError, match="Phase 2"):
        src.load_metadata()
    with pytest.raises(FileNotFoundError, match="Phase 2"):
        src.load_observations(2023, 2023)


def test_source_end_to_end_from_files(tmp_path, monkeypatch):
    data_dir = tmp_path / "BR"
    data_dir.mkdir(parents=True)
    METADATA.to_csv(data_dir / "br_md.csv", index=False)
    fc_hours("CJU_A", "2023-01-01", "2024-01-01", 0.35).to_csv(
        data_dir / "br_fc.csv", index=False
    )
    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path)

    obs = ONSBrazilSource().load_observations(2023, 2023)
    assert list(obs.columns) == ["ID", "year"] + [f"obs_{m}" for m in range(1, 13)]
    assert obs.iloc[0]["obs_6"] == pytest.approx(0.35, abs=1e-6)


def test_curtailment_mask_file_flows_through_source(tmp_path, monkeypatch):
    """The optional mask file on disk is picked up and applied at load."""
    data_dir = tmp_path / "BR"
    data_dir.mkdir(parents=True)
    METADATA.to_csv(data_dir / "br_md.csv", index=False)
    fc_hours("CJU_A", "2023-01-01", "2024-01-01", 0.35).to_csv(
        data_dir / "br_fc.csv", index=False
    )
    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path)

    unmasked = ONSBrazilSource().load_observations(2023, 2023)
    assert unmasked.iloc[0]["obs_4"] == pytest.approx(0.35, abs=1e-6)

    pd.DataFrame({"ID": ["CJU_A"], "year": [2023], "month": [4]}).to_csv(
        data_dir / "br_curtailment_mask.csv", index=False)
    masked = ONSBrazilSource().load_observations(2023, 2023)
    assert np.isnan(masked.iloc[0]["obs_4"])
    assert masked.iloc[0]["obs_5"] == pytest.approx(unmasked.iloc[0]["obs_5"])  # only Apr


def test_default_train_years_used_when_unspecified(tmp_path, monkeypatch):
    data_dir = tmp_path / "BR"
    data_dir.mkdir(parents=True)
    METADATA.to_csv(data_dir / "br_md.csv", index=False)
    lo, hi = ONSBrazilSource().default_train_years
    inside = fc_hours("CJU_A", f"{lo}-01-01", f"{lo}-02-01", 0.3)
    outside = fc_hours("CJU_A", f"{hi + 5}-01-01", f"{hi + 5}-02-01", 0.3)
    pd.concat([inside, outside], ignore_index=True).to_csv(
        data_dir / "br_fc.csv", index=False)
    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path)

    obs = ONSBrazilSource().load_observations()
    assert obs["year"].min() >= lo and obs["year"].max() <= hi
