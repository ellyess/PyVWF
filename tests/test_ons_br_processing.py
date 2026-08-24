"""Brazil (ONS) processing: FC reshape, complex metadata, curtailment account.

All fixtures are synthetic; real ONS/ANEEL acquisition is Phase 2. The FC
timestamps are Brasília-labelled on purpose: the UTC conversion is part of the
contract under test, not an implementation detail.
"""
import numpy as np
import pandas as pd
import pytest

from vwf.datasets.ons_br import (
    build_br_metadata,
    commissioning_from_siga,
    constrained_off_account,
    curtailment_mask_months,
    monthly_cf_from_fc,
    wind_complexes_from_fc,
)


def fc_hours(id_ons, start, end, cf, *, tipo="Eólica", cap=100.0, lat=-5.0, lon=-40.0):
    """Hourly FC rows on the [start, end) grid, Brasília-labelled."""
    times = pd.date_range(start, end, freq="h", inclusive="left")
    return pd.DataFrame(
        {
            "id_ons": id_ons,
            "nom_usina_conjunto": f"Conj {id_ons}",
            "nom_tipousina": tipo,
            "din_instante": times,
            "val_fatorcapacidade": float(cf),
            "val_capacidadeinstalada": float(cap),
            "val_latitudesecoletora": lat,
            "val_longitudesecoletora": lon,
            "id_subsistema": "NE",
            "nom_estado": "CEARA",
        }
    )


def test_wind_complexes_filters_and_aggregates():
    fc = pd.concat([
        fc_hours("CJU_A", "2023-03-01", "2023-03-02", 0.4, cap=100.0),
        # capacity ramps: a later chunk at higher capacity -> max wins
        fc_hours("CJU_A", "2023-06-01", "2023-06-02", 0.5, cap=150.0),
        fc_hours("SOLAR_X", "2023-03-01", "2023-03-02", 0.9, tipo="Fotovoltaica"),
    ])
    comp = wind_complexes_from_fc(fc)
    assert set(comp["ID"]) == {"CJU_A"}  # solar excluded
    row = comp.iloc[0]
    assert row["capacity_mw"] == pytest.approx(150.0)  # max over the window
    assert row["lat"] == pytest.approx(-5.0)
    assert row["state"] == "CEARA"


def test_monthly_cf_is_mean_of_hourly():
    fc = fc_hours("CJU_A", "2023-03-01", "2023-04-01", 0.4)
    wide = monthly_cf_from_fc(fc, 2023, 2023, min_coverage=0.0)
    row = wide[wide["year"] == 2023].iloc[0]
    # March in Brasília leaks its last 3 h into April UTC, so March mean is over
    # slightly fewer hours but the constant CF is still 0.4.
    assert row["obs_3"] == pytest.approx(0.4, abs=1e-6)


def test_monthly_bins_are_utc_not_brasilia():
    """A single hour at 23:00 Brasília on 30 June is 01 July 02:00 UTC and must
    land in JULY. Binning in local time would put it in June, distinguishable
    on this fixture (mirrors the AU market-time test)."""
    fc = fc_hours("CJU_A", "2023-06-30 23:00", "2023-07-01 00:00", 1.0)
    wide = monthly_cf_from_fc(fc, 2023, 2023, min_coverage=0.0)
    row = wide.iloc[0]
    assert np.isnan(row["obs_6"])           # nothing lands in June UTC
    assert row["obs_7"] == pytest.approx(1.0)  # the hour is July UTC


def test_low_coverage_month_is_nan():
    # Half of March present -> below the 0.9 floor -> NaN; floor off -> computes.
    fc = fc_hours("CJU_A", "2023-03-01", "2023-03-16", 0.4)
    strict = monthly_cf_from_fc(fc, 2023, 2023)
    loose = monthly_cf_from_fc(fc, 2023, 2023, min_coverage=0.0)
    assert np.isnan(strict.iloc[0]["obs_3"])
    assert loose.iloc[0]["obs_3"] == pytest.approx(0.4, abs=1e-6)


def test_constrained_off_account_separates_curtailment():
    # 90 MWmed delivered + 10 curtailed every hour -> fraction 0.1.
    times = pd.date_range("2023-05-01", "2023-06-01", freq="h", inclusive="left")
    coff = pd.DataFrame({
        "id_ons": "CJU_A",
        "din_instante": times,
        "val_geracao": 90.0,
        "val_geracaolimitada": 10.0,
    })
    acct = constrained_off_account(coff)
    row = acct[acct["month"] == 5].iloc[0]
    # The fraction is the point (delivered:curtailed held at 9:1 every hour);
    # robust to the UTC shift that moves a few edge hours between months.
    assert row["curtailed_fraction"] == pytest.approx(0.1)
    assert row["delivered_mwmed"] == pytest.approx(9.0 * row["curtailed_mwmed"])
    assert row["curtailed_mwmed"] > 0


def test_curtailment_mask_threshold():
    account = pd.DataFrame({
        "ID": ["A", "B"],
        "year": [2023, 2023],
        "month": [5, 6],
        "delivered_mwmed": [90.0, 98.0],
        "curtailed_mwmed": [10.0, 2.0],
        "curtailed_fraction": [0.10, 0.02],
    })
    mask = curtailment_mask_months(account, threshold=0.05)
    assert mask["ID"].tolist() == ["A"]  # only the 10% month is masked
    assert (2023, 5) == (mask.iloc[0]["year"], mask.iloc[0]["month"])


def test_build_metadata_contract_and_provenance():
    comp = wind_complexes_from_fc(fc_hours("CJU_A", "2023-03-01", "2023-03-02", 0.4))
    md = build_br_metadata(comp, height=100.0, model="2019COE_Market_Average_2.6MW_121")
    required = {"ID", "lon", "lat", "height", "capacity", "model", "type",
                "commissioning_date", "height_source"}
    assert required <= set(md.columns)
    row = md.iloc[0]
    assert row["capacity"] == pytest.approx(100_000.0)  # MW -> kW
    assert row["height_source"] == "default-uniform"
    assert row["type"] == "onshore"


def test_build_metadata_drops_missing_coordinates():
    comp = wind_complexes_from_fc(fc_hours("CJU_A", "2023-03-01", "2023-03-02", 0.4))
    comp.loc[comp["ID"] == "CJU_A", ["lon", "lat"]] = np.nan
    md = build_br_metadata(comp, height=100.0, model="M")
    assert len(md) == 0


def test_commissioning_from_siga_via_ceg():
    fc = fc_hours("CJU_A", "2023-03-01", "2023-03-02", 0.4)
    fc["ceg"] = "EOL.RS.001"  # this complex carries a CEG
    siga = pd.DataFrame({
        "CodCEG": ["EOL.RS.001", "EOL.RS.001", "UHE.XX.9"],
        "SigTipoGeracao": ["EOL", "EOL", "UHE"],
        "DatEntradaOperacao": ["2015-06-01", "2013-01-01", "2000-01-01"],
    })
    out = commissioning_from_siga(siga, fc)
    assert out.iloc[0]["ID"] == "CJU_A"
    # Earliest EOL operating date among the complex's CEGs (hydro ignored).
    assert out.iloc[0]["commissioning_date"] == pd.Timestamp("2013-01-01")
