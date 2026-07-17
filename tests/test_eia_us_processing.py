"""EIA-US raw-data processing: EIA-923 reshape, EIA-860 capacity, USWTDB join.

All fixtures are synthetic; real EIA/USWTDB acquisition is Phase 2. The
fixtures carry the schedule's real quirks on purpose — thousands separators,
the ``.`` missing marker, the ``-9999`` USWTDB sentinel, and the wind-vs-other
fuel filter — because handling those is part of the contract under test.
"""
import numpy as np
import pandas as pd
import pytest

from vwf.datasets.eia_us import (
    build_us_metadata,
    plant_hub_heights_from_uswtdb,
    wind_capacity_from_eia860,
    wind_generation_from_eia923,
)

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def eia923_frame():
    """Two plants (one wind, one gas) + a second wind row to exercise summing."""
    rows = []
    # Plant 100: wind, monthly respondent, 1,000 MWh every month (comma-formatted).
    row = {"Plant Id": 100, "Reported Fuel Type Code": "WND",
           "Reported Prime Mover": "WT", "YEAR": 2020, "Respondent Frequency": "M"}
    for m in MONTHS:
        row[f"Netgen_{m}"] = "1,000"
    row["Netgen_March"] = "."  # withheld month -> NaN, not 0
    rows.append(row)
    # Plant 200: gas, must be filtered out.
    row = {"Plant Id": 200, "Reported Fuel Type Code": "NG",
           "Reported Prime Mover": "CT", "YEAR": 2020, "Respondent Frequency": "M"}
    for m in MONTHS:
        row[f"Netgen_{m}"] = "5,000"
    rows.append(row)
    # Plant 300: wind, annual respondent (imputed monthly cells).
    row = {"Plant Id": 300, "Reported Fuel Type Code": "WND",
           "Reported Prime Mover": "WT", "YEAR": 2020, "Respondent Frequency": "A"}
    for m in MONTHS:
        row[f"Netgen_{m}"] = "800"
    rows.append(row)
    return pd.DataFrame(rows)


def test_wind_generation_filters_and_melts():
    long = wind_generation_from_eia923(eia923_frame())
    assert set(long["ID"]) == {"100", "300"}  # gas plant 200 dropped
    p100 = long[long["ID"] == "100"]
    assert len(p100) == 12  # one row per month
    jan = p100[p100["month"] == 1].iloc[0]
    assert jan["net_gen_mwh"] == pytest.approx(1000.0)  # comma parsed
    mar = p100[p100["month"] == 3].iloc[0]
    assert np.isnan(mar["net_gen_mwh"])  # "." -> NaN, not 0
    assert (p100["respondent_frequency"] == "M").all()
    assert (long[long["ID"] == "300"]["respondent_frequency"] == "A").all()


def test_wind_generation_tolerates_multiline_headers():
    """The real EIA-923 workbook wraps headers across two lines, so pandas
    reads 'Reported\\nFuel Type Code', 'Respondent\\nFrequency',
    'Netgen\\nJanuary'. The reshape must handle them exactly as the single-line
    fixtures (Phase 2 caught this on the real 2021 file)."""
    frame = eia923_frame()
    frame = frame.rename(
        columns={
            "Reported Fuel Type Code": "Reported\nFuel Type Code",
            "Respondent Frequency": "Respondent\nFrequency",
            **{f"Netgen_{m}": f"Netgen\n{m}" for m in MONTHS},
        }
    )
    long = wind_generation_from_eia923(frame)
    assert set(long["ID"]) == {"100", "300"}
    jan = long[(long["ID"] == "100") & (long["month"] == 1)].iloc[0]
    assert jan["net_gen_mwh"] == pytest.approx(1000.0)
    assert (long[long["ID"] == "300"]["respondent_frequency"] == "A").all()


def test_wind_generation_sums_multiple_wind_rows():
    """A plant with two wind rows (e.g. two prime-mover groupings) sums."""
    frame = eia923_frame()
    extra = frame[frame["Plant Id"] == 100].copy()
    for m in MONTHS:
        extra[f"Netgen_{m}"] = "250"
    extra["Netgen_March"] = "250"
    doubled = pd.concat([frame, extra], ignore_index=True)
    long = wind_generation_from_eia923(doubled)
    jan = long[(long["ID"] == "100") & (long["month"] == 1)].iloc[0]
    assert jan["net_gen_mwh"] == pytest.approx(1250.0)  # 1000 + 250


def test_wind_generation_rejects_conflicting_frequency():
    frame = eia923_frame()
    dup = frame[frame["Plant Id"] == 100].copy()
    dup["Respondent Frequency"] = "A"  # same plant-year, other flag
    with pytest.raises(ValueError, match="conflicting respondent"):
        wind_generation_from_eia923(pd.concat([frame, dup], ignore_index=True))


def eia860_generators():
    return pd.DataFrame(
        {
            "Plant Code": [100, 100, 300, 400],
            "Prime Mover": ["WT", "WT", "WT", "CT"],  # last is gas, ignored
            "Nameplate Capacity (MW)": ["50.0", "50.0", "120.0", "300.0"],
            "Operating Year": [2015, 2016, 2018, 2010],
            "Operating Month": [6, 3, 11, 1],
        }
    )


def eia860_plants():
    return pd.DataFrame(
        {
            "Plant Code": [100, 300, 400],
            "Plant Name": ["Alpha Wind", "Gamma Wind", "Gas Plant"],
            "Latitude": ["41.5", "35.2", "29.0"],
            "Longitude": ["-104.2", "-101.9", "-95.0"],
        }
    )


def test_wind_capacity_sums_and_dates():
    cap = wind_capacity_from_eia860(eia860_generators(), eia860_plants())
    assert set(cap["ID"]) == {"100", "300"}  # gas-only plant 400 dropped
    p100 = cap[cap["ID"] == "100"].iloc[0]
    assert p100["capacity_mw"] == pytest.approx(100.0)  # 50 + 50
    # Earliest operating month across the plant's wind generators.
    assert p100["commissioning_date"] == pd.Timestamp("2015-06-01")
    assert p100["lat"] == pytest.approx(41.5)
    assert p100["lon"] == pytest.approx(-104.2)
    assert p100["site_name"] == "Alpha Wind"


def uswtdb_frame():
    return pd.DataFrame(
        {
            "eia_id": [100, 100, 100, 300, np.nan],
            "t_cap": [2000.0, 2000.0, 3000.0, 2500.0, 2000.0],
            "t_hh": [80.0, 80.0, 100.0, -9999.0, 90.0],  # -9999 sentinel on one
            "t_rd": [90.0, 90.0, 110.0, 100.0, 90.0],
            "t_manu": ["GE", "GE", "Vestas", "GE", "GE"],
            "t_model": ["GE1.5", "GE1.5", "V100", "GE2.5", "GE1.5"],
        }
    )


def test_hub_heights_capacity_weighted_and_sentinel():
    hh = plant_hub_heights_from_uswtdb(uswtdb_frame())
    assert set(hh["ID"]) == {"100", "300"}  # NaN eia_id row dropped
    p100 = hh[hh["ID"] == "100"].iloc[0]
    # Capacity-weighted: (80*2000 + 80*2000 + 100*3000) / 7000
    expected = (80 * 2000 + 80 * 2000 + 100 * 3000) / 7000
    assert p100["hub_height_m"] == pytest.approx(expected)
    assert p100["n_turbines"] == 3
    # Dominant model by installed capacity: Vestas V100 has 3000 > GE 4000?
    # GE1.5 total 4000 kW vs V100 3000 kW -> GE wins.
    assert p100["uswtdb_model"] == "GE GE1.5"
    # Plant 300's only hub height is the -9999 sentinel -> NaN, not -9999.
    p300 = hh[hh["ID"] == "300"].iloc[0]
    assert np.isnan(p300["hub_height_m"])


def test_build_metadata_contract_and_height_provenance():
    cap = wind_capacity_from_eia860(eia860_generators(), eia860_plants())
    hh = plant_hub_heights_from_uswtdb(uswtdb_frame())
    md = build_us_metadata(cap, hh, default_height=100.0, model="Synthetic.Onshore2000")

    required = {"ID", "lon", "lat", "height", "capacity", "model", "type",
                "commissioning_date", "height_source"}
    assert required <= set(md.columns)
    p100 = md[md["ID"] == "100"].iloc[0]
    assert p100["capacity"] == pytest.approx(100_000.0)  # MW -> kW
    assert p100["height_source"] == "uswtdb-capacity-weighted"
    assert p100["type"] == "onshore"
    # Plant 300 had no valid hub height -> uniform default, flagged as such.
    p300 = md[md["ID"] == "300"].iloc[0]
    assert p300["height"] == pytest.approx(100.0)
    assert p300["height_source"] == "default-uniform"


def test_build_metadata_drops_plants_without_coordinates():
    cap = wind_capacity_from_eia860(eia860_generators(), eia860_plants())
    # Blank out plant 300's coordinates: it must not appear in the metadata.
    cap.loc[cap["ID"] == "300", ["lon", "lat"]] = np.nan
    md = build_us_metadata(cap, None, default_height=100.0, model="M")
    assert set(md["ID"]) == {"100"}
    assert (md["height_source"] == "default-uniform").all()  # no USWTDB passed
