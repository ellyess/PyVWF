"""AU-NEM raw-data processing: MMS parsing, fleet join, chunked partials."""
import io

import numpy as np
import pandas as pd
import pytest

from vwf.config import PyVWFPaths
from vwf.datasets.aemo_au import (
    build_au_metadata,
    farms_from_gwpt,
    join_fleet_to_gwpt,
    normalise_farm_name,
    parse_mms_table,
    wind_fleet_from_gen_info,
)
from vwf.sources import AEMONemSource
from vwf.sources.aemo import (
    combine_partials,
    scada_partial_aggregate,
    scada_to_monthly_cf,
)

MMS_TEXT = """C,SETP.WORLD,DVD_DISPATCH_UNIT_SCADA,AEMO,PUBLIC,2020/02/07,00:00:20,0,,0
I,DISPATCH,UNIT_SCADA,1,SETTLEMENTDATE,DUID,SCADAVALUE
D,DISPATCH,UNIT_SCADA,1,"2020/01/01 00:05:00",ARWF1,120.5
D,DISPATCH,UNIT_SCADA,1,"2020/01/01 00:05:00",COAL_G1,500
D,DISPATCH,UNIT_SCADA,1,"2020/01/01 00:10:00",ARWF1,118.2
C,"END OF REPORT",3
"""


def test_parse_mms_table_real_structure():
    df = parse_mms_table(io.StringIO(MMS_TEXT))
    assert list(df.columns) == ["SETTLEMENTDATE", "DUID", "SCADAVALUE"]
    assert len(df) == 3  # footer C row dropped, D rows kept
    assert df["DUID"].tolist() == ["ARWF1", "COAL_G1", "ARWF1"]
    assert df["SETTLEMENTDATE"].iloc[0] == "2020/01/01 00:05:00"  # quotes stripped


def test_parse_mms_table_rejects_wrong_shapes():
    with pytest.raises(ValueError, match="no 'I' header"):
        parse_mms_table(io.StringIO("C,only,a,comment\nD,1,2,3,4,5\n"))
    two_tables = MMS_TEXT + "I,DISPATCH,OTHER,1,A,B\nD,DISPATCH,OTHER,1,x,y\n"
    with pytest.raises(ValueError, match="multi-table"):
        parse_mms_table(io.StringIO(two_tables))


@pytest.mark.parametrize(
    "a,b",
    [
        ("Ararat Wind Farm", "Ararat wind farm"),
        ("Bald Hills Wind Farm", "Bald Hills WF"),
        ("Argoon Wind Farm (RES Group) - KCI", "Argoon wind farm"),
        ("Snowtown Wind Farm Stage 2", "Snowtown wind farm"),
    ],
)
def test_normalise_farm_name_matches_variants(a, b):
    assert normalise_farm_name(a) == normalise_farm_name(b)


def gen_info_frame():
    return pd.DataFrame(
        {
            "Site Name": [
                "Boco Rock Wind Farm", "Boco Rock Wind Farm",  # 2 unit groups, 1 DUID
                "Ararat Wind Farm", "Proposed Farm", "Gas Plant", "WA Wind Farm",
            ],
            "Technology Type": ["Wind", "Wind", "Wind", "Wind", "Gas Turbine", "Wind"],
            "DUID": ["BOCORWF1", "BOCORWF1", "ARWF1", None, "GAS1", "WAWF1"],
            "Commitment Status": [
                "In Service", "In Service", "In Service", "Publicly Announced",
                "In Service", "In Service",
            ],
            "Region": ["NSW1", "NSW1", "VIC1", "VIC1", "VIC1", "WEM"],
            "Agg Nameplate Capacity (MW AC)": [14.40, 98.60, 240.0, 100.0, 300.0, 50.0],
            "Full Commercial Use Date": [
                pd.NaT, pd.NaT, pd.Timestamp("2017-06-01"), pd.NaT, pd.NaT, pd.NaT,
            ],
        }
    )


def test_wind_fleet_from_gen_info():
    fleet = wind_fleet_from_gen_info(gen_info_frame())
    assert sorted(fleet["ID"]) == ["ARWF1", "BOCORWF1"]  # no DUID-less, gas, or WEM
    boco = fleet[fleet["ID"] == "BOCORWF1"].iloc[0]
    # Unit-group capacities SUM (9x1.60 + 58x1.70 style), not max/first.
    assert boco["capacity_mw"] == pytest.approx(113.0)
    ararat = fleet[fleet["ID"] == "ARWF1"].iloc[0]
    assert ararat["fcud"] == pd.Timestamp("2017-06-01")


def gwpt_frame():
    return pd.DataFrame(
        {
            "Country/Area": ["Australia"] * 4 + ["Germany"],
            "Status": ["operating", "operating", "operating", "construction", "operating"],
            "Project Name": [
                "Ararat wind farm", "Boco Rock wind farm", "Boco Rock wind farm",
                "Future farm", "Deutsch Windpark",
            ],
            "Capacity (MW)": [240.0, 60.0, 53.0, 90.0, 40.0],
            "Latitude": [-37.24, -36.90, -36.94, -35.0, 52.0],
            "Longitude": [143.04, 149.20, 149.30, 148.0, 9.0],
            "Start year": [2017, 2015, 2016, None, 2010],
        }
    )


def test_farms_from_gwpt_weighted_centroid():
    farms = farms_from_gwpt(gwpt_frame())
    assert sorted(farms["gwpt_name"]) == ["Ararat wind farm", "Boco Rock wind farm"]
    boco = farms[farms["gwpt_name"] == "Boco Rock wind farm"].iloc[0]
    assert boco["gwpt_capacity_mw"] == pytest.approx(113.0)
    # Capacity-weighted centroid, not plain mean: (60*-36.90 + 53*-36.94)/113
    expected_lat = (60.0 * -36.90 + 53.0 * -36.94) / 113.0
    assert boco["lat"] == pytest.approx(expected_lat)
    assert boco["lat"] != pytest.approx((-36.90 + -36.94) / 2, abs=1e-6)
    assert boco["start_year"] == 2015


def test_join_and_metadata_contract():
    fleet = wind_fleet_from_gen_info(gen_info_frame())
    farms = farms_from_gwpt(gwpt_frame())
    matched, unmatched_fleet, unmatched_gwpt = join_fleet_to_gwpt(fleet, farms)
    assert len(matched) == 2 and len(unmatched_fleet) == 0 and len(unmatched_gwpt) == 0

    md = build_au_metadata(matched, height=100.0, model="2019COE_Market_Average_2.6MW_121")
    required = {"ID", "lon", "lat", "height", "capacity", "model", "type",
                "commissioning_date"}
    assert required <= set(md.columns)
    boco = md[md["ID"] == "BOCORWF1"].iloc[0]
    assert boco["capacity"] == pytest.approx(113_000.0)  # MW -> kW
    # FCUD missing -> falls back to GWPT start year.
    assert boco["commissioning_date"] == pd.Timestamp("2015-01-01")
    ararat = md[md["ID"] == "ARWF1"].iloc[0]
    assert ararat["commissioning_date"] == pd.Timestamp("2017-06-01")  # FCUD wins
    assert (md["height_source"] == "default-uniform").all()


def test_duid_aliases_project_phase_and_below_threshold():
    """The phase-aware alias resolver: project targets aggregate phases (any
    status), phase targets take per-phase coordinates, BT: targets read the
    Below Threshold sheet, multi-target DUIDs get a weighted centroid."""
    from vwf.datasets.aemo_au import resolve_duid_aliases

    unmatched = pd.DataFrame(
        {
            "ID": ["AAWF1", "BBWF1", "CCWF1", "DDWF1"],
            "site_name": ["Alpha WF", "Beta Stage 1", "Gamma", "Delta"],
            "region": ["VIC1"] * 4,
            "capacity_mw": [100.0, 50.0, 12.0, 30.0],
            "fcud": [pd.NaT] * 4,
        }
    )
    gwpt_data = pd.DataFrame(
        {
            "Country/Area": ["Australia"] * 4,
            "Status": ["mothballed", "operating", "operating", "operating"],
            "Project Name": ["Alpha wind farm", "Beta wind farm", "Beta wind farm",
                             "Epsilon wind farm"],
            "Phase Name": ["--", "1", "2", "--"],
            "Capacity (MW)": [100.0, 50.0, 48.0, 20.0],
            "Latitude": [-37.0, -36.0, -36.5, -38.0],
            "Longitude": [143.0, 144.0, 144.5, 145.0],
            "Start year": [2020, 2018, 2019, 2015],
        }
    )
    gwpt_below = pd.DataFrame(
        {
            "Country/Area": ["Australia"],
            "Project Name": ["Gamma wind farm"],
            "Capacity (MW)": [12.0],
            "Latitude": [-37.5],
            "Longitude": [143.5],
            "Start year": [2012],
        }
    )
    alias_map = {
        "AAWF1": "Alpha wind farm",                    # any-status project
        "BBWF1": "Beta wind farm|1",                   # single phase
        "CCWF1": "BT:Gamma wind farm",                 # below threshold
        "DDWF1": "Alpha wind farm;Epsilon wind farm",  # multi-target centroid
    }
    matched, still = resolve_duid_aliases(unmatched, alias_map, gwpt_data, gwpt_below)
    assert len(matched) == 4 and len(still) == 0
    assert (matched["match_source"] == "alias").all()

    by_id = matched.set_index("ID")
    assert by_id.loc["AAWF1", "lat"] == pytest.approx(-37.0)  # mothballed OK: human-approved
    assert by_id.loc["BBWF1", "lat"] == pytest.approx(-36.0)  # phase 1, not the centroid
    assert by_id.loc["CCWF1", "lat"] == pytest.approx(-37.5)  # BT sheet
    # Capacity-weighted centroid over 100 MW @ -37 and 20 MW @ -38:
    assert by_id.loc["DDWF1", "lat"] == pytest.approx((100 * -37.0 + 20 * -38.0) / 120)
    # GI capacity stays authoritative; target capacity is context.
    assert by_id.loc["DDWF1", "capacity_mw"] == pytest.approx(30.0)
    assert by_id.loc["DDWF1", "gwpt_capacity_mw"] == pytest.approx(120.0)


def test_duid_alias_typo_fails_loudly():
    from vwf.datasets.aemo_au import resolve_duid_aliases

    unmatched = pd.DataFrame(
        {"ID": ["AAWF1"], "site_name": ["Alpha"], "region": ["VIC1"],
         "capacity_mw": [100.0], "fcud": [pd.NaT]}
    )
    gwpt_data = pd.DataFrame(
        {"Country/Area": ["Australia"], "Status": ["operating"],
         "Project Name": ["Alpha wind farm"], "Phase Name": ["--"],
         "Capacity (MW)": [100.0], "Latitude": [-37.0], "Longitude": [143.0],
         "Start year": [2020]}
    )
    with pytest.raises(ValueError, match="matched no GWPT row"):
        resolve_duid_aliases(unmatched, {"AAWF1": "Alpah wind farm"}, gwpt_data)


# ---------------------------------------------------------------------------
# Chunked partials: AEMO archives are cut on MARKET-month boundaries and
# straddle UTC months. Must-distinguish: the straddle is real on this
# fixture, and recombination fixes it.
# ---------------------------------------------------------------------------

def five_min(start, end, duid, mw):
    times = pd.date_range(start, end, freq="5min", inclusive="left")
    return pd.DataFrame({"timestamp": times, "ID": duid, "mw": float(mw)})


def test_chunked_partials_equal_one_shot_across_utc_boundary():
    # Two "archive" chunks cut at the market-time month boundary (Feb 1 AEST).
    chunk_jan = five_min("2021-01-01", "2021-02-01", "F1", 50.0)
    chunk_feb = five_min("2021-02-01", "2021-03-01", "F1", 50.0)
    one_shot = scada_partial_aggregate(pd.concat([chunk_jan, chunk_feb]))
    combined = combine_partials(
        [scada_partial_aggregate(chunk_jan), scada_partial_aggregate(chunk_feb)]
    )

    key = ["ID", "year", "month"]
    pd.testing.assert_frame_equal(
        one_shot.sort_values(key).reset_index(drop=True),
        combined.sort_values(key).reset_index(drop=True),
    )

    # The straddle exists: the market-Feb chunk contributes to UTC January
    # (00:00-09:55 AEST on 1 Feb is 31 Jan UTC), so naive per-chunk months
    # would double-report January. Distinguishable, not vacuous.
    feb_partial = scada_partial_aggregate(chunk_feb)
    assert ((feb_partial["month"] == 1).any())


def test_fast_path_equals_raw_scada_path(tmp_path, monkeypatch):
    """AEMONemSource with precomputed partials returns exactly what the raw
    SCADA path returns — the no-drift guarantee for the fast path."""
    scada = five_min("2021-01-01", "2022-01-01", "F1", 25.0)
    metadata = pd.DataFrame(
        {
            "ID": ["F1"], "lon": [149.0], "lat": [-35.0], "height": [100.0],
            "capacity": [100_000.0], "model": ["M"], "type": ["onshore"],
        }
    )
    data_dir = tmp_path / "AU_NEM"
    data_dir.mkdir()
    metadata.to_csv(data_dir / "au_nem_md.csv", index=False)
    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path)

    # Raw path
    scada.to_csv(data_dir / "au_nem_scada.csv", index=False)
    slow = AEMONemSource().load_observations(2021, 2021)

    # Fast path: partials present take precedence.
    partials = scada_partial_aggregate(scada)
    partials.to_csv(data_dir / "au_nem_scada_monthly_partials.csv", index=False)
    fast = AEMONemSource().load_observations(2021, 2021)

    pd.testing.assert_frame_equal(fast, slow)
    assert fast.iloc[0]["obs_6"] == pytest.approx(0.25, abs=0.005)

    # ...and the fixture distinguishes the paths: corrupt the partials and the
    # result must change (i.e. the fast path is really being read).
    broken = partials.copy()
    broken["energy_mwh"] *= 2
    broken.to_csv(data_dir / "au_nem_scada_monthly_partials.csv", index=False)
    doubled = AEMONemSource().load_observations(2021, 2021)
    assert doubled.iloc[0]["obs_6"] == pytest.approx(0.50, abs=0.01)


def test_capacity_history_and_mask_reasons():
    """DUDETAIL history dedup + the three mask reasons, each triggered."""
    from vwf.datasets.aemo_au import capacity_mask_months, registered_capacity_history

    dudetail = pd.DataFrame(
        {
            "DUID": ["F1"] * 3 + ["F1"],  # duplicate row + a version bump
            "EFFECTIVEDATE": ["2021-01-01", "2021-01-01", "2021-03-15", "2021-03-15"],
            "VERSIONNO": [1, 1, 1, 2],
            "REGISTEREDCAPACITY": [50.0, 50.0, 99.0, 100.0],
        }
    )
    hist = registered_capacity_history(dudetail)
    assert len(hist) == 2  # dedup + version collapse
    assert hist["REGISTEREDCAPACITY"].tolist() == [50.0, 100.0]  # v2 wins

    scada_months = pd.DataFrame(
        {
            "ID": ["F1"] * 4,
            "year": [2020, 2021, 2021, 2021],
            "month": [12, 2, 3, 6],
        }
    )
    mask = capacity_mask_months(hist, scada_months)
    by_key = {(r["year"], r["month"]): r["reason"] for _, r in mask.iterrows()}
    assert by_key[(2020, 12)] == "pre-registration"
    assert by_key[(2021, 2)] == "below-final-build"   # at 50 of final 100
    assert by_key[(2021, 3)] == "mid-month-change"    # 50 -> 100 on 15 March
    assert (2021, 6) not in by_key                    # fully built: kept


def test_capacity_mask_flows_through_source(tmp_path, monkeypatch):
    """Must-distinguish: with the mask file present the masked month is NaN;
    without it the same month has a value."""
    scada = five_min("2021-01-01", "2022-01-01", "F1", 25.0)
    metadata = pd.DataFrame(
        {"ID": ["F1"], "lon": [149.0], "lat": [-35.0], "height": [100.0],
         "capacity": [100_000.0], "model": ["M"], "type": ["onshore"]}
    )
    data_dir = tmp_path / "AU_NEM"
    data_dir.mkdir()
    metadata.to_csv(data_dir / "au_nem_md.csv", index=False)
    scada.to_csv(data_dir / "au_nem_scada.csv", index=False)
    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path)

    unmasked = AEMONemSource().load_observations(2021, 2021)
    assert unmasked.iloc[0]["obs_6"] == pytest.approx(0.25, abs=0.005)

    pd.DataFrame({"ID": ["F1"], "year": [2021], "month": [6],
                  "reason": ["below-final-build"]}).to_csv(
        data_dir / "au_nem_capacity_mask.csv", index=False)
    masked = AEMONemSource().load_observations(2021, 2021)
    assert np.isnan(masked.iloc[0]["obs_6"])
    # ...and only that month: July untouched.
    assert masked.iloc[0]["obs_7"] == pytest.approx(unmasked.iloc[0]["obs_7"])


def test_partials_compose_with_full_transform():
    """finalise(combine(partials)) == scada_to_monthly_cf on the same data."""
    from vwf.sources.aemo import finalise_monthly_cf

    scada = five_min("2021-01-01", "2021-07-01", "F1", 40.0)
    metadata = pd.DataFrame({"ID": ["F1"], "capacity": [100_000.0]})
    direct = scada_to_monthly_cf(scada, metadata, 2021, 2021)

    halves = [scada.iloc[: len(scada) // 2], scada.iloc[len(scada) // 2:]]
    via_partials = finalise_monthly_cf(
        combine_partials([scada_partial_aggregate(h) for h in halves]),
        metadata, 2021, 2021,
    )
    pd.testing.assert_frame_equal(direct, via_partials)
    assert not np.isnan(direct.iloc[0]["obs_3"])
