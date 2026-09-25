"""Curve resolution: which power curve each unit was actually simulated on.

Every country-level run on the bundled library simulated a 100 kW
distributed-wind turbine, because the grids name Vestas models the library
lacks and a missing model falls back, with a one-off warning and no record, to
the table's first column. These tests pin that fallback's identity, check the
resolution record against what the simulation actually does, and check the
driver writes it where the numbers are.
"""

import json
import warnings
from importlib import resources

import pandas as pd
import pytest

import test_pipeline as tp
import vwf.wind as wind
from test_harness_driver import make_spec
from vwf.config import PyVWFPaths
from vwf.curves import _default_power_curve, add_models
from vwf.harness.driver import CurveSubstitutionError, run_evaluate, run_train, run_transfer
from vwf.provenance import curve_resolution, summarise_curve_resolution
from vwf.sources import InMemoryCountrySource, get_source

#: The curve the bundled library substitutes for any model it lacks. Pinned
#: because it decides results: if the library is reordered, every country-level
#: number changes, and this is where that shows.
BUNDLED_FALLBACK = "2019COE_DW100_100kW_27.6"


@pytest.fixture(scope="module")
def bundled_curves():
    return pd.read_csv(str(resources.files("vwf.resources") / "power_curves.csv"))


@pytest.fixture
def synthetic_dk(tmp_path, monkeypatch):
    """The synthetic Denmark input tree of tests/test_harness_driver.py."""
    tp._write_era5(tmp_path / "era5")
    fleet = tp._write_fleet(tmp_path / "observations/turbine" / "DK")
    monkeypatch.setattr(PyVWFPaths, "INPUT_ROOT", tmp_path)
    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path / "observations/turbine")
    monkeypatch.setattr(PyVWFPaths, "ERA5_DATA", tmp_path / "era5")
    return {"root": tmp_path, "fleet": fleet}


def _fleet(models, capacities=None, **extra):
    n = len(models)
    return pd.DataFrame(
        {
            "ID": [f"u{i}" for i in range(n)],
            "lat": [55.2 + 0.1 * i for i in range(n)],
            "lon": [8.2 + 0.1 * i for i in range(n)],
            "height": [100.0] * n,
            "capacity": capacities if capacities is not None else [1000.0] * n,
            "model": models,
            **extra,
        }
    )


# --- the fallback's identity --------------------------------------------------


def test_bundled_fallback_identity_is_pinned(bundled_curves):
    """The simulation's fallback, the country-grid default and the resolution
    record all name the same curve, and for the bundled library it is this one.
    The defect stays until the study decides; the surprise does not."""
    assert wind.default_curve_key(bundled_curves) == BUNDLED_FALLBACK
    assert _default_power_curve(bundled_curves) == BUNDLED_FALLBACK

    _, curve_by_model = wind._get_power_curve_cache(bundled_curves)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert curve_by_model["Vestas.V90.3000"] is curve_by_model[BUNDLED_FALLBACK]


def test_the_log_matches_the_simulation(reanalysis, bundled_curves):
    """A unit with a missing model simulates bit-identically to the same unit
    carrying the key the resolution record reports. Fails if the record and the
    lookup ever drift apart."""
    missing = _fleet(["Vestas.V90.3000"])
    reported = curve_resolution(missing, bundled_curves)["curve_used"].iloc[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, cf_missing = wind.simulate_wind(reanalysis, missing, bundled_curves)
    _, cf_reported = wind.simulate_wind(reanalysis, _fleet([reported]), bundled_curves)
    pd.testing.assert_frame_equal(cf_missing, cf_reported)


# --- the record itself --------------------------------------------------------


def test_a_fully_resolved_fleet_records_no_substitution(bundled_curves):
    res = curve_resolution(_fleet(["VestasV47_660kW_47", "NREL_Reference_5MW_126"]), bundled_curves)
    assert (res["status"] == "resolved").all()
    assert (res["curve_used"] == res["requested"]).all()
    assert (res["origin"] == "open").all()
    summary = summarise_curve_resolution(res)
    assert summary["substituted_capacity_share"] == 0.0
    assert summary["substitutions"] == {}
    assert summary["open_capacity_share"] == pytest.approx(1.0)


def test_a_missing_model_records_its_substitute_and_capacity_share(bundled_curves):
    res = curve_resolution(
        _fleet(["VestasV47_660kW_47", "Vestas.V90.3000"], capacities=[1000.0, 3000.0]),
        bundled_curves,
    ).set_index("requested")
    assert res.loc["Vestas.V90.3000", "status"] == "substituted"
    assert res.loc["Vestas.V90.3000", "curve_used"] == BUNDLED_FALLBACK
    assert res.loc["Vestas.V90.3000", "capacity_share"] == pytest.approx(0.75)
    summary = summarise_curve_resolution(res.reset_index())
    assert summary["substituted_capacity_share"] == pytest.approx(0.75)
    assert summary["substitutions"] == {"Vestas.V90.3000": BUNDLED_FALLBACK}


def test_country_grid_on_the_bundled_library_is_fully_substituted(bundled_curves):
    """Regression for the defect: every scorecard country grid names one of
    these three, and against the bundled library all resolve to the 100 kW
    fallback."""
    grid = _fleet(["Vestas.V80.2000", "Vestas.V90.2000", "Vestas.V90.3000"])
    summary = summarise_curve_resolution(curve_resolution(grid, bundled_curves))
    assert summary["substituted_capacity_share"] == pytest.approx(1.0)
    assert set(summary["substitutions"].values()) == {BUNDLED_FALLBACK}


def test_origin_is_decided_by_curve_values_not_names(bundled_curves):
    """A key only counts as open if it carries the open curve: an extra column
    is external, and so is an open name whose values were changed."""
    table = bundled_curves[["data$speed", "VestasV47_660kW_47"]].copy()
    table["Licensed.Model"] = table["VestasV47_660kW_47"] * 0.99
    table["VestasV47_660kW_47"] = table["VestasV47_660kW_47"] * 0.98
    res = curve_resolution(_fleet(["Licensed.Model", "VestasV47_660kW_47"]), table).set_index(
        "requested"
    )
    assert res.loc["Licensed.Model", "origin"] == "external"
    assert res.loc["VestasV47_660kW_47", "origin"] == "external"


def test_assignment_is_read_from_the_metadata(bundled_curves):
    fleet = _fleet(
        ["VestasV47_660kW_47", "VestasV47_660kW_47"],
        model_source=["verified", "matched-scale-and-specific-power"],
    )
    res = curve_resolution(fleet, bundled_curves)
    assert res["assigned_by"].iloc[0] == "matched-scale-and-specific-power;verified"
    assert (
        curve_resolution(_fleet(["VestasV47_660kW_47"]), bundled_curves)["assigned_by"].iloc[0]
        == "as-given"
    )


# --- add_models records how it matched ---------------------------------------


def _metadata(manufacturer, capacity, diameter):
    return pd.DataFrame(
        {
            "ID": ["a"],
            "manufacturer": [manufacturer],
            "capacity": [capacity],
            "diameter": [diameter],
            "height": [80.0],
            "lon": [9.0],
            "lat": [56.0],
        }
    )


def test_add_models_records_a_manufacturer_match():
    # A Vestas V27 (225 kW, 27 m rotor) matches on manufacturer, but the fuzzy
    # match also admits EWT, whose DW54 has the identical specific power, so the
    # tier is asserted and the machine is not: it is a match within 1 W/m2 from
    # a fuzzily matched manufacturer, not necessarily the turbine's own.
    out = add_models(_metadata("VestasV", 225.0, 27.0))
    assert out["model_match"].iloc[0] == "fuzzy-manufacturer+specific-power"
    assert out["model"].iloc[0] in {"VestasV27_225kW_27", "EWT_DW54_900kW_54"}


def test_add_models_records_a_specific_power_only_match():
    out = add_models(_metadata("qqqq", 225.0, 27.0))
    assert out["model_match"].iloc[0] == "specific-power-only"


# --- the driver writes it where the numbers are -------------------------------


def test_driver_records_resolution_in_train_and_evaluate(synthetic_dk):

    spec = make_spec()
    out = synthetic_dk["root"] / "validation"
    train_dir = run_train(spec, out, mode="onshore", run_name="t")
    eval_dir = run_evaluate(spec, train_dir, out, mode="onshore", run_name="e")

    for run in (train_dir, eval_dir):
        res = pd.read_csv(run / "curve_resolution.csv")
        assert (res["status"] == "resolved").all()
        assert set(res["assigned_by"]) <= {
            "fuzzy-manufacturer+specific-power",
            "specific-power-only",
        }
        manifest = json.loads((run / "run_manifest.json").read_text())
        assert manifest["curve_resolution"]["substituted_capacity_share"] == 0.0

    metrics = pd.read_csv(eval_dir / "metrics.csv")
    assert (metrics["substituted_capacity_share"] == 0.0).all()


def _country_grid(model):
    return pd.DataFrame(
        {
            "ID": ["g0", "g1", "g2", "g3"],
            "lon": [8.1, 8.3, 9.2, 9.4],
            "lat": [55.2, 55.4, 55.6, 55.8],
            "height": [100.0] * 4,
            "capacity": [2000.0, 2000.0, 4000.0, 4000.0],
            "model": [model] * 4,
            "cluster": [0, 0, 1, 1],
            "type": ["onshore"] * 4,
        }
    )


def _country_source(grid, year):
    idx = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h", tz="UTC")
    return InMemoryCountrySource(grid, pd.DataFrame({"capacity_factor": 0.2}, index=idx))


COUNTRY_SPEC = dict(
    source="in-memory-country", obs_level="country", obs_unit="country", cluster_list=(2,)
)


def test_a_country_run_with_a_missing_model_is_refused_and_recorded(synthetic_dk):
    """Every grid point names one key, so a missing curve is the whole row.

    Until 2026-09-25 this was recorded and the run went on, on the 100 kW
    fallback; the record is still written before the refusal.
    """
    spec = make_spec(**COUNTRY_SPEC)
    out = synthetic_dk["root"] / "cl"
    with pytest.raises(CurveSubstitutionError, match=r"Vestas\.V90\.3000.*input/combined"):
        run_train(
            spec, out, run_name="t", source=_country_source(_country_grid("Vestas.V90.3000"), 2015)
        )
    res = pd.read_csv(out / "DK" / "train-t" / "curve_resolution.csv")
    assert res["curve_used"].tolist() == [BUNDLED_FALLBACK]
    assert res["status"].tolist() == ["substituted"]


def test_a_country_run_whose_model_resolves_trains_and_evaluates(synthetic_dk):
    spec = make_spec(**COUNTRY_SPEC)
    out = synthetic_dk["root"] / "cl"
    grid = _country_grid("VestasV82_1.65MW_82")
    train_dir = run_train(spec, out, run_name="t", source=_country_source(grid, 2015))
    eval_dir = run_evaluate(spec, train_dir, out, run_name="e", source=_country_source(grid, 2016))
    metrics = pd.read_csv(eval_dir / "metrics.csv")
    assert (metrics["substituted_capacity_share"] == 0.0).all()


def test_a_country_evaluation_with_a_missing_model_is_refused(synthetic_dk):
    """The training fleet resolved; the test-year fleet names a key that does not."""
    spec = make_spec(**COUNTRY_SPEC)
    out = synthetic_dk["root"] / "cl"
    train_dir = run_train(
        spec, out, run_name="t", source=_country_source(_country_grid("VestasV82_1.65MW_82"), 2015)
    )
    with pytest.raises(CurveSubstitutionError):
        run_evaluate(
            spec,
            train_dir,
            out,
            run_name="e",
            source=_country_source(_country_grid("Vestas.V90.3000"), 2016),
        )


def test_driver_transfer_records_resolution(synthetic_dk):

    source_spec = make_spec(code="AU-NEM")
    target_spec = make_spec()
    out = synthetic_dk["root"] / "tr"
    train_dir = run_train(
        source_spec,
        out,
        mode="onshore",
        run_name="t",
        source=get_source("european-turbine", "DK"),
    )
    tr_dir = run_transfer(source_spec, train_dir, target_spec, out, mode="onshore", run_name="x")
    assert (tr_dir / "curve_resolution.csv").is_file()
    metrics = pd.read_csv(tr_dir / "metrics.csv")
    assert (metrics["substituted_capacity_share"] == 0.0).all()
    manifest = json.loads((tr_dir / "run_manifest.json").read_text())
    assert "curve_resolution" in manifest
