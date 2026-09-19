"""The curve library study driver's refusal (scripts/studies/method-curve-library/curve_library_study.py).

Two of the study's conditions reassign model keys and two of its registered
predictions say the reassignment will change nothing measurable. So a condition
whose overrides never applied produces exactly the result those predictions
expect, and the study's most likely bug is indistinguishable from its most
likely true result. These tests pin the refusal that separates them: no
condition runs on a fleet that is not the one it asked for.
"""

import importlib.util
import json
import types
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "curve_library_study",
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "studies"
    / "method-curve-library"
    / "curve_library_study.py",
)
study = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(study)


def fleet(ids=("a", "b", "c"), models=("X", "Y", "Z"), capacity=(1.0, 2.0, 7.0)):
    return pd.DataFrame({"ID": list(ids), "model": list(models), "capacity": list(capacity)})


def overrides(mapping):
    return pd.Series(mapping, name="model").rename_axis("ID")


def test_a_fleet_that_matches_the_request_passes():
    study.check_overrides(fleet(models=("P", "Q", "Z")), overrides({"a": "P", "b": "Q"}), "train")


def test_an_override_that_reached_nothing_is_refused():
    """The failure that matters: an override keyed on ids the fleet does not
    use applies to no unit, and the run then simulates the row unchanged."""
    with pytest.raises(study.OverrideError) as e:
        study.check_overrides(fleet(), overrides({"x": "P", "y": "Q"}), "train")
    assert "not in the fleet" in str(e.value)
    assert "produces the same result as a condition that changes nothing" in str(e.value)


def test_integer_ids_on_one_side_do_not_silently_miss():
    """The realistic version of the same bug: the register keys on integers and
    the fleet on strings, so every lookup misses."""
    numeric = fleet(ids=(1, 2, 3), models=("P", "Q", "Z"))
    study.check_overrides(numeric, overrides({1: "P", 2: "Q"}), "train")
    study.check_overrides(numeric, overrides({"1": "P", "2": "Q"}), "train")


def test_a_unit_carrying_the_wrong_key_is_refused_with_its_share():
    with pytest.raises(study.OverrideError) as e:
        study.check_overrides(fleet(), overrides({"a": "P", "c": "W"}), "evaluate")
    message = str(e.value)
    assert "evaluate:" in message and "2 units" in message
    assert "80.00%" in message  # 1 + 7 of 10 by capacity
    assert "'X' not 'P'" in message


def test_the_message_names_a_few_units_not_all_of_them():
    many = fleet(ids=[f"u{i}" for i in range(50)], models=["X"] * 50, capacity=[1.0] * 50)
    with pytest.raises(study.OverrideError) as e:
        study.check_overrides(many, overrides({f"u{i}": "P" for i in range(50)}), "train")
    assert str(e.value).count("not 'P'") == study.SHOWN


def test_applying_an_override_leaves_other_units_alone():
    got = study.apply_overrides(fleet(), overrides({"b": "Q"}))
    assert list(got["model"]) == ["X", "Q", "Z"]


def test_the_condition_is_applied_where_the_fleet_enters():
    def prep(*args, **kwargs):
        return "obs", fleet()

    _, turb_info = study.applied_fleet(overrides({"a": "P"}))(prep)()
    assert list(turb_info["model"]) == ["P", "Y", "Z"]


def test_the_wrapper_on_the_loader_checks_and_does_not_apply():
    """If it applied, it would repair an override that never reached
    prep_country, after the simulation the scalar is fitted from, and the run
    would look correct while being a hybrid of two conditions."""

    def loader(*args, **kwargs):
        return "obs", fleet(), "reanalysis", "curves"

    wrapped = study.checked_fleet(overrides({"a": "P"}), "train")(loader)
    with pytest.raises(study.OverrideError) as e:
        wrapped()
    assert "'X' not 'P'" in str(e.value)


def test_a_loader_whose_fleet_already_carries_the_condition_passes():
    def loader(*args, **kwargs):
        return "obs", fleet(models=("P", "Y", "Z")), "reanalysis", "curves"

    wrapped = study.checked_fleet(overrides({"a": "P"}), "train")(loader)
    _, turb_info, _, _ = wrapped()
    assert list(turb_info["model"]) == ["P", "Y", "Z"]


def test_a_wrapper_whose_override_cannot_apply_refuses():
    def loader(*args, **kwargs):
        return "obs", fleet(), "reanalysis", "curves"

    wrapped = study.checked_fleet(overrides({"zz": "P"}), "train")(loader)
    with pytest.raises(study.OverrideError):
        wrapped()


def test_the_override_precedes_the_simulation_the_scalar_is_fitted_from(monkeypatch, tmp_path):
    """The defect this pins: train_set simulates the fleet before it returns,
    and the scalar is fitted as obs/sim from that frame. An override applied to
    the returned frame leaves the scalar on the old assignment. The keys the
    simulation sees must already be the condition's."""
    seen = {}

    def prep_country(*args, **kwargs):
        return "obs", fleet()

    module = types.SimpleNamespace(prep_country=prep_country)

    def train_set(*args, **kwargs):
        _, turb_info = module.prep_country()  # what train_set does first
        seen["at_simulation"] = list(turb_info["model"])  # then it simulates
        return "gen_cf", turb_info, "reanalysis", "curves"

    def run_train(spec, out_root, **kwargs):
        fake_driver.train_set(spec)
        (tmp_path / "train").mkdir(exist_ok=True)
        return tmp_path / "train"

    def run_evaluate(spec, train_dir, out_root, **kwargs):
        fake_driver.val_set(spec)
        (tmp_path / "evaluate").mkdir(exist_ok=True)
        return tmp_path / "evaluate"

    fake_driver = types.SimpleNamespace(
        train_set=train_set, val_set=train_set, run_train=run_train, run_evaluate=run_evaluate
    )
    monkeypatch.setattr(study, "vwf_data", module)
    monkeypatch.setattr(study, "driver", fake_driver)
    monkeypatch.setattr(study, "load_region", lambda config: "spec")

    study.run_condition("XX", "T1", tmp_path, overrides({"a": "P"}), config=Path("ignored.toml"))
    assert seen["at_simulation"] == ["P", "Y", "Z"]
    assert module.prep_country is prep_country  # restored afterwards


def test_the_curve_library_is_checked_by_hash(tmp_path):
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"curve_library": {"power_curves_sha256": "abc123"}})
    )
    study.check_library(tmp_path, "abc123")
    study.check_library(tmp_path, None)  # unnamed: nothing to check
    with pytest.raises(study.OverrideError) as e:
        study.check_library(tmp_path, "def456")
    assert "not the def456" in str(e.value)


def test_a_variant_root_links_everything_but_the_library(tmp_path):
    base = tmp_path / "base"
    (base / "era5").mkdir(parents=True)
    (base / "observations").mkdir()
    (base / "reference").mkdir()
    (base / "reference" / "power_curves.csv").write_text("open")
    other = tmp_path / "combined" / "reference"
    other.mkdir(parents=True)
    (other / "power_curves.csv").write_text("combined")

    root = study.variant_root(base, other, tmp_path / "variant")
    assert (root / "era5").resolve() == (base / "era5").resolve()
    assert (root / "reference" / "power_curves.csv").read_text() == "combined"
    assert all(p.is_symlink() for p in root.iterdir())


def test_building_a_variant_root_twice_is_idempotent(tmp_path):
    base = tmp_path / "base"
    (base / "reference").mkdir(parents=True)
    other = tmp_path / "other"
    other.mkdir()
    study.variant_root(base, other, tmp_path / "v")
    root = study.variant_root(base, other, tmp_path / "v")
    assert (root / "reference").resolve() == other.resolve()


def test_a_unit_declared_absent_from_this_phase_is_allowed():
    """A table covers both fleets, and the two fleets are not the same set.
    A unit the table declared this phase does not hold is not a miss."""
    study.check_overrides(
        fleet(models=("P", "Y", "Z")),
        overrides({"a": "P", "gone": "Q"}),
        "evaluate",
        expected_absent=["gone"],
    )


def test_an_undeclared_miss_is_still_refused_when_others_are_declared():
    """The declaration narrows the check, it does not switch it off: the
    type-mismatch bug misses every unit and declares none of them."""
    with pytest.raises(study.OverrideError) as e:
        study.check_overrides(
            fleet(), overrides({"x": "P", "gone": "Q"}), "train", expected_absent=["gone"]
        )
    assert "1 of 2 requested units" in str(e.value)
    assert "were not declared absent" in str(e.value)


def test_a_unit_declared_absent_that_is_present_is_refused():
    """The table's inventory is older than the fleet the run loads, so the two
    disagreeing means the table was built against something else."""
    with pytest.raises(study.OverrideError) as e:
        study.check_overrides(
            fleet(models=("P", "Y", "Z")), overrides({"a": "P"}), "evaluate", expected_absent=["b"]
        )
    assert "declares absent from this fleet are in it" in str(e.value)


def test_a_phase_the_table_reaches_nothing_in_is_refused():
    with pytest.raises(study.OverrideError) as e:
        study.check_overrides(
            fleet(), overrides({"x": "P", "y": "Q"}), "train", expected_absent=["x", "y"]
        )
    assert "reaches nothing here" in str(e.value)


def test_a_table_with_membership_columns_declares_absences(tmp_path):
    path = tmp_path / "T2_XX.csv"
    path.write_text("ID,model,in_train,in_test\na,P,True,True\nb,Q,True,False\nc,R,False,True\n")
    table, absent = study.read_table(path)
    assert list(table) == ["P", "Q", "R"]
    assert absent == {"train": ["c"], "evaluate": ["b"]}


def test_a_table_without_membership_columns_declares_nothing(tmp_path):
    """The single-fleet tables of 2026-09-13 and earlier: every unit named is
    expected in both fleets, which is the strict reading they were built for."""
    path = tmp_path / "T2_XX.csv"
    path.write_text("ID,model\na,P\nb,Q\n")
    table, absent = study.read_table(path)
    assert list(table) == ["P", "Q"] and absent == {}


def test_the_recorded_overrides_say_which_of_them_applied_here(tmp_path):
    study._write_overrides(tmp_path, overrides({"a": "P", "b": "Q"}), absent=["b"])
    got = pd.read_csv(tmp_path / "curve_overrides.csv")
    assert list(got["applied_here"]) == [True, False]
