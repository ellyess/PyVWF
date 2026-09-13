"""The curve library study driver's refusal (scripts/analysis/curve_library_study.py).

Two of the study's conditions reassign model keys and two of its registered
predictions say the reassignment will change nothing measurable. So a condition
whose overrides never applied produces exactly the result those predictions
expect, and the study's most likely bug is indistinguishable from its most
likely true result. These tests pin the refusal that separates them: no
condition runs on a fleet that is not the one it asked for.
"""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "curve_library_study",
    Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "curve_library_study.py",
)
study = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(study)


def fleet(ids=("a", "b", "c"), models=("X", "Y", "Z"), capacity=(1.0, 2.0, 7.0)):
    return pd.DataFrame({"ID": list(ids), "model": list(models),
                         "capacity": list(capacity)})


def overrides(mapping):
    return pd.Series(mapping, name="model").rename_axis("ID")


def test_a_fleet_that_matches_the_request_passes():
    study.check_overrides(fleet(models=("P", "Q", "Z")),
                          overrides({"a": "P", "b": "Q"}), "train")


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
    assert "80.00%" in message                      # 1 + 7 of 10 by capacity
    assert "'X' not 'P'" in message


def test_the_message_names_a_few_units_not_all_of_them():
    many = fleet(ids=[f"u{i}" for i in range(50)], models=["X"] * 50,
                 capacity=[1.0] * 50)
    with pytest.raises(study.OverrideError) as e:
        study.check_overrides(many, overrides({f"u{i}": "P" for i in range(50)}), "train")
    assert str(e.value).count("not 'P'") == study.SHOWN


def test_applying_an_override_leaves_other_units_alone():
    got = study.apply_overrides(fleet(), overrides({"b": "Q"}))
    assert list(got["model"]) == ["X", "Q", "Z"]


def test_the_wrapper_overrides_and_checks_the_frame_the_run_will_fit():
    calls = {}

    def loader(*args, **kwargs):
        calls["called"] = True
        return "obs", fleet(), "reanalysis", "curves"

    wrapped = study.patched_fleet(overrides({"a": "P"}), "train")(loader)
    _, turb_info, _, _ = wrapped()
    assert calls["called"] and list(turb_info["model"]) == ["P", "Y", "Z"]


def test_a_wrapper_whose_override_cannot_apply_refuses():
    def loader(*args, **kwargs):
        return "obs", fleet(), "reanalysis", "curves"

    wrapped = study.patched_fleet(overrides({"zz": "P"}), "train")(loader)
    with pytest.raises(study.OverrideError):
        wrapped()


def test_the_curve_library_is_checked_by_hash(tmp_path):
    (tmp_path / "run_manifest.json").write_text(json.dumps(
        {"curve_library": {"power_curves_sha256": "abc123"}}))
    study.check_library(tmp_path, "abc123")
    study.check_library(tmp_path, None)                  # unnamed: nothing to check
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
