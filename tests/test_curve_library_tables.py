"""The curve library study's table construction (scripts/studies/method-curve-library/curve_library_tables.py).

A run fits the training fleet and is scored on the test fleet, and the two are
not the same set. A table built from the training fleet alone therefore leaves
the units installed after the training window on their own keys at evaluation,
which is not the condition anyone registered. These tests pin the construction
that covers both fleets, and the refusal that fires when one unit would need
two keys.
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "curve_library_tables",
    Path(__file__).resolve().parents[1] / "scripts" / "studies" / "method-curve-library" / "curve_library_tables.py",
)
tables = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tables)


def frame(ids, models, capacity=None, diameter=None):
    out = pd.DataFrame({"ID": list(ids), "model": list(models)})
    out["capacity"] = list(capacity) if capacity else [1000.0] * len(out)
    out["diameter"] = list(diameter) if diameter else [80.0] * len(out)
    return out


def fleets(monkeypatch, train, test):
    monkeypatch.setattr(tables, "train_fleet_of", lambda code: train)
    monkeypatch.setattr(tables, "test_fleet_of", lambda code: test)


def test_the_union_covers_both_fleets_and_says_which_holds_each_unit(monkeypatch):
    fleets(monkeypatch, frame(["a", "b"], ["X", "Y"]), frame(["b", "c"], ["Y", "Z"]))
    got = tables.load_fleets("XX", "C2")
    assert list(got.union["ID"]) == ["a", "b", "c"]
    assert list(got.union["in_train"]) == [True, True, False]
    assert list(got.union["in_test"]) == [False, True, True]
    assert len(got.train) == 2 and len(got.test) == 2


def test_a_unit_whose_key_differs_between_the_fleets_is_refused(monkeypatch):
    """One unit would then owe the table two keys, and which one applied would
    depend on the phase, which is not a condition anyone registered."""
    fleets(monkeypatch, frame(["a"], ["X"]), frame(["a"], ["Y"]))
    with pytest.raises(SystemExit) as e:
        tables.load_fleets("XX", "C2")
    assert "different model" in str(e.value)


def test_a_field_the_condition_does_not_read_may_differ(monkeypatch):
    """The country grids carry a per-year capacity, so a grid point is 0 MW in
    the training fleet and 11 MW in the test one. C2 maps a model key to a
    substitute and reads no capacity, so that is not a conflict for C2."""
    fleets(monkeypatch, frame(["a"], ["X"], capacity=[0.0]),
           frame(["a"], ["X"], capacity=[11.0]))
    assert len(tables.load_fleets("XX", "C2").union) == 1
    with pytest.raises(SystemExit):
        tables.load_fleets("XX", "T2")          # T2 picks a band from the rating


def test_the_union_row_of_a_shared_unit_is_the_training_one(monkeypatch):
    fleets(monkeypatch, frame(["a"], ["X"], capacity=[0.0]),
           frame(["a"], ["X"], capacity=[11.0]))
    assert float(tables.load_fleets("XX", "C2").union.loc[0, "capacity"]) == 0.0


def test_a_written_table_carries_the_membership_and_drops_unmatched_units(tmp_path):
    fleet = frame(["a", "b"], ["X", "Y"]).assign(in_train=[True, False],
                                                 in_test=[True, True])
    table = tables.write_table(tmp_path, "T1_XX", fleet, pd.Series(["P", None]))
    assert list(table["ID"]) == ["a"] and list(table["in_train"]) == [True]
    assert (tmp_path / "T1_XX.csv").exists()


def test_coverage_is_reported_against_each_fleet_separately(tmp_path):
    got = tables.coverage(
        tables.Fleets(frame(["a", "b", "c"], ["X", "Y", "Z"]),
                      frame(["a", "b"], ["X", "Y"]), frame(["b", "c"], ["Y", "Z"])),
        pd.DataFrame({"ID": ["a", "b"], "model": ["P", "Q"]}))
    assert got["reached_train"] == 2 and got["reached_test"] == 1
    assert got["capacity_share_train"] == 1.0 and got["capacity_share_test"] == 0.5
