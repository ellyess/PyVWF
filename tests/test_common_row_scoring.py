"""Conditions compared with each other are scored on the same rows.

Scored one at a time, a corrected variant with no value for some units was
compared with the uncorrected variant on a different set of rows. Chile's
scorecard row is the case that exposed it: the corrected score covered 55
plants and the uncorrected one 59, and the four missing plants were the worst.
These tests pin the rule and its record.
"""

import json

import numpy as np
import pandas as pd
import pytest

from test_harness_driver import make_spec
from pyvwf.harness import driver
from pyvwf.harness.driver import run_evaluate, run_train
from pyvwf.harness.skill import (
    restrict_to_common_rows,
    skill_metrics,
    summarise_exclusions,
)

KEYS = ["ID", "year", "month"]


def _frame(values: dict[tuple[str, int], float], obs: float = 0.3) -> pd.DataFrame:
    rows = [
        {
            "ID": unit,
            "year": 2019,
            "month": month,
            "cf_sim": sim,
            "cf_obs": obs,
            "capacity": {"A": 1.0, "B": 2.0, "C": 3.0}[unit],
        }
        for (unit, month), sim in values.items()
    ]
    return pd.DataFrame(rows)


def _two_conditions():
    """C has no corrected value at all and is the worst unit uncorrected; B
    lacks its corrected value in month 2."""
    unc = _frame(
        {
            ("A", 1): 0.35,
            ("A", 2): 0.35,
            ("B", 1): 0.40,
            ("B", 2): 0.40,
            ("C", 1): 0.90,
            ("C", 2): 0.90,
        }
    )
    cor = _frame(
        {
            ("A", 1): 0.31,
            ("A", 2): 0.31,
            ("B", 1): 0.32,
            ("B", 2): np.nan,
            ("C", 1): np.nan,
            ("C", 2): np.nan,
        }
    )
    return {"uncorrected": unc, "fixed_2": cor}


def test_every_condition_is_scored_on_the_rows_all_can_score():
    frames = _two_conditions()
    restricted, excluded = restrict_to_common_rows(frames, KEYS)

    expected = {("A", 1), ("A", 2), ("B", 1)}
    for frame in restricted.values():
        assert set(zip(frame["ID"], frame["month"])) == expected
    assert skill_metrics(restricted["uncorrected"])["n_samples"] == 3
    assert skill_metrics(restricted["fixed_2"])["n_samples"] == 3

    assert set(zip(excluded["ID"], excluded["month"])) == {("B", 2), ("C", 1), ("C", 2)}
    assert set(excluded["missing_in"]) == {"fixed_2"}
    assert excluded.set_index(["ID", "month"])["capacity"].to_dict() == {
        ("B", 2): 2.0,
        ("C", 1): 3.0,
        ("C", 2): 3.0,
    }


def test_separate_scoring_flattered_the_correction():
    """Must-distinguish: the old per-condition scoring and the common-row
    scoring give different gains on this fixture, and the old one is larger
    because it dropped the worst unit from one side only."""
    frames = _two_conditions()
    old_gain = (
        skill_metrics(frames["uncorrected"])["rmse"] - skill_metrics(frames["fixed_2"])["rmse"]
    )
    restricted, _ = restrict_to_common_rows(frames, KEYS)
    new_gain = (
        skill_metrics(restricted["uncorrected"])["rmse"]
        - skill_metrics(restricted["fixed_2"])["rmse"]
    )
    assert old_gain > new_gain + 0.1


def test_summary_records_share_and_wholly_excluded_units():
    frames = _two_conditions()
    _, excluded = restrict_to_common_rows(frames, KEYS)
    summary = summarise_exclusions(frames, excluded, KEYS)
    assert summary["n_rows_scorable"] == 6
    assert summary["n_rows_scored"] == 3
    assert summary["n_rows_excluded"] == 3
    # Capacity: excluded B(2) + C(3) + C(3) = 8 of A 2 + B 4 + C 6 = 12.
    assert summary["excluded_share"] == pytest.approx(8 / 12)
    assert summary["units_wholly_excluded"] == ["C"]


def test_no_op_when_every_condition_is_complete():
    unc = _frame({("A", 1): 0.35, ("B", 1): 0.40})
    cor = _frame({("A", 1): 0.31, ("B", 1): 0.32})
    restricted, excluded = restrict_to_common_rows({"u": unc, "c": cor}, KEYS)
    assert excluded.empty
    assert list(excluded.columns) == [*KEYS, "capacity", "missing_in"]
    pd.testing.assert_frame_equal(restricted["u"], unc)
    pd.testing.assert_frame_equal(restricted["c"], cor)


def test_a_row_no_condition_can_score_is_not_listed_as_excluded():
    """A missing observation removes the row for everyone; it was never part
    of any comparison, so it is not an exclusion."""
    unc = _frame({("A", 1): 0.35, ("B", 1): 0.40})
    cor = _frame({("A", 1): 0.31, ("B", 1): 0.32})
    for frame in (unc, cor):
        frame.loc[frame["ID"] == "B", "cf_obs"] = np.nan
    restricted, excluded = restrict_to_common_rows({"u": unc, "c": cor}, KEYS)
    assert excluded.empty
    assert list(restricted["u"]["ID"]) == ["A"]


def test_monthly_aggregate_keys_count_rows_without_a_weight():
    months = pd.period_range("2023-01", periods=4, freq="M")
    unc = pd.DataFrame({"ym": months, "cf_sim": 0.3, "cf_obs": 0.25})
    cor = unc.assign(cf_sim=[0.26, 0.26, np.nan, 0.26])
    frames = {"uncorrected": unc, "fixed_1": cor}
    restricted, excluded = restrict_to_common_rows(frames, ["ym"], weight=None)
    assert [len(f) for f in restricted.values()] == [3, 3]
    assert list(excluded["ym"]) == [months[2]]
    summary = summarise_exclusions(frames, excluded, ["ym"], weight=None, unit=None)
    assert summary["excluded_share"] == pytest.approx(0.25)


def test_evaluate_scores_variants_on_common_rows_and_records_the_exclusion(
    synthetic_dk,
    monkeypatch,
):
    """End to end: one unit's corrected values are removed, as a failed
    cluster fit would. Both variants are then scored on the remaining units,
    and the run output names the unit and the variant that lacked it."""
    spec = make_spec()
    out = synthetic_dk["root"] / "validation"
    train_dir = run_train(spec, out, mode="onshore", run_name="t1")

    clean_dir = run_evaluate(spec, train_dir, out, mode="onshore", run_name="clean")
    clean = pd.read_csv(clean_dir / "metrics.csv")
    assert (clean["excluded_share"] == 0).all()
    assert pd.read_csv(clean_dir / driver.SCORING_EXCLUSIONS_NAME).empty
    assert clean["n_units"].nunique() == 1

    real_get = driver.get_correction
    dropped: list[str] = []

    class DropOneUnit:
        def __init__(self, model):
            self._model = model

        def apply(self, *args, **kwargs):
            sim, cor_cf = self._model.apply(*args, **kwargs)
            unit = [c for c in cor_cf.columns if c != "time"][0]
            dropped.append(str(unit))
            cor_cf = cor_cf.copy()
            cor_cf[unit] = np.nan
            return sim, cor_cf

    monkeypatch.setattr(driver, "get_correction", lambda name: DropOneUnit(real_get(name)))
    with pytest.warns(UserWarning, match="excluded from every variant's score"):
        eval_dir = run_evaluate(spec, train_dir, out, mode="onshore", run_name="nan")

    metrics = pd.read_csv(eval_dir / "metrics.csv")
    unc = metrics[metrics["variant"] == "uncorrected"].iloc[0]
    cor = metrics[metrics["variant"] == "affine-wind"].iloc[0]
    n_clean = int(clean["n_units"].iloc[0])
    assert unc["n_units"] == cor["n_units"] == n_clean - 1
    assert unc["n_samples"] == cor["n_samples"]
    assert (metrics["excluded_share"] > 0).all()

    exclusions = pd.read_csv(eval_dir / driver.SCORING_EXCLUSIONS_NAME)
    assert set(exclusions["ID"].astype(str)) == {dropped[0]}
    assert set(exclusions["missing_in"]) == {"fixed_2"}
    assert set(exclusions["scope"]) == {"fleet"}

    manifest = json.loads((eval_dir / "run_manifest.json").read_text())
    record = manifest["common_row_scoring"]["fleet"]
    assert record["units_wholly_excluded"] == [dropped[0]]
    assert record["n_rows_excluded"] == len(exclusions)
