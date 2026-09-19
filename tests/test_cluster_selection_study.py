"""The cluster selection runner (scripts/studies/method-cluster-selection/cluster_selection_study.py).

The two tests that matter pin the defect that contaminated its first run: a run
directory keyed on the region code alone, so two fleet modes of one region
collided, and one aggregate file per invocation, so a per-process loop left
only the last row on disk.
"""

import importlib.util
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "cluster_selection_study",
    REPO / "scripts" / "studies" / "method-cluster-selection" / "cluster_selection_study.py",
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_the_run_name_carries_the_fleet_mode():
    """_run_dir keys on the region code and the run name and nothing else, so
    without the mode here DK onshore and DK offshore share a directory."""
    assert study.run_tag("onshore", "fold-2016") != study.run_tag("offshore", "fold-2016")
    assert study.run_tag("onshore", "fold-2016") == "onshore-fold-2016"


def test_forward_chaining_never_validates_on_a_year_it_trained_on():
    plan = study.folds((2015, 2019))
    assert plan == [
        ((2015, 2015), 2016),
        ((2015, 2016), 2017),
        ((2015, 2017), 2018),
        ((2015, 2018), 2019),
    ]
    for (first, last), year in plan:
        assert year > last, "a fold trained on a year at or after the one it validates"


def test_a_single_year_window_has_no_folds():
    assert study.folds((2019, 2019)) == []


def scores(values: dict[int, list[float]]) -> pd.DataFrame:
    rows = []
    for k, per_fold in values.items():
        for i, v in enumerate(per_fold):
            rows.append({"num_clu": k, "fold_year": 2016 + i, "rmse": v, "mae": v})
    return pd.DataFrame(rows)


def test_the_rule_takes_the_smallest_count_inside_one_standard_error():
    """k=2 minimises, but k=1 is within a standard error of it, so k=1 wins.
    This is Denmark offshore's case, where the rule overrode a choice that
    would have matched the chapter."""
    frame = scores({1: [0.101, 0.103, 0.099], 2: [0.100, 0.102, 0.098], 10: [0.200, 0.202, 0.198]})
    selected, best, threshold = study.one_standard_error(frame, "rmse")
    assert best == 2 and selected == 1
    assert threshold > frame[frame.num_clu == 2]["rmse"].mean()


def test_a_count_outside_the_interval_is_not_taken_however_small():
    frame = scores({1: [0.50, 0.50, 0.50], 2: [0.10, 0.10, 0.10], 10: [0.11, 0.11, 0.11]})
    selected, best, _ = study.one_standard_error(frame, "rmse")
    assert best == 2 and selected == 2, "a large gap must not be crossed"


def test_one_fold_collapses_the_interval_to_the_minimum():
    """The sample standard deviation is undefined on one fold and comes back
    NaN. The rule must then pick the minimising count, not raise or take the
    smallest count in the grid."""
    frame = scores({1: [0.20], 2: [0.10], 10: [0.11]})
    selected, best, _ = study.one_standard_error(frame, "rmse")
    assert best == 2 and selected == 2


def test_the_grid_is_capped_at_the_fleet_size():
    assert study.grid_for("offshore", 4) == (1, 2, 3)
    assert study.grid_for("onshore", 30) == (1, 10, 25)
    assert study.grid_for("onshore", 10_000) == study.ONSHORE_GRID
