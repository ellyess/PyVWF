"""A run's training-fleet files are checked to agree before one is read.

Two analysis scripts took the first ``train_turb_info_*.csv`` of a sorted glob
(AGENTS.md: "never sort a glob and take the last entry"). A run writes one file
per cluster count, and on every run on disk they differ only in ``cluster``, so
the pick was harmless; nothing would have noticed the day they did not.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
_SPEC = importlib.util.spec_from_file_location(
    "curve_match_audit_under_test", ROOT / "scripts" / "analysis" / "curve_match_audit.py"
)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def write(path, clusters, models=("A", "B")):
    pd.DataFrame({"ID": ["u1", "u2"], "model": list(models), "cluster": clusters}).to_csv(
        path, index=False
    )
    return path


def test_files_differing_only_in_cluster_give_the_first(tmp_path):
    a = write(tmp_path / "train_turb_info_1.csv", [0, 0])
    b = write(tmp_path / "train_turb_info_10.csv", [0, 1])
    fleet = audit.training_fleet([b, a])
    assert fleet["cluster"].tolist() == [0, 0]  # _1 sorts first, as the old pick


def test_files_disagreeing_on_the_fleet_are_refused(tmp_path):
    a = write(tmp_path / "train_turb_info_1.csv", [0, 0])
    b = write(tmp_path / "train_turb_info_10.csv", [0, 1], models=("A", "C"))
    with pytest.raises(ValueError, match="disagree"):
        audit.training_fleet([a, b])


def test_no_file_is_refused():
    with pytest.raises(FileNotFoundError):
        audit.training_fleet([])
