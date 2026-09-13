"""The paired comparison driver's self-check (scripts/analysis/eu_rerun_compare.py).

Before it resamples anything, the driver rebuilds each run's frames and refuses
unless the rebuilt point metric reproduces that run's own ``metrics.csv`` to
1e-12. The rebuild reconstructs every row, but a run scores only the rows all
of its variants can score and writes the rest to ``scoring_exclusions.csv``. So
the check has to apply each run's own exclusions before comparing, or it
refuses a sound run for using the scoring convention it was written under.
"""
import importlib.util
from pathlib import Path

import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "eu_rerun_compare",
    Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "eu_rerun_compare.py",
)
compare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare)

KEYS = ["ID", "year", "month"]


def frame():
    return pd.DataFrame({"ID": ["a", "a", "b"], "year": [2022, 2022, 2022],
                         "month": [1, 2, 1], "cf_sim": [0.3, 0.4, 0.5],
                         "cf_obs": [0.31, 0.39, 0.52]})


def exclusions(tmp_path, rows):
    pd.DataFrame(rows, columns=["scope", "ID", "year", "month"]).to_csv(
        tmp_path / "scoring_exclusions.csv", index=False)
    return tmp_path


def test_a_run_that_excluded_nothing_keeps_every_row(tmp_path):
    got = compare.drop_run_exclusions(frame(), tmp_path, KEYS, "fleet")
    assert len(got) == 3


def test_the_rows_a_run_excluded_are_dropped(tmp_path):
    ev = exclusions(tmp_path, [("fleet", "a", 2022, 2)])
    got = compare.drop_run_exclusions(frame(), ev, KEYS, "fleet")
    assert list(zip(got["ID"], got["month"])) == [("a", 1), ("b", 1)]


def test_an_exclusion_for_another_scope_is_not_applied(tmp_path):
    """A zonal run excludes rows per zone as well; the fleet score is not
    restricted by the per-zone scope's exclusions."""
    ev = exclusions(tmp_path, [("per-zone", "a", 2022, 2)])
    assert len(compare.drop_run_exclusions(frame(), ev, KEYS, "fleet")) == 3


def test_ids_that_differ_only_in_type_still_match(tmp_path):
    """The register writes an integer ID and the exclusions file reads it back
    as one; the frame carries a string. Comparing them raw would drop nothing
    and the check would then refuse the run it was meant to admit."""
    numeric = frame().assign(ID=[1, 1, 2])
    ev = exclusions(tmp_path, [("fleet", 1, 2022, 2)])
    got = compare.drop_run_exclusions(numeric, ev, KEYS, "fleet")
    assert len(got) == 2 and list(got["month"]) == [1, 1]


def test_the_returned_frame_keeps_its_original_values(tmp_path):
    """Dropping rows must not leave the keys coerced to strings behind it."""
    numeric = frame().assign(ID=[1, 1, 2])
    ev = exclusions(tmp_path, [("fleet", 1, 2022, 2)])
    got = compare.drop_run_exclusions(numeric, ev, KEYS, "fleet")
    assert got["ID"].tolist() == [1, 2]
