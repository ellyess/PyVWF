"""Pin every recorded paired-bootstrap output before the bootstrap is promoted.

Five scripts compute paired bootstrap intervals, each with its own copy of the
resampling: ``baseline_bootstrap.py``, ``unit_concentration.py``,
``common_row_rescore.py``, ``roughness_treatment_study.py`` and
``eu_rerun_compare.py``. Their outputs are the evidence behind correction
notices in the scorecard and behind the roughness and European re-run
findings. Promoting the resampling into ``vwf`` must leave every one of those
files byte for byte as it is.

Each case reruns one script for one row, in its own process with the input
root the row was run on, and compares every file it writes with the recorded
file under ``output/``. The inputs (the run frames under the git-ignored
``output/`` and confidential observations under ``input/``) are local, so the
cases skip where they are absent, which includes CI.

Before this file was written, every case was run at 51807f8 and every file
matched. The eu_rerun_compare invocation is not recorded anywhere; the one
below is the one that reproduces all eight recorded comparisons exactly.

The files were recorded under pandas 2.3.3, and under pandas 2 the comparison
is byte for byte. Under pandas 3 the turbine-level rows, whose sums run over
units, round differently in the last place (differences near 1e-17, such as a
confidence bound of ...835 against ...834), so under pandas 3 each file is
compared as a table: non-numeric columns exactly, numeric ones to 1e-15.
"""
from __future__ import annotations

import filecmp
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytestmark = [pytest.mark.realdata, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
STUDY = ROOT / "output" / "curve_library_study_2026-09-11"
BACKFILL = ROOT / "output" / "validation" / "curve_resolution_backfill_2026-09-11"
ROUGHNESS = ROOT / "output" / "roughness_treatment_2026-09-12"
EU_RERUN = ROOT / "output" / "eu_rerun_2026-09-12"

# The input root each row's manifest records.
COMBINED = {"DE", "DK", "UK", "US", "BR", "AU-NEM", "NZ"}
ALL_ROWS = ["BE", "ES", "FR", "IE", "IT", "NO", "PT", "SE",
            "DE", "DK", "UK", "US", "BR", "AU-NEM", "NZ", "CL", "AR"]
TURBINE_ROWS = ["DE", "DK", "UK", "US", "BR", "AU-NEM", "NZ", "CL", "AR"]


def _root(code: str) -> str:
    return "input/combined" if code in COMBINED else "input"


def _run(script: str, args: list[str], code: str, *, extra_path: str = "") -> None:
    """Run ``scripts/<script>`` for one row, as its own process."""
    env = dict(os.environ, PYVWF_INPUT=_root(code),
               PYTHONPATH=os.pathsep.join(p for p in ("src", extra_path) if p))
    done = subprocess.run([sys.executable, str(SCRIPTS / script), *args], cwd=ROOT,
                          env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]


PANDAS_MAJOR = int(pd.__version__.split(".")[0])


def _same(produced: Path, recorded: Path, names: list[str]) -> None:
    for name in names:
        if PANDAS_MAJOR < 3:
            assert filecmp.cmp(produced / name, recorded / name, shallow=False), name
            continue
        got, want = pd.read_csv(produced / name), pd.read_csv(recorded / name)
        pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=0, atol=1e-15,
                                      obj=name)


def _need(*paths: Path) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"local run records absent: {missing[0]}")


@pytest.mark.parametrize("code", ALL_ROWS)
def test_baseline_bootstrap_reproduces(code, tmp_path):
    recorded = STUDY / "baseline_bootstrap"
    _need(BACKFILL / code, recorded / f"{code}_bootstrap.csv")
    _run("analysis/baseline_bootstrap.py", [code, str(tmp_path)], code)
    _same(tmp_path, recorded, [f"{code}_bootstrap.csv", f"{code}_reproduction.csv"])


@pytest.mark.parametrize("code", TURBINE_ROWS)
def test_unit_concentration_reproduces(code, tmp_path):
    recorded = STUDY / "baseline_bootstrap"
    _need(BACKFILL / code, recorded / f"{code}_concentration.csv")
    _run("studies/scorecard/unit_concentration.py", [code, str(tmp_path)], code)
    _same(tmp_path, recorded, [f"{code}_concentration.csv", f"{code}_top5_units.csv"])


@pytest.mark.parametrize("code", ALL_ROWS)
def test_common_row_rescore_reproduces(code, tmp_path):
    recorded = STUDY / "common_row_rescore"
    _need(BACKFILL / code, recorded / f"{code}_rescore.csv")
    _run("analysis/common_row_rescore.py", [code, str(tmp_path)], code)
    names = [f"{code}_rescore.csv"]
    if (recorded / f"{code}_common_rows_bootstrap.csv").exists():
        names.append(f"{code}_common_rows_bootstrap.csv")
    _same(tmp_path, recorded, names)


@pytest.mark.parametrize("code", ["DK", "FR"])
def test_roughness_treatment_study_reproduces(code, tmp_path):
    _need(ROUGHNESS / "R0" / code, ROUGHNESS / "R1" / code)
    r0 = next((ROUGHNESS / "R0" / code).glob("evaluate-*"))
    r1 = next((ROUGHNESS / "R1" / code).glob("evaluate-*"))
    _run("studies/method-roughness-treatment/roughness_treatment_study.py",
         [code, str(r0), str(r1), str(tmp_path)], code)
    # DK's recorded files sit in a subdirectory, FR's at the top level.
    recorded = ROUGHNESS / "analysis" / code
    if not recorded.is_dir():
        recorded = ROUGHNESS / "analysis"
    _same(tmp_path, recorded, [f"{code}_roughness_{kind}.csv"
                               for kind in ("comparison", "losses", "excluded_rows")])


@pytest.mark.parametrize("code", ["BE", "DE", "DK", "FR", "IE", "NO", "SE", "UK"])
def test_eu_rerun_compare_reproduces(code, tmp_path):
    _need(BACKFILL / code, EU_RERUN / "new" / code)
    conditions = [f"published={next((BACKFILL / code).glob('evaluate-*-backfill'))}"]
    if (EU_RERUN / "oldfiles_derived" / code).is_dir():
        conditions.append(
            f"oldfiles_derived={next((EU_RERUN / 'oldfiles_derived' / code).glob('evaluate-*'))}")
    conditions.append(f"new={next((EU_RERUN / 'new' / code).glob('evaluate-*'))}")
    _run("analysis/eu_rerun_compare.py", [code, str(tmp_path), *conditions], code,
         extra_path="scripts/analysis")
    _same(tmp_path, EU_RERUN / "analysis",
          [f"{code}_rerun_comparison.csv", f"{code}_rerun_excluded_rows.csv"])
