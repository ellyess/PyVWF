"""Pin every recorded paired-bootstrap output before the bootstrap is promoted.

Three scripts compute paired bootstrap intervals: ``baseline_bootstrap.py``,
``unit_concentration.py`` and ``common_row_rescore.py``. Their outputs are the
evidence behind correction notices in the scorecard. Two more,
``roughness_treatment_study.py`` and ``eu_rerun_compare.py``, were pinned here
until their studies were archived at tag ``archive/research-2026-10-27``.
Promoting the resampling into ``pyvwf`` must leave every one of those
files byte for byte as it is.

Each case reruns one script for one row, in its own process with the input
root the row was run on, and compares every file it writes with the recorded
file under ``output/``. The inputs (the run frames under the git-ignored
``output/`` and confidential observations under ``input/``) are local, so the
cases skip where they are absent, which includes CI.

Before this file was written, every case was run at 51807f8 and every file
matched.

The files were recorded under pandas 2.3.3. Each file is compared as a table:
non-numeric columns exactly, numeric ones to 1e-15. Byte comparison was used
under pandas 2 until 2026-09-20, when it began failing the UK rows: d68b542
put every weighted mean behind one helper, and the reassociated sum moved six
diagnostic columns of the UK row in their last bit, at most 1.7e-16 absolute
and 3e-15 relative. No figure any of these files reports changed, at any
precision the scorecard or a findings document quotes. A detector that fires
on the last bit of a float reports the order of a summation, not a result, so
the tolerance that pandas 3 already needed for the same reason now applies to
both.

1e-15 is far below anything reportable and far above reassociation noise:
these files carry capacity factors and their errors, quoted to three or four
decimals, so a real movement is 1e-4 at the smallest and clears the tolerance
by eleven orders of magnitude.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from support import ROOT

pytestmark = pytest.mark.realdata

SCRIPTS = ROOT / "scripts"
STUDY = ROOT / "output" / "curve_library_study_2026-09-11"
BACKFILL = ROOT / "output" / "validation" / "curve_resolution_backfill_2026-09-11"

# The input root each row's manifest records.
COMBINED = {"DE", "DK", "UK", "US", "BR", "AU-NEM", "NZ"}
ALL_ROWS = [
    "BE",
    "ES",
    "FR",
    "IE",
    "IT",
    "NO",
    "PT",
    "SE",
    "DE",
    "DK",
    "UK",
    "US",
    "BR",
    "AU-NEM",
    "NZ",
    "CL",
    "AR",
]
TURBINE_ROWS = ["DE", "DK", "UK", "US", "BR", "AU-NEM", "NZ", "CL", "AR"]

# Rows whose recorded runs no longer reproduce from current inputs, and so are
# not rerun here. These scripts rescore a recorded run with the fleet loaded
# from the input root today, and an evaluate run does not save its fleet. FR's
# grids were rebuilt on 2026-09-29 on the corrected fleet register (acfbbfa),
# which took 41 MW off its 2023 grid, and every FR case then failed its
# script's own check that the rescored run reproduces metrics.csv: by 3.5e-5
# to 7.2e-5 across the baseline bootstrap's RMSE and MBE, and by 7.1e-5 in
# uncorrected RMSE, the first figure the roughness and EU re-run scripts check
# (the common-row rescore reports no figure). The recorded FR runs are on the
# 100 kW fallback curve, whose figures the scorecard's third notice of
# 2026-09-25 already superseded, so they are kept in output/ as the record of
# what those studies produced, and no longer rerun. On 2026-10-16 every country
# grid was rebuilt again, with offshore farms on their own points (fc8c7f5) and
# Hywind Tampen out of Norway's fleet (6b3e6d9), and the seven other country
# rows failed the same way: all eighteen of their cases here, at their
# scripts' own reproduction check (Belgium's uncorrected RMSE by 0.075, its
# 2,256 MW of offshore capacity now simulated at sea). Their recorded runs are
# fallback-curve figures too, so they leave on the same terms, and the
# turbine-level rows go on checking the scripts.
REGRIDDED = {"BE", "ES", "FR", "IE", "IT", "NO", "PT", "SE"}
RESCORED_ROWS = [c for c in ALL_ROWS if c not in REGRIDDED]


def _root(code: str) -> str:
    return "input/combined" if code in COMBINED else "input"


def _run(script: str, args: list[str], code: str) -> None:
    """Run ``scripts/<script>`` for one row, as its own process."""
    env = dict(
        os.environ,
        PYVWF_INPUT=_root(code),
        PYTHONPATH="src",
    )
    done = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]


def _same(produced: Path, recorded: Path, names: list[str]) -> None:
    for name in names:
        got, want = pd.read_csv(produced / name), pd.read_csv(recorded / name)
        pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=0, atol=1e-15, obj=name)


def _need(*paths: Path) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"local run records absent: {missing[0]}")


@pytest.mark.parametrize("code", RESCORED_ROWS)
def test_baseline_bootstrap_reproduces(code, tmp_path):
    recorded = STUDY / "baseline_bootstrap"
    _need(BACKFILL / code, recorded / f"{code}_bootstrap.csv")
    _run("analysis/baseline_bootstrap.py", [code, str(tmp_path)], code)
    _same(tmp_path, recorded, [f"{code}_bootstrap.csv", f"{code}_reproduction.csv"])


@pytest.mark.parametrize("code", TURBINE_ROWS)
def test_unit_concentration_reproduces(code, tmp_path):
    recorded = STUDY / "baseline_bootstrap"
    _need(BACKFILL / code, recorded / f"{code}_concentration.csv")
    _run("analysis/unit_concentration.py", [code, str(tmp_path)], code)
    _same(tmp_path, recorded, [f"{code}_concentration.csv", f"{code}_top5_units.csv"])


@pytest.mark.parametrize("code", RESCORED_ROWS)
def test_common_row_rescore_reproduces(code, tmp_path):
    recorded = STUDY / "common_row_rescore"
    _need(BACKFILL / code, recorded / f"{code}_rescore.csv")
    _run("analysis/common_row_rescore.py", [code, str(tmp_path)], code)
    names = [f"{code}_rescore.csv"]
    if (recorded / f"{code}_common_rows_bootstrap.csv").exists():
        names.append(f"{code}_common_rows_bootstrap.csv")
    _same(tmp_path, recorded, names)
