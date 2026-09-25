"""Pin the fixed-slice factors of two real fits before the offset search changes.

Issue #18 found that the offset search leaves converged fits up to 1e-4 in
capacity factor from their root, and that its residual test then sends some of
them to a fallback that lands elsewhere. Replacing the search will move every
factor a little. These cases pin the factors as the search produces them, so
the change's effect on each cluster is measured rather than assumed: the DK
scorecard row's configuration (k=100) and the CL row's (k=10), both on the
fixed slice alone, and two country-level national rows (FR k=10, ES k=4),
which reach the joint national fit. They are change detectors, not guards (see
CONTRIBUTING.md): the fixtures were recorded on main before the bracketed
search, and re-recorded in the commit that introduced it.

Each case trains in its own process with the input root its row runs on, and
compares ``factors_fixed_<k>.csv`` byte for byte with the recorded file. The
inputs are local, so the cases skip where they are absent, which includes CI.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PINS = Path(__file__).resolve().parent / "data" / "pins" / "offset_fits"

pytestmark = [pytest.mark.realdata, pytest.mark.slow]

# case: (scorecard config, input root, cluster count, ERA5 directory the config reads)
CASES = {
    "dk_k100": ("configs/regions/scorecard/dk_k100.toml", "input/combined", 100, "era5/EU_2026-09"),
    "cl_k10": ("configs/regions/scorecard/cl_k10.toml", "input", 10, "era5/CL"),
    # Country-level national fits with more than one cluster, added 2026-09-25
    # when the router was found sending them to the per-cluster solver
    # (ac26f6a): until then no pin reached the joint fit, so the realdata
    # stamp passed while every such factor could move. Re-recorded the same day
    # when the joint fit's stopping tolerances were tightened: the ES offsets
    # had stayed at the zero start (about 2e-5 m/s) and moved to 0.003 to 0.011
    # m/s; the FR offsets moved by at most 5.3e-4 m/s; scalars and accepted
    # years unchanged in both. Moved to input/combined and re-recorded the same
    # day, when a country run on a curve the library lacks started to be
    # refused: the grids name licensed Vestas curves, and until then every unit
    # ran on the bundled 100 kW fallback.
    "fr_country_k10": (
        "configs/regions/scorecard/fr_country.toml",
        "input/combined",
        10,
        "era5/EU_2026-09",
    ),
    "es_country_k4": (
        "configs/regions/scorecard/es_country.toml",
        "input/combined",
        4,
        "era5/EU_2026-09",
    ),
}

TRAIN = """
import dataclasses, sys
from pyvwf.harness import driver
from pyvwf.harness.regions import load_region
spec = dataclasses.replace(load_region(sys.argv[1]), time_slices=("fixed",))
print(driver.run_train(spec, sys.argv[2], run_name="pin"))
"""


@pytest.mark.parametrize("case", list(CASES))
def test_fixed_slice_factors_are_as_recorded(case, tmp_path):
    config, root, k, era5 = CASES[case]
    if not (ROOT / root / era5).is_dir():
        pytest.skip(f"{root}/{era5} is local only")
    env = dict(os.environ, PYVWF_INPUT=root, PYTHONPATH="src", PYVWF_OFFSET_WORKERS="0")
    done = subprocess.run(
        [sys.executable, "-c", TRAIN, config, str(tmp_path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    run_dir = Path(done.stdout.strip().splitlines()[-1])
    got = (run_dir / f"factors_fixed_{k}.csv").read_bytes()
    assert got == (PINS / f"{case}_factors_fixed_{k}.csv").read_bytes()
