"""Pin the GWPT loading and the plant-name normalisers before they are promoted.

The Global Wind Power Tracker (GWPT) is read, filtered and joined in several
places: ``scripts/process/cammesa_ar.py`` and ``cen_cl.py`` (each with its own
``norm`` and country filter), ``scripts/process/windstats.py`` (an inline
filter), and ``scripts/region_tools/weight_country_grid_points.py``
(``load_gwpt`` and ``fleet_for``, which ``repair_country_capacity.py`` imports
through ``sys.path``). ``vwf.datasets.windstats._norm`` and
``vwf.datasets.aemo_au.normalise_farm_name`` normalise names too. These are
not one function: the filters differ in whether they strip the country string,
and the normalisers drop different words. Promotion must keep each variant, so
each is pinned as it behaves today.

Two layers:

- the four normalisers on every plant name in the tracked curated tables plus
  a few edge cases, against ``tests/data/pins/gwpt/normalised_names.csv``;
  these run in CI;
- the filters on the real GWPT workbook, by sha256, and the Argentina and
  Chile processing chains end to end against the production files. The
  workbook, the raw downloads and openpyxl are local, so this layer skips in
  CI.

The windstats filter is inline in that script's ``main``, and the WindStats
source it runs on is confidential and not on this machine. Its pin is the
script's own expression, copied into ``_windstats_filter`` below and hashed on
the workbook at 51807f8. A promoted filter must reproduce that hash.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from vwf.datasets.aemo_au import normalise_farm_name
from vwf.datasets.windstats import _norm as windstats_norm

ROOT = Path(__file__).resolve().parents[1]
CURATION = ROOT / "configs" / "curation"
PINS = Path(__file__).resolve().parent / "data" / "pins" / "gwpt"
GWPT = ROOT / "input" / "reference" / "gwpt" / "Global-Wind-Power-Tracker-February-2026.xlsx"
PRODUCTION = ROOT / "input" / "observations" / "turbine"


def _load(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str((ROOT / relpath).parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


ar = _load("scripts/process/cammesa_ar.py", "process_cammesa_ar")
cl = _load("scripts/process/cen_cl.py", "process_cen_cl")
weights = _load("scripts/region_tools/weight_country_grid_points.py", "weight_country_grid_points")

NORMALISERS = {
    "cammesa_ar": ar.norm,
    "cen_cl": cl.norm,
    "windstats": windstats_norm,
    "aemo_au": normalise_farm_name,
}

EDGE_CASES = [
    "Parque Eólico La Castellana II",
    "PE Los Meandros III",
    "Wind Farm del Sur IV",
    "Rawson (Genneia) S.A.",
    "  Ñandú windpark  ",
    "Hornsdale Wind Farm Stage 2",
    "Macarthur WF (AGL)",
    "",
]


def curated_names() -> list[str]:
    names = []
    for table in ("ar_turbine_specs", "cl_turbine_specs", "au_turbine_models", "nz_wind_farms"):
        names += pd.read_csv(CURATION / f"{table}.csv")["site_name"].astype(str).tolist()
    return names + EDGE_CASES


def normalised_table() -> pd.DataFrame:
    names = curated_names()
    return pd.DataFrame({"name": names,
                         **{k: [f(n) for n in names] for k, f in NORMALISERS.items()}})


def test_normalisers_on_curated_names():
    got = normalised_table()
    want = pd.read_csv(PINS / "normalised_names.csv", keep_default_na=False)
    pd.testing.assert_frame_equal(got, want)


# --------------------------------------------------------------------------
# The real workbook and the processing chains (local only)
# --------------------------------------------------------------------------

needs_gwpt = pytest.mark.skipif(
    not GWPT.is_file() or importlib.util.find_spec("openpyxl") is None,
    reason="the GWPT workbook and openpyxl are local only")


def _digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv().encode()).hexdigest()


def _windstats_filter(g: pd.DataFrame, country: str) -> pd.DataFrame:
    # Verbatim from scripts/process/windstats.py main() at 51807f8.
    return g[(g["Country/Area"].astype(str).str.strip() == country)
             & (g["Status"].astype(str).str.lower() == "operating")]


def _filter_pins() -> dict[str, str]:
    table = pd.read_csv(PINS / "filter_sha256.csv")
    return dict(zip(table["case"], table["sha256"]))


def filter_results(gwpt: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every filtered frame the pins cover, keyed by case name."""
    out = {
        "gwpt_argentina": ar.gwpt_argentina(GWPT),
        "gwpt_chile": cl.gwpt_chile(GWPT),
    }
    for cc, country in (("ES", "Spain"), ("SE", "Sweden"), ("FI", "Finland")):
        out[f"windstats_{cc}"] = _windstats_filter(gwpt, country)
    for code in sorted(weights.GWPT_COUNTRY):
        for year in (None, 2015, 2019, 2023):
            out[f"fleet_for_{code}_{year}"] = weights.fleet_for(gwpt, code, year)
    return out


@needs_gwpt
def test_gwpt_filters_on_the_real_workbook():
    pins = _filter_pins()
    got = {k: _digest(v) for k, v in filter_results(weights.load_gwpt(GWPT)).items()}
    assert set(got) == set(pins)
    changed = sorted(k for k in pins if got[k] != pins[k])
    assert not changed, changed


def _process(script: str, out: Path, root: str) -> None:
    env = dict(os.environ, PYVWF_INPUT=root, PYTHONPATH="src")
    done = subprocess.run([sys.executable, f"scripts/process/{script}.py", "--out", str(out)],
                          cwd=ROOT, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]


@needs_gwpt
@pytest.mark.parametrize("script, code", [("cammesa_ar", "AR"), ("cen_cl", "CL")])
def test_processing_chain_reproduces_production(script, code, tmp_path):
    """Process, then apply the turbine specs, as the runbook does.

    apply_turbine_specs.py edits the metadata in place under the input root,
    so it runs against a temporary root holding the processed file and a link
    to the open library, which is the library these two rows run on.
    """
    lc = code.lower()
    raw = {"AR": ROOT / "input" / "raw" / "cammesa", "CL": ROOT / "input" / "raw" / "cen"}[code]
    if not raw.is_dir() or not (PRODUCTION / code / f"{lc}_md.csv").is_file():
        pytest.skip("raw downloads or production files absent")
    root = tmp_path / "root"
    out = root / "observations" / "turbine" / code
    _process(script, out, "input")
    (root / "reference").symlink_to(ROOT / "input" / "reference")
    env = dict(os.environ, PYVWF_INPUT=str(root), PYTHONPATH="src")
    done = subprocess.run(
        [sys.executable, "scripts/region_tools/apply_turbine_specs.py", code,
         "--specs", f"configs/curation/{lc}_turbine_specs.csv"],
        cwd=ROOT, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    for produced in sorted(out.iterdir()):
        if produced.name.endswith(".bak.csv"):
            continue
        recorded = PRODUCTION / code / produced.name
        assert recorded.read_bytes() == produced.read_bytes(), produced.name
    shutil.rmtree(root)
