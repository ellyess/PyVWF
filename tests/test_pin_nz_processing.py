"""Pin the New Zealand processing before it moves into the package.

The capacity history, the build mask and the Generation_MD melt that produce
the NZ inputs were written in ``scripts/process/emi_nz.py``, and they are the
production path: the register-based versions in ``vwf.datasets.emi_nz`` are
tested but unused. These tests pinned what that code produced before it moved
into ``vwf.datasets.emi_nz``, so the move could only move code, not change
output. Only the names the unit layer calls changed with the move.

Three layers:

- the three helpers on the committed curated tables, against fixtures in
  ``tests/data/pins/nz/``;
- the whole script on synthetic ``Generation_MD`` files, which exercises the
  melt loop (fuel-code spellings, header casing, Gen_Code case, the known
  absentee, an unmapped code), against the same fixture directory;
- the whole script on the real EMI downloads, against the sha256 of the
  production files, when those downloads are present. They are not
  redistributable, so this layer skips in CI.
"""
from __future__ import annotations

import hashlib
import importlib.util
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "process" / "emi_nz.py"
CURATION = ROOT / "configs" / "curation"
PINS = Path(__file__).resolve().parent / "data" / "pins" / "nz"
RAW_EMI = ROOT / "input" / "raw" / "emi"

# sha256 of the production files a run on the real downloads wrote on
# 2026-07-23 (input/observations/turbine/NZ/), reproduced byte for byte by a
# rerun at 51807f8. nz_md.csv is not pinned here: its curve keys depend on the
# input root's curve library, not on the NZ logic.
PRODUCTION_SHA256 = {
    "nz_obs.csv": "651ffcc8d27879fa0965a1f10f8a991ef63982d3bd5c27c1af6fe29a638329f1",
    "nz_build_mask.csv": "cef75b4ba5ce2696d132e64f77de42fe0d6ac9165f2b82965abfe5032c9f4bbf",
}


def _load_script():
    spec = importlib.util.spec_from_file_location("process_emi_nz", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nz = _load_script()

import vwf.datasets.emi_nz as emi  # noqa: E402  (the promoted helpers)


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(PINS / name, dtype={"ID": str})


# --------------------------------------------------------------------------
# The helpers, on the committed curated tables
# --------------------------------------------------------------------------

def test_gen_code_map_on_the_curated_farms():
    farms, _, _ = emi.load_curated_tables(CURATION)
    got = pd.DataFrame(sorted(emi.gen_code_map(farms).items()), columns=["gen_code", "ID"])
    pd.testing.assert_frame_equal(got, _read("gen_code_map.csv"))


def test_capacity_history_on_the_curated_tables():
    farms, stages, _ = emi.load_curated_tables(CURATION)
    got = emi.capacity_history_from_curation(farms, stages)
    want = _read("capacity_history.csv")
    want["effective_from"] = pd.to_datetime(want["effective_from"])
    pd.testing.assert_frame_equal(got, want, check_dtype=False)
    assert got["capacity"].dtype.kind in "if"


def test_mask_from_windows_on_the_curated_windows():
    _, _, windows = emi.load_curated_tables(CURATION)
    pd.testing.assert_frame_equal(emi.mask_from_windows(windows), _read("mask_from_windows.csv"))


# --------------------------------------------------------------------------
# The whole script, on synthetic Generation_MD files
# --------------------------------------------------------------------------

def _generation_md(year: int, month: int, rows: list[tuple[str, str, int]],
                   *, date_header: str = "Trading_Date") -> pd.DataFrame:
    """One month of synthetic Generation_MD.

    ``rows`` is (Gen_Code, Fuel_Code, days of data from the 1st). Every
    trading period of a normal 48-period day gets a deterministic kWh value;
    TP49 and TP50 are empty, as in the real files.
    """
    days = pd.date_range(f"{year}-{month:02d}-01", periods=pd.Period(f"{year}-{month:02d}").days_in_month)
    records = []
    for k, (gen_code, fuel, n_days) in enumerate(rows):
        for d in days[:n_days]:
            values = {f"TP{tp}": 1000.0 * (k + 1) + 10.0 * d.day + tp for tp in range(1, 49)}
            values.update({"TP49": np.nan, "TP50": np.nan})
            records.append({
                "Site_Code": "SYN", "POC_Code": "SYN0000", "Nwk_Code": "SYN",
                "Gen_Code": gen_code, "Fuel_Code": fuel, "Tech_Code": fuel,
                date_header: d.strftime("%Y-%m-%d"), **values,
            })
    return pd.DataFrame.from_records(records)


def _write_raw(raw: Path) -> None:
    raw.mkdir()
    # January: both fuel spellings, a mixed-case Gen_Code, a hydro row the
    # fuel filter must drop, and the documented absentee.
    _generation_md(2021, 1, [
        ("twf_12", "Wind", 31),
        ("TE_APITI", "WIN", 31),
        ("aratiatia", "Hydro", 31),
        ("mahinerangi", "Wind", 31),
    ]).to_csv(raw / "202101_Generation_MD.csv", index=False)
    # February: the lower-case header the 2019 files use, and a farm with too
    # few days to pass the coverage screen.
    _generation_md(2021, 2, [
        ("twf_12", "Wind", 28),
        ("te_apiti", "Wind", 20),
    ], date_header="Trading_date").to_csv(raw / "202102_Generation_MD.csv", index=False)


def _run_script(argv: list[str]) -> None:
    old = sys.argv
    sys.argv = ["emi_nz.py", *argv]
    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    finally:
        sys.argv = old


def test_script_on_synthetic_generation_md(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    _write_raw(raw)
    _run_script(["--raw", str(raw), "--configs", str(CURATION),
                 "--years", "2021", "2021", "--out", str(out)])

    pd.testing.assert_frame_equal(
        pd.read_csv(out / "nz_obs.csv", dtype={"ID": str}), _read("synthetic_nz_obs.csv"))
    pd.testing.assert_frame_equal(
        pd.read_csv(out / "nz_build_mask.csv", dtype={"ID": str}), _read("mask_from_windows.csv"))

    # The metadata contract, minus the curve keys, which depend on the input
    # root's curve library rather than on the NZ logic.
    md = pd.read_csv(out / "nz_md.csv", dtype={"ID": str}).drop(columns=["model", "model_source"])
    pd.testing.assert_frame_equal(md, _read("synthetic_nz_md.csv"))


def test_script_refuses_an_unmapped_wind_gen_code(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _generation_md(2021, 1, [("brand_new_farm", "Wind", 31)]).to_csv(
        raw / "202101_Generation_MD.csv", index=False)
    with pytest.raises(SystemExit, match="brand_new_farm"):
        _run_script(["--raw", str(raw), "--configs", str(CURATION),
                     "--years", "2021", "2021", "--out", str(tmp_path / "out")])


# --------------------------------------------------------------------------
# The whole script, on the real EMI downloads (local only)
# --------------------------------------------------------------------------

@pytest.mark.skipif(not any(RAW_EMI.glob("*_Generation_MD.csv")),
                    reason="the EMI Generation_MD downloads are local only")
def test_script_reproduces_the_production_files(tmp_path):
    out = tmp_path / "out"
    _run_script(["--raw", str(RAW_EMI), "--configs", str(CURATION), "--out", str(out)])
    for name, digest in PRODUCTION_SHA256.items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest, name
