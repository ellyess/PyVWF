"""Pillar B: synthetic Southern-Hemisphere ground truth through the FULL pipeline.

D2's pre-flight (validation design, docs + progress log): there is no legacy
AU path to bit-diff against, so confidence in the SH/AU numbers is
constructed. This test plants a KNOWN seasonal bias in a synthetic AU-shaped
dataset — reanalysis over-blows by 30% during JJA (austral winter) — and runs
the complete real stack: AEMO-format 5-minute SCADA files on disk →
AEMONemSource (AEST→UTC binning) → harness run_train with an SH-season config
→ run_evaluate on a held-out year.

What it can and cannot prove, stated honestly:

- Within-region training with METEOROLOGICAL seasons is partition-invariant:
  NH and SH mappings group the same four month-sets ({12,1,2}, {3,4,5},
  {6,7,8}, {9,10,11}); only the NAMES differ. So the fit maths cannot be
  hemisphere-wrong. The real SH risks are (a) the labels on the factors
  table — AU's downward correction must sit under "winter" meaning JJA — and
  (b) fit-time/apply-time label consistency: factors trained under SH labels
  but applied through the NH map merge onto the wrong months.
- Test 1 pins (a): the planted JJA bias is recovered UNDER THE RIGHT NAME.
- Test 2 pins the end-to-end value: corrected beats uncorrected on held-out.
- Test 3 pins (b), must-distinguish: applying the SH-trained factors through
  the NH mapping (the exact bug the seasons seam eliminated) must be WORSE
  THAN NO CORRECTION AT ALL — it leaves JJA biased and breaks the previously
  unbiased DJF.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import vwf.wind as wind
from vwf.config import PyVWFPaths
from vwf.data import load_power_curves, val_set
from vwf.harness.driver import run_evaluate, run_train
from vwf.harness.regions import RegionSpec
from vwf.sources.aemo import AEMONemSource

SH_SEASONS = {
    "summer": (12, 1, 2),
    "autumn": (3, 4, 5),
    "winter": (6, 7, 8),
    "spring": (9, 10, 11),
}
NH_SEASONS = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}

YEARS = (2021, 2022, 2023)  # 2021-2022 train (source default window), 2023 held out
TEST_YEAR = 2023
JJA = (6, 7, 8)
BIAS = 1.30  # reanalysis over-blows true winds by 30% during JJA
LATS = np.array([-36.0, -35.0, -34.0])
LONS = np.array([148.0, 149.0, 150.0])
Z0 = 0.05
CAPACITY_KW = 100_000.0  # 100 MW per farm
MODEL = "2019COE_Market_Average_2.6MW_121"

FARMS = pd.DataFrame(
    {
        "ID": ["WFARM1", "WFARM2"],
        "lon": [149.0, 150.0],
        "lat": [-35.0, -34.0],
        "height": [100.0, 100.0],
        "capacity": [CAPACITY_KW, CAPACITY_KW],
        "model": [MODEL, MODEL],
        "type": ["onshore", "onshore"],
    }
)


def _true_daily_wind(rng, days):
    """True wind per day: gentle noise on a base sitting on the curve's rising
    flank, constant within each day so daily-mean == the value and the
    CF conversion commutes through the pipeline's daily resampling."""
    return np.clip(8.0 + rng.normal(0.0, 0.6, size=len(days)), 3.0, None)


def _curve_cf():
    """CF-vs-speed interpolator for the fixture's model, from the same table
    the pipeline loads."""
    curves = load_power_curves()
    speeds = curves[curves.columns[0]].to_numpy(float)
    cf = curves[MODEL].to_numpy(float)
    return lambda ws: np.interp(ws, speeds, cf)


@pytest.fixture(scope="module")
def au_world(tmp_path_factory):
    """Synthetic AU on disk: biased ERA5 + AEMO-format SCADA from TRUE winds."""
    root = tmp_path_factory.mktemp("au")
    rng = np.random.default_rng(7)

    days = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31", freq="D")  # UTC
    true_wind = _true_daily_wind(rng, days)

    # --- ERA5 (UTC, hourly), planted bias: x1.30 during JJA ---------------
    biased = np.where(np.isin(days.month, JJA), true_wind * BIAS, true_wind)
    hours = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31 23:00", freq="h")
    hourly = np.repeat(biased, 24)[: len(hours)]
    field = np.tile(hourly[:, None, None], (1, len(LATS), len(LONS)))
    shear = np.log(10 / Z0) / np.log(100 / Z0)
    era5_dir = root / "era5"
    era5_dir.mkdir()
    xr.Dataset(
        {
            "u100": (("time", "lat", "lon"), field / np.sqrt(2.0)),
            "v100": (("time", "lat", "lon"), field / np.sqrt(2.0)),
            "u10": (("time", "lat", "lon"), field * shear / np.sqrt(2.0)),
            "v10": (("time", "lat", "lon"), field * shear / np.sqrt(2.0)),
        },
        coords={"time": hours, "lat": LATS, "lon": LONS},
    ).to_netcdf(era5_dir / "era5_synthetic_AU.nc")

    # --- AEMO-format SCADA from TRUE (unbiased) winds ----------------------
    # 5-minute AEST timestamps; each maps to its UTC day's true CF. The farms
    # sit exactly on grid nodes, so pipeline interpolation returns the cell
    # value and the only planted difference is the JJA bias.
    to_cf = _curve_cf()
    day_cf = pd.Series(to_cf(true_wind), index=days)
    stamps_aest = pd.date_range(
        f"{YEARS[0]}-01-01 00:00", f"{YEARS[-1]}-12-31 23:55", freq="5min"
    )
    utc_day = (stamps_aest - pd.Timedelta(hours=10)).normalize()
    cf_per_stamp = day_cf.reindex(utc_day).to_numpy()
    keep = ~np.isnan(cf_per_stamp)  # first 10 AEST hours map before the window

    scada = pd.concat(
        [
            pd.DataFrame(
                {
                    "timestamp": stamps_aest[keep],
                    "ID": farm_id,
                    "mw": cf_per_stamp[keep] * (CAPACITY_KW / 1000.0),
                }
            )
            for farm_id in FARMS["ID"]
        ],
        ignore_index=True,
    )

    data_dir = root / "turbine_level_data" / "AU_NEM"
    data_dir.mkdir(parents=True)
    FARMS.to_csv(data_dir / "au_nem_md.csv", index=False)
    scada.to_csv(data_dir / "au_nem_scada.csv", index=False)

    return {"root": root, "day_cf": day_cf}


@pytest.fixture
def au_paths(au_world, monkeypatch):
    root = au_world["root"]
    monkeypatch.setattr(PyVWFPaths, "INPUT_ROOT", root)
    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", root / "turbine_level_data")
    monkeypatch.setattr(PyVWFPaths, "ERA5_DATA", root / "era5")
    return au_world


def au_spec() -> RegionSpec:
    return RegionSpec(
        code="AU-NEM",
        name="Australia (synthetic pillar B)",
        source="aemo-nem",
        obs_level="turbine",
        obs_unit="farm",
        train_years=(2021, 2022),
        test_years=(TEST_YEAR,),
        era5_path="era5",
        bbox=(147.0, 151.0, -37.0, -33.0),
        file_tag="AU",
        correction_model="affine-wind",
        cluster_list=(1,),
        time_slices=("season",),
        seasons=SH_SEASONS,
    )


@pytest.fixture(scope="module")
def trained(au_world, tmp_path_factory):
    """Train + evaluate once through the real driver; shared by the tests."""
    root = au_world["root"]
    # module-scoped, so patch paths manually rather than via the fixture.
    old = (PyVWFPaths.INPUT_ROOT, PyVWFPaths.TURBINE_DATA, PyVWFPaths.ERA5_DATA)
    PyVWFPaths.INPUT_ROOT = root
    PyVWFPaths.TURBINE_DATA = root / "turbine_level_data"
    PyVWFPaths.ERA5_DATA = root / "era5"
    try:
        out = tmp_path_factory.mktemp("runs")
        spec = au_spec()
        train_dir = run_train(spec, out, run_name="b")
        eval_dir = run_evaluate(spec, train_dir, out, run_name="b")
        factors = pd.read_csv(train_dir / "factors_season_1.csv")
        metrics = pd.read_csv(eval_dir / "metrics.csv")
    finally:
        PyVWFPaths.INPUT_ROOT, PyVWFPaths.TURBINE_DATA, PyVWFPaths.ERA5_DATA = old
    return {"factors": factors, "metrics": metrics, "spec": spec}


def test_planted_jja_bias_recovered_under_the_winter_label(trained):
    """(a) Labels: the downward correction must sit under 'winter' == JJA."""
    factors = trained["factors"].set_index("season")
    assert set(factors.index) == {"winter", "spring", "summer", "autumn"}

    # JJA is over-blown by 30%: the winter-labelled scalar must pull hard down.
    assert factors.loc["winter", "scalar"] < 0.90
    # DJF (summer label) is unbiased: near-unit scalar.
    assert factors.loc["summer", "scalar"] == pytest.approx(1.0, abs=0.08)
    # And the two must be well separated (must-distinguish, not both ~1).
    assert factors.loc["summer", "scalar"] - factors.loc["winter", "scalar"] > 0.10


def test_correction_beats_uncorrected_on_held_out_year(trained):
    metrics = trained["metrics"]
    unc = metrics[metrics["variant"] == "uncorrected"].iloc[0]
    cor = metrics[metrics["variant"] == "affine-wind"].iloc[0]
    assert cor["rmse"] < unc["rmse"]
    assert abs(cor["mbe"]) < abs(unc["mbe"])


def test_nh_label_application_is_worse_than_no_correction(trained, au_paths):
    """(b) Fit/apply label consistency, must-distinguish: SH-trained factors
    applied through the NH map leave JJA biased AND break unbiased DJF —
    strictly worse than not correcting at all. This is the bug mode the
    seasons seam exists to eliminate, demonstrated on the full pipeline's
    real fitted factors."""
    spec = trained["spec"]
    factors = trained["factors"]
    day_cf = au_paths["day_cf"]

    src = AEMONemSource()
    obs_cf, turb_info, reanalysis, power_curves = val_set(
        "AU-NEM",
        calc_z0=True,
        mode="all",
        year_test=TEST_YEAR,
        obs_level="turbine",
        source=src,
        era5_dir=au_paths["root"] / "era5",
        bbox=spec.bbox,
    )
    clus_info = turb_info.copy()
    clus_info["cluster"] = 0

    def monthly_rmse(sim_cf: pd.DataFrame) -> float:
        sim = sim_cf.copy()
        sim["time"] = pd.to_datetime(sim["time"])
        farm_cols = [c for c in sim.columns if c != "time"]
        sim_m = sim.set_index("time")[farm_cols].mean(axis=1).resample("ME").mean()
        truth_m = day_cf[day_cf.index.year == TEST_YEAR].resample("ME").mean()
        aligned = pd.concat([sim_m, truth_m], axis=1, keys=["sim", "truth"]).dropna()
        return float(np.sqrt(((aligned["sim"] - aligned["truth"]) ** 2).mean()))

    _, unc = wind.simulate_wind(reanalysis, clus_info, power_curves)
    _, cor_sh = wind.simulate_wind(
        reanalysis, clus_info, power_curves, factors, "season", seasons=spec.seasons
    )
    _, cor_nh = wind.simulate_wind(
        reanalysis, clus_info, power_curves, factors, "season", seasons=NH_SEASONS
    )

    rmse_unc = monthly_rmse(unc)
    rmse_sh = monthly_rmse(cor_sh)
    rmse_nh = monthly_rmse(cor_nh)

    # The three outcomes must be distinguishable and correctly ordered:
    # consistent SH apply fixes the bias; NH apply is worse than doing nothing.
    assert rmse_sh < 0.5 * rmse_unc, (rmse_sh, rmse_unc)
    assert rmse_nh > rmse_unc, (rmse_nh, rmse_unc)
    assert rmse_nh > 2.0 * rmse_sh, (rmse_nh, rmse_sh)
