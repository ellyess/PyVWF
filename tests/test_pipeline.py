"""End-to-end tests for the PyVWF orchestration on a synthetic fleet.

These drive the real, unmocked pipeline — the same code path a user gets from
``pyvwf-train``:

    ERA5-shaped winds -> hub-height extrapolation -> power curve -> per-cluster
    linear wind-speed correction (w' = scalar*w + offset) fitted against
    observed capacity factors -> corrected simulation -> error metrics

Nothing is stubbed: `PyVWF.train` and `PyVWF.simulate_cf` read from disk,
cluster the fleet, run the numerical offset fit, and write their real outputs.
The only thing synthetic is the *data* — a small fleet with a deliberately
planted bias (the reanalysis blows harder than the turbines actually generate),
so we know which way a correct correction must move.

The fixture writes the on-disk layout the loaders expect, rather than
monkeypatching the loaders, so the schema contract is exercised too: if the
expected filename or column set changes, these fail.
"""
from __future__ import annotations

from calendar import monthrange

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from vwf.config import PyVWFPaths
from vwf.data import cluster_train_set, train_set
from vwf.metrics import calculate_error
from vwf.vwf import PyVWF


# The reanalysis spans these years; observations are generated for the same
# window so the training merge (on year+month) has something to join on.
YEARS = (2015, 2016)
YEAR_TEST = 2016

# The planted bias: turbines actually generate at this capacity factor, while
# the synthetic reanalysis winds are strong enough to simulate well above it.
# A correct bias correction must therefore scale the wind speeds *down*.
TRUE_CF = 0.25


def _write_era5(era5_dir, seed=11):
    """Hourly u/v at 10 m and 100 m on a small grid inside the DK bounding box.

    A west-east wind gradient gives the two turbine clusters genuinely
    different biases, so a per-cluster correction has something to learn that a
    single global correction could not capture.
    """
    era5_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    times = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31 23:00", freq="h")
    lats = np.array([55.0, 55.5, 56.0])
    lons = np.array([8.0, 8.75, 9.5])
    shape = (len(times), len(lats), len(lons))

    # Windier in the west (low lon) than the east.
    gradient = np.linspace(1.5, -1.5, len(lons))[None, None, :]
    base = 10.0 + gradient + rng.normal(0.0, 1.5, size=shape)
    base = np.clip(base, 0.5, None)

    # Split into u/v components; 10 m winds follow a log profile with z0 ~ 0.05 m.
    u100 = base / np.sqrt(2.0)
    v100 = base / np.sqrt(2.0)
    shear = np.log(10 / 0.05) / np.log(100 / 0.05)
    u10, v10 = u100 * shear, v100 * shear

    xr.Dataset(
        {
            "u100": (("time", "lat", "lon"), u100),
            "v100": (("time", "lat", "lon"), v100),
            "u10": (("time", "lat", "lon"), u10),
            "v10": (("time", "lat", "lon"), v10),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    ).to_netcdf(era5_dir / "era5_synthetic.nc")


def _write_fleet(dk_dir):
    """Six onshore turbines in two spatial groups, in the DK loader's schema.

    Manufacturer/capacity/diameter are chosen so `add_models` resolves them to
    the bundled synthetic power curve `Synthetic.Onshore2000` (p_density ~398).
    """
    dk_dir.mkdir(parents=True, exist_ok=True)

    west = [(55.1, 8.1), (55.4, 8.2), (55.2, 8.3)]
    east = [(55.6, 9.3), (55.9, 9.4), (55.7, 9.2)]
    rows = []
    for i, (lat, lon) in enumerate(west + east):
        rows.append(
            {
                "ID": f"t{i}",
                "manufacturer": "Synthetic",
                "capacity": 2000.0,     # kW
                "diameter": 80.0,
                "height": 100.0,
                "lon": lon,
                "lat": lat,
                "location_type": "Land",
            }
        )
    fleet = pd.DataFrame(rows)
    fleet.to_csv(dk_dir / "dk_md.csv", index=False)

    # Monthly generation (kWh) consistent with TRUE_CF, which is how
    # prep_country inverts it: cf = generation / (days * 24 * capacity_kW).
    obs = []
    for turb in fleet.itertuples():
        for year in YEARS:
            for month in range(1, 13):
                hours = monthrange(year, month)[1] * 24
                obs.append(
                    {
                        "ID": turb.ID,
                        "year": year,
                        "month": month,
                        "generation_kwh": TRUE_CF * hours * turb.capacity,
                    }
                )
    pd.DataFrame(obs).to_csv(dk_dir / "dk_obs_2002_2020.csv", index=False)
    return fleet


@pytest.fixture
def synthetic_dk(tmp_path, monkeypatch):
    """Lay out a synthetic DK dataset on disk and point PyVWFPaths at it."""
    _write_era5(tmp_path / "era5")
    fleet = _write_fleet(tmp_path / "turbine_level_data" / "DK")

    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path / "turbine_level_data")
    monkeypatch.setattr(PyVWFPaths, "ERA5_DATA", tmp_path / "era5")
    return {"root": tmp_path, "fleet": fleet}


# ---------------------------------------------------------------------------
# train_set / cluster_train_set — the data-preparation layer
# ---------------------------------------------------------------------------

def test_train_set_pairs_observations_with_simulations(synthetic_dk):
    gen_cf, turb_info, reanalysis, power_curves = train_set(
        "DK", calc_z0=True, mode="onshore"
    )

    assert {"ID", "year", "month", "obs", "sim"} <= set(gen_cf.columns)
    assert len(turb_info) == len(synthetic_dk["fleet"])
    assert "model" in turb_info.columns  # add_models resolved a power curve
    assert turb_info["model"].isin(power_curves.columns).all()

    # Capacity factors must be physical, and the observations must come back as
    # the CF we planted (generation was written as TRUE_CF * hours * capacity).
    assert gen_cf["sim"].between(0.0, 1.0).all()
    assert gen_cf["obs"].to_numpy() == pytest.approx(TRUE_CF, abs=1e-6)

    # The planted bias: the reanalysis is windier than reality, so the
    # uncorrected simulation must over-predict.
    assert gen_cf["sim"].mean() > TRUE_CF

    # prep_era5 resamples to daily
    assert reanalysis.sizes["time"] == 731  # 2015 + 2016 (leap)


def test_cluster_train_set_fits_one_scalar_per_cluster(synthetic_dk):
    gen_cf, turb_info, _, _ = train_set("DK", calc_z0=True, mode="onshore")
    bias_df, clus_info = cluster_train_set(gen_cf, "fixed", 2, turb_info)

    assert set(bias_df["cluster"].unique()) == {0, 1}
    assert "scalar" in bias_df.columns
    assert set(clus_info["cluster"].unique()) == {0, 1}

    # Simulation over-predicts, so the fitted scalar must pull it down.
    assert (bias_df["scalar"] < 1.0).all()


def test_cluster_train_set_respects_temporal_resolution(synthetic_dk):
    """`month` fits a correction per calendar month, `fixed` a single one.

    The bias table keeps the year dimension (one row per year x cluster x
    slice); the per-year rows are collapsed into a single factor per cluster
    when `train` writes the correction-factor file.
    """
    gen_cf, turb_info, _, _ = train_set("DK", calc_z0=True, mode="onshore")
    n_years, n_clusters = len(YEARS), 2

    fixed, _ = cluster_train_set(gen_cf, "fixed", n_clusters, turb_info)
    monthly, _ = cluster_train_set(gen_cf, "month", n_clusters, turb_info)
    seasonal, _ = cluster_train_set(gen_cf, "season", n_clusters, turb_info)

    assert len(fixed) == n_clusters * n_years
    assert len(monthly) == n_clusters * n_years * 12
    assert len(seasonal) == n_clusters * n_years * 4

    assert set(fixed["time_slice"].unique()) == {"1/1"}
    assert set(int(m) for m in monthly["time_slice"].unique()) == set(range(1, 13))
    assert set(seasonal["time_slice"].unique()) == {
        "winter", "spring", "summer", "autumn"
    }


# ---------------------------------------------------------------------------
# PyVWF.train + simulate_cf — the full orchestration
# ---------------------------------------------------------------------------

@pytest.fixture
def trained_model(synthetic_dk):
    """Run the real training pipeline once; reused by the assertions below.

    dask_n_workers=0 selects the sequential offset fit, avoiding a
    LocalCluster inside the test process.
    """
    model = PyVWF(
        str(synthetic_dk["root"] / "out"),
        "DK",
        True,             # correct
        True,             # calc_z0
        "onshore",
        [2],              # cluster_list
        ["fixed"],        # time_res_list
    )
    model.train(dask_n_workers=0)
    return model, synthetic_dk


def test_train_writes_correction_factors(trained_model):
    model, _ = trained_model
    factors_path = (
        f"{model.directory_path}/training/correction-factors/DK_factors_fixed_2.csv"
    )
    factors = pd.read_csv(factors_path)

    assert {"cluster", "scalar", "offset"} <= set(factors.columns)
    assert len(factors) == 2  # one row per cluster
    assert factors["scalar"].notna().all()
    assert factors["offset"].notna().all()

    # The reanalysis over-predicts, so the learned correction must reduce wind
    # speed: scalar < 1 (and/or a negative offset), never inflate it.
    assert (factors["scalar"] < 1.0).all()


def test_train_writes_the_training_fleet(trained_model):
    model, ctx = trained_model
    turb_info = pd.read_csv(
        f"{model.directory_path}/training/simulated-turbines/DK_train_turb_info.csv"
    )
    assert len(turb_info) == len(ctx["fleet"])
    assert {"ID", "lat", "lon", "capacity", "model"} <= set(turb_info.columns)


def test_simulate_cf_correction_reduces_error(trained_model):
    """The headline claim: the trained correction brings the simulated capacity
    factors closer to the observations than the uncorrected simulation."""
    model, _ = trained_model
    model.simulate_cf(YEAR_TEST)

    cf_dir = f"{model.directory_path}/results/capacity-factor"
    obs = pd.read_csv(f"{cf_dir}/DK_{YEAR_TEST}_obs_cf.csv")
    unc = pd.read_csv(f"{cf_dir}/DK_{YEAR_TEST}_unc_cf.csv")
    cor = pd.read_csv(f"{cf_dir}/DK_{YEAR_TEST}_fixed_2_cor_cf.csv")
    turb_info = pd.read_csv(
        f"{model.directory_path}/training/simulated-turbines/DK_{YEAR_TEST}_turb_info.csv"
    )

    unc_rmse, unc_mae, unc_mbe = calculate_error("total", unc, obs, turb_info)
    cor_rmse, cor_mae, cor_mbe = calculate_error("total", cor, obs, turb_info)

    # Uncorrected over-predicts (positive bias); the correction must shrink it.
    assert unc_mbe > 0
    assert abs(cor_mbe) < abs(unc_mbe)
    assert cor_rmse < unc_rmse
    assert cor_mae < unc_mae


def test_simulate_cf_outputs_are_physical(trained_model):
    model, ctx = trained_model
    model.simulate_cf(YEAR_TEST)

    cf_dir = f"{model.directory_path}/results/capacity-factor"
    cor = pd.read_csv(f"{cf_dir}/DK_{YEAR_TEST}_fixed_2_cor_cf.csv")

    turbine_cols = [c for c in cor.columns if c != "time"]
    assert len(turbine_cols) == len(ctx["fleet"])

    values = cor[turbine_cols].to_numpy()
    assert np.isfinite(values).all(), "corrected CF contains NaN/inf"
    assert ((values >= 0.0) & (values <= 1.0)).all(), "corrected CF outside [0, 1]"


def test_legacy_runs_write_a_provenance_manifest(trained_model):
    """PyVWF.train and simulate_cf self-describe their curve library
    (design §6): the manifest records synthetic-vs-external and the run
    parameters, so legacy outputs are attributable too."""
    import json

    model, _ = trained_model
    manifest_path = f"{model.directory_path}/run_manifest.json"
    manifest = json.loads(open(manifest_path).read())
    assert manifest["run_mode"] == "legacy-train"
    assert manifest["country"] == "DK"
    assert manifest["curve_library"]["library"] == "synthetic-bundled"

    model.simulate_cf(YEAR_TEST)
    manifest = json.loads(open(manifest_path).read())
    assert manifest["run_mode"] == "legacy-simulate"
    assert manifest["year_test"] == YEAR_TEST


def test_manifest_failure_never_aborts_a_run(trained_model, monkeypatch, capsys):
    """The never-abort condition (design §6): a manifest-write failure logs a
    warning and the run completes."""
    import vwf.harness.provenance as provenance

    def boom(*args, **kwargs):
        raise RuntimeError("disk full (simulated)")

    monkeypatch.setattr(provenance, "write_manifest_safe", boom)
    model, _ = trained_model
    model.simulate_cf(YEAR_TEST)  # must not raise
    assert "could not write run manifest" in capsys.readouterr().out


def test_simulate_cf_is_idempotent(trained_model):
    """Re-running must reuse the existing outputs rather than recompute or
    corrupt them — the guard that lets a long sweep be resumed."""
    model, _ = trained_model
    model.simulate_cf(YEAR_TEST)

    cf_path = (
        f"{model.directory_path}/results/capacity-factor/"
        f"DK_{YEAR_TEST}_fixed_2_cor_cf.csv"
    )
    first = pd.read_csv(cf_path)

    model.simulate_cf(YEAR_TEST)
    second = pd.read_csv(cf_path)

    pd.testing.assert_frame_equal(first, second)
