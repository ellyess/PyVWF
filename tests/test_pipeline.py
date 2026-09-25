"""The data pipeline on a synthetic fleet, and the fixture the harness tests share.

These drive the real, unmocked data functions the harness delegates to:

    ERA5-shaped winds -> hub-height extrapolation -> power curve -> paired
    simulated and observed capacity factors -> per-cluster scalars

The only thing synthetic is the *data*: a small fleet with a deliberately
planted bias (the reanalysis blows harder than the turbines actually generate),
so we know which way a correct correction must move. The full train and
evaluate loop runs through the harness in ``test_harness_driver.py``.

The fixture writes the on-disk layout the loaders expect, rather than
monkeypatching the loaders, so the schema contract is exercised too: if the
expected filename or column set changes, these fail. Until 2026-09-24 this file
also drove the legacy ``PyVWF`` class end to end; that class was removed.
"""

from __future__ import annotations

from calendar import monthrange

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyvwf.data import cluster_train_set, train_set


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

    Manufacturer/capacity/diameter give a p_density of ~398, so `add_models`
    resolves them to the nearest bundled open-library curve
    (`NREL_Reference_5MW_126`, p_density ~401).
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
                "capacity": 2000.0,  # kW
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


# ---------------------------------------------------------------------------
# train_set / cluster_train_set: the data-preparation layer
# ---------------------------------------------------------------------------


def test_train_set_pairs_observations_with_simulations(synthetic_dk):
    gen_cf, turb_info, reanalysis, power_curves = train_set("DK", calc_z0=True, mode="onshore")

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
    assert set(seasonal["time_slice"].unique()) == {"winter", "spring", "summer", "autumn"}
