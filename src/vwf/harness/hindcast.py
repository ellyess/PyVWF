"""Apply a trained correction over a long ERA5 window: wind resource in context.

Answers "how does this period's wind resource compare with the record?": take a
correction already trained and validated for a region, apply it to every ERA5
year available, collapse the fleet to one capacity-weighted national monthly
capacity factor, and rank a chosen period against the rest of the record. Useful
for placing a low- or high-wind year in multi-year context, and for separating a
genuine resource anomaly from a modelling artefact.

It reuses the validated harness apply path unchanged (``model.apply`` for the
corrected series, ``wind.simulate_wind`` for the uncorrected), so the hindcast
inherits the same correction the scorecard validated.

DATA REQUIREMENT: the length of the context is the length of the ERA5 archive on
disk for the region. A genuine multi-decade "is the wind getting weaker" claim
needs ERA5 back to 1979 (a large CDS fetch, see scripts/fetch/era5.py); with only
a few years present this still produces the series and the ranking, but the
percentile context is only as deep as the years available. The function reports
how many years it used so the note can be honest about it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import vwf.wind as wind
from vwf.clustering import cluster_turbines
from vwf.config import PyVWFPaths
from vwf.data import load_power_curves, val_set
from vwf.harness.corrections import get_correction
from vwf.harness.regions import RegionSpec
from vwf.datasets.era5 import prep_era5


def _national_monthly(cf_wide: pd.DataFrame, turb_info: pd.DataFrame) -> pd.Series:
    """Capacity-weighted national monthly mean CF from a wide (time x ID) frame.

    NaN-skipping, reweighting on the grid points present each timestep, matching
    the country aggregation the evaluate path uses.
    """
    cap = turb_info.assign(ID=turb_info["ID"].astype(str)).set_index("ID")["capacity"]
    cols = [c for c in cf_wide.columns if c != "time" and str(c) in cap.index]
    caps = cap[[str(c) for c in cols]].to_numpy(float)
    vals = cf_wide[cols].to_numpy(float)
    present = ~np.isnan(vals)
    wsum = np.where(present, caps, 0.0).sum(axis=1)
    nat = np.where(wsum > 0, np.where(present, vals * caps, 0.0).sum(axis=1) / wsum, np.nan)
    frame = pd.DataFrame({"time": pd.to_datetime(cf_wide["time"]), "cf": nat})
    return frame.groupby(frame["time"].dt.to_period("M"))["cf"].mean()


def run_hindcast(
    spec: RegionSpec,
    train_run_dir: str | Path,
    *,
    num_clu: int,
    time_res: str,
    fleet_year: int | None = None,
    mode: str = "all",
    calc_z0: bool = True,
    era5_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Corrected and uncorrected national monthly CF over the full ERA5 window.

    Args:
        spec: Region config.
        train_run_dir: Train run holding ``factors_<slice>_<k>.csv`` and
            ``train_turb_info_<k>.csv``.
        num_clu: Cluster count (matches a trained factors file).
        time_res: Time slice of the factors to apply.
        fleet_year: Year whose installed fleet defines the sites (defaults to the
            region's first test year). The resource is evaluated at that fleet.
        mode: ``"all"``, ``"onshore"`` or ``"offshore"``.
        era5_dir: ERA5 directory; defaults to ``<INPUT>/<spec.era5_path>``.

    Returns:
        Tidy DataFrame with ``year``, ``month``, ``cf_uncorrected``,
        ``cf_corrected``, one row per calendar month in the archive.
    """
    train_run_dir = Path(train_run_dir)
    fleet_year = int(fleet_year if fleet_year is not None else spec.test_years[0])
    era5_dir = Path(era5_dir) if era5_dir else PyVWFPaths.INPUT_ROOT / spec.era5_path

    # Prepared fleet (curves, hub heights) for the chosen year, via the harness.
    _, turb_info, _, _ = val_set(
        spec.code, calc_z0, mode, year_test=fleet_year, obs_level="turbine",
        era5_dir=era5_dir, bbox=spec.bbox,
    )
    power_curves = load_power_curves()

    # Full-window reanalysis (no year filter): the length of the context.
    reanalysis = prep_era5(spec.code, False, calc_z0, bbox=spec.bbox, era5_dir=era5_dir)

    # Cluster the fleet against the training fleet so the factors' cluster ids align.
    train_fleet = pd.read_csv(train_run_dir / f"train_turb_info_{num_clu}.csv")
    clus_info = cluster_turbines(
        num_clu, train_fleet, False, turb_info,
        min_cluster_size=spec.min_cluster_size,
    )
    factors = pd.read_csv(train_run_dir / f"factors_{time_res}_{num_clu}.csv")

    model = get_correction(spec.correction_model)
    _, cor_cf = model.apply(
        reanalysis, clus_info, power_curves, factors, time_res, seasons=spec.seasons
    )
    _, unc_cf = wind.simulate_wind(reanalysis, clus_info, power_curves)

    cor_m = _national_monthly(cor_cf, clus_info).rename("cf_corrected")
    unc_m = _national_monthly(unc_cf, clus_info).rename("cf_uncorrected")
    out = pd.concat([unc_m, cor_m], axis=1).reset_index()
    out["year"] = out["time"].dt.year
    out["month"] = out["time"].dt.month
    return out[["year", "month", "cf_uncorrected", "cf_corrected"]]


def rank_in_context(monthly: pd.DataFrame, column: str = "cf_corrected") -> pd.DataFrame:
    """Add each month's percentile rank within its own calendar month across years.

    Removes the seasonal cycle by ranking (say) every January against all other
    Januaries in the record, so "this was a low-wind month" is a like-for-like
    statement. ``pct`` is 0-1 (0 = lowest in the record for that calendar month).
    Also adds the calendar-month mean and the anomaly against it.
    """
    df = monthly.copy()
    df["month_mean"] = df.groupby("month")[column].transform("mean")
    df["anomaly"] = df[column] - df["month_mean"]
    df["pct"] = df.groupby("month")[column].rank(pct=True)
    df["n_years"] = df.groupby("month")[column].transform("count")
    return df
