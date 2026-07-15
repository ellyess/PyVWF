"""Harness driver: train, evaluate, and transfer runs for one region config.

The thin CLI in ``scripts/validate_region.py`` wraps these functions; the
logic lives here so it is importable and tested.

Transfer semantics are normative (design §7): collapse the source region's
factors to ONE capacity-weighted (scalar, offset) per time-slice, apply
uniformly to the target, and match seasonal slices by season NAME under the
TARGET's season definitions. Nothing else — no spatial matching schemes on
this branch. The driver also enforces the approved pair set: transfer runs
must have AU-NEM on exactly one side.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import vwf.wind as wind
from vwf.clustering import cluster_turbines
from vwf.config import PyVWFPaths
from vwf.data import train_set, val_set
from vwf.harness.corrections import get_correction
from vwf.harness.provenance import write_manifest_safe
from vwf.harness.regions import RegionSpec
from vwf.harness.skill import collapse_pseudo_replicates, skill_metrics
from vwf.sources import ObservationSource, get_source

#: The approved transfer pair set (Phase 0 sign-off): AU-NEM against Europe,
#: in either direction. Everything else is out of scope on this branch.
TRANSFER_HUB = "AU-NEM"


def check_transfer_pair(source_code: str, target_code: str) -> None:
    """Reject transfer pairs outside the approved AU↔Europe set."""
    codes = {source_code.upper(), target_code.upper()}
    if source_code.upper() == target_code.upper():
        raise ValueError(f"transfer requires two different regions, got {codes}")
    if TRANSFER_HUB not in codes:
        raise ValueError(
            f"transfer pair ({source_code}, {target_code}) is outside the approved "
            f"set: {TRANSFER_HUB} must be on exactly one side (design §7). "
            "Other pairs are out of scope on this branch."
        )


def resolve_source(spec: RegionSpec) -> ObservationSource:
    """Resolve the region's observation source from the registry by name."""
    if spec.source == "entsoe-country":
        raise NotImplementedError(
            "Country-level file-backed loading (entsoe-country) is wired up "
            "with the Europe re-runs in Phase 2. Pass an explicit source= "
            "(e.g. InMemoryCountrySource) until then."
        )
    return get_source(spec.source, spec.code)


def _era5_dir(spec: RegionSpec) -> Path:
    return PyVWFPaths.INPUT_ROOT / spec.era5_path


def _run_dir(out_root: str | Path, spec: RegionSpec, mode: str, run_name: str | None) -> Path:
    stamp = run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(out_root) / spec.code / f"{mode}-{stamp}"


def _monthly_long(cf_wide: pd.DataFrame) -> pd.DataFrame:
    """Melt a (time x ID) simulation frame to monthly-mean long format."""
    cf = cf_wide.copy()
    cf["time"] = pd.to_datetime(cf["time"])
    monthly = cf.groupby(pd.Grouper(key="time", freq="ME")).mean().reset_index()
    long = monthly.melt(id_vars=["time"], var_name="ID", value_name="cf_sim")
    long["year"] = long["time"].dt.year
    long["month"] = long["time"].dt.month
    long["ID"] = long["ID"].astype(str)
    return long[["ID", "year", "month", "cf_sim"]]


def _tidy_eval_frame(
    sim_cf: pd.DataFrame, obs_cf: pd.DataFrame, turb_info: pd.DataFrame
) -> pd.DataFrame:
    """Merge simulated and observed monthly CFs into the skill-frame shape."""
    sim_long = _monthly_long(sim_cf)

    obs = obs_cf.copy()
    obs["time"] = pd.to_datetime(obs["time"])
    obs_long = obs.melt(id_vars=["time"], var_name="ID", value_name="cf_obs")
    obs_long["year"] = obs_long["time"].dt.year
    obs_long["month"] = obs_long["time"].dt.month
    obs_long["ID"] = obs_long["ID"].astype(str)
    obs_long = obs_long[["ID", "year", "month", "cf_obs"]]

    merged = sim_long.merge(obs_long, on=["ID", "year", "month"], how="inner")
    capacity = turb_info[["ID", "capacity"]].copy()
    capacity["ID"] = capacity["ID"].astype(str)
    return merged.merge(capacity, on="ID", how="left")


def run_train(
    spec: RegionSpec,
    out_root: str | Path,
    *,
    source: ObservationSource | None = None,
    calc_z0: bool = True,
    mode: str = "all",
    run_name: str | None = None,
) -> Path:
    """Train correction factors for every (cluster, slice) combo in the config.

    Returns the run directory, containing ``factors_<slice>_<n>.csv``, the
    training fleet, and ``run_manifest.json``.
    """
    source = source if source is not None else resolve_source(spec)
    gen_cf, turb_info, reanalysis, power_curves = train_set(
        spec.code,
        calc_z0,
        mode,
        obs_level=spec.obs_level,
        source=source,
        era5_dir=_era5_dir(spec),
        bbox=spec.bbox,
    )

    model = get_correction(spec.correction_model)
    run_dir = _run_dir(out_root, spec, "train", run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    for num_clu in spec.cluster_list:
        for time_res in spec.time_slices:
            factors, clus_info = model.fit(
                gen_cf,
                turb_info,
                reanalysis,
                power_curves,
                num_clusters=num_clu,
                time_res=time_res,
                seasons=spec.seasons,
                obs_level=spec.obs_level,
            )
            factors.to_csv(run_dir / f"factors_{time_res}_{num_clu}.csv", index=False)
            clus_info.to_csv(
                run_dir / f"train_turb_info_{num_clu}.csv", index=False
            )

    write_manifest_safe(run_dir, spec, extra={"run_mode": "train", "fleet_mode": mode})
    return run_dir


def run_evaluate(
    spec: RegionSpec,
    train_run_dir: str | Path,
    out_root: str | Path,
    *,
    year: int | None = None,
    source: ObservationSource | None = None,
    calc_z0: bool = True,
    mode: str = "all",
    run_name: str | None = None,
) -> Path:
    """Evaluate a trained run against a held-out year.

    Writes ``metrics.csv`` with one row per variant (uncorrected baseline
    plus each factors file) and returns the evaluation run directory.
    """
    if spec.obs_level != "turbine":
        raise NotImplementedError(
            "Country-level evaluation lands with the Europe re-runs (Phase 2)."
        )
    year = int(year if year is not None else spec.test_years[0])
    train_run_dir = Path(train_run_dir)

    source = source if source is not None else resolve_source(spec)
    obs_cf, turb_info, reanalysis, power_curves = val_set(
        spec.code,
        calc_z0,
        mode,
        year_test=year,
        obs_level=spec.obs_level,
        source=source,
        era5_dir=_era5_dir(spec),
        bbox=spec.bbox,
    )

    model = get_correction(spec.correction_model)
    run_dir = _run_dir(out_root, spec, f"evaluate-{year}", run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    def _skill_row(sim_cf: pd.DataFrame, variant: str, num_clu, time_res) -> dict:
        tidy = _tidy_eval_frame(sim_cf, obs_cf, turb_info)
        tidy = collapse_pseudo_replicates(tidy, spec)
        metrics = skill_metrics(tidy)
        return {"variant": variant, "num_clu": num_clu, "time_res": time_res, **metrics}

    _, unc_cf = wind.simulate_wind(reanalysis, turb_info, power_curves)
    rows.append(_skill_row(unc_cf, "uncorrected", 1, "none"))

    for factors_path in sorted(train_run_dir.glob("factors_*.csv")):
        time_res, num_clu_str = factors_path.stem.split("_")[1:3]
        num_clu = int(num_clu_str)
        factors = pd.read_csv(factors_path)
        train_fleet = pd.read_csv(train_run_dir / f"train_turb_info_{num_clu}.csv")
        clus_info = cluster_turbines(num_clu, train_fleet, False, turb_info)
        _, cor_cf = model.apply(
            reanalysis, clus_info, power_curves, factors, time_res, seasons=spec.seasons
        )
        rows.append(_skill_row(cor_cf, spec.correction_model, num_clu, time_res))

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)
    write_manifest_safe(
        run_dir,
        spec,
        extra={
            "run_mode": "evaluate",
            "evaluation_year": year,
            "trained_from": str(train_run_dir),
        },
    )
    return run_dir


def collapse_factors(
    factors: pd.DataFrame, cluster_capacity: pd.Series, time_res: str
) -> pd.DataFrame:
    """Collapse per-cluster factors to one (scalar, offset) per time-slice.

    The collapse is CAPACITY-weighted over source clusters (design §7.1):
    weights are the installed capacity behind each cluster in the SOURCE
    region's training fleet, never an unweighted mean.
    """
    merged = factors.copy()
    merged["_w"] = merged["cluster"].map(cluster_capacity)
    if merged["_w"].isna().any():
        missing = sorted(int(c) for c in merged.loc[merged["_w"].isna(), "cluster"].unique())
        raise ValueError(
            f"no capacity weight for cluster(s) {missing}: the factors table and "
            "the source training fleet disagree"
        )
    merged["_ws"] = merged["scalar"] * merged["_w"]
    merged["_wo"] = merged["offset"] * merged["_w"]
    grouped = merged.groupby(time_res, as_index=False)[["_ws", "_wo", "_w"]].sum()
    collapsed = pd.DataFrame(
        {
            "cluster": 0,
            time_res: grouped[time_res],
            "scalar": grouped["_ws"] / grouped["_w"],
            "offset": grouped["_wo"] / grouped["_w"],
        }
    )
    return collapsed


def run_transfer(
    source_spec: RegionSpec,
    source_run_dir: str | Path,
    target_spec: RegionSpec,
    out_root: str | Path,
    *,
    year: int | None = None,
    target_source: ObservationSource | None = None,
    calc_z0: bool = True,
    mode: str = "all",
    run_name: str | None = None,
) -> Path:
    """Apply a source region's collapsed correction to a target region.

    Design §7, exactly: capacity-weighted collapse per slice, uniform
    application, season-name matching under the TARGET's definitions (the
    collapsed factors carry season NAMES; applying them with
    ``seasons=target_spec.seasons`` is what maps AU winter-trained factors
    onto the target's winter months).
    """
    check_transfer_pair(source_spec.code, target_spec.code)
    if target_spec.obs_level != "turbine":
        raise NotImplementedError(
            "Country-level transfer targets land with the Europe re-runs (Phase 2)."
        )
    source_run_dir = Path(source_run_dir)
    year = int(year if year is not None else target_spec.test_years[0])

    target_source = (
        target_source if target_source is not None else resolve_source(target_spec)
    )
    obs_cf, turb_info, reanalysis, power_curves = val_set(
        target_spec.code,
        calc_z0,
        mode,
        year_test=year,
        obs_level=target_spec.obs_level,
        source=target_source,
        era5_dir=_era5_dir(target_spec),
        bbox=target_spec.bbox,
    )

    model = get_correction(target_spec.correction_model)
    run_dir = _run_dir(out_root, target_spec, f"transfer-from-{source_spec.code}", run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    def _skill_row(sim_cf: pd.DataFrame, variant: str, time_res) -> dict:
        tidy = _tidy_eval_frame(sim_cf, obs_cf, turb_info)
        tidy = collapse_pseudo_replicates(tidy, target_spec)
        metrics = skill_metrics(tidy)
        return {"variant": variant, "time_res": time_res, **metrics}

    _, unc_cf = wind.simulate_wind(reanalysis, turb_info, power_curves)
    rows.append(_skill_row(unc_cf, "uncorrected", "none"))

    # Uniform application: every target site is cluster 0.
    target_info = turb_info.copy()
    target_info["cluster"] = 0

    for factors_path in sorted(source_run_dir.glob("factors_*.csv")):
        time_res, num_clu_str = factors_path.stem.split("_")[1:3]
        factors = pd.read_csv(factors_path)
        source_fleet = pd.read_csv(
            source_run_dir / f"train_turb_info_{num_clu_str}.csv"
        )
        cluster_capacity = source_fleet.groupby("cluster")["capacity"].sum()
        collapsed = collapse_factors(factors, cluster_capacity, time_res)
        collapsed.to_csv(
            run_dir / f"collapsed_factors_{time_res}_{num_clu_str}.csv", index=False
        )
        _, cor_cf = model.apply(
            reanalysis,
            target_info,
            power_curves,
            collapsed,
            time_res,
            seasons=target_spec.seasons,  # season-NAME matching, design §7.3
        )
        rows.append(
            _skill_row(cor_cf, f"transfer-from-{source_spec.code}-{num_clu_str}", time_res)
        )

    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
    write_manifest_safe(
        run_dir,
        target_spec,
        extra={
            "run_mode": "transfer",
            "transfer_source_region": source_spec.code,
            "transfer_source_run": str(source_run_dir),
            "transfer_semantics": "capacity-weighted-collapse, uniform, season-name-matched",
            "evaluation_year": year,
        },
    )
    return run_dir
