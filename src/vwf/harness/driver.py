"""Harness driver: train, evaluate, and transfer runs for one region config.

The thin CLI in ``scripts/analysis/validate_region.py`` wraps these functions; the
logic lives here so it is importable and tested.

Transfer semantics are normative (design §7): collapse the source region's
factors to ONE capacity-weighted (scalar, offset) per time-slice, apply
uniformly to the target, and match seasonal slices by season NAME under the
TARGET's season definitions. Nothing else: no spatial matching schemes on
this branch. The driver also enforces the approved pair set: transfer runs
must have AU-NEM on exactly one side.
"""
from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

import vwf.wind as wind
from vwf.clustering import cluster_turbines
from vwf.config import PyVWFPaths
from vwf.data import assign_country_clusters, train_set, val_set
from vwf.harness.corrections import fit_quality, get_correction
from vwf.harness.provenance import write_manifest_safe
from vwf.harness.regions import RegionSpec
from vwf.harness.skill import collapse_pseudo_replicates, skill_metrics
from vwf.sources import (
    EntsoeFileSource,
    EntsoeZonalFileSource,
    ObservationSource,
    get_source,
)

#: The transfer pair set that has been validated: AU-NEM against Europe, in
#: either direction. Other pairings are untested rather than unsupported.
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


def resolve_source(
    spec: RegionSpec, split: Literal["train", "test"] = "train"
) -> ObservationSource:
    """Resolve the region's observation source from the registry by name.

    Country-level file-backed regions ("entsoe-country", "entsoe-zonal") read a
    different file per split, so the source is built per split; turbine-level
    sources ignore the split and are resolved by country from the registry.
    """
    if spec.source in ("entsoe-country", "entsoe-zonal"):
        cls = EntsoeFileSource if spec.source == "entsoe-country" else EntsoeZonalFileSource
        return cls(spec.code, split, spec.train_years, spec.test_years[0])
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
                min_cluster_size=spec.min_cluster_size,
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
    score_zones: ObservationSource | None = None,
) -> Path:
    """Evaluate a trained run against a held-out year.

    Writes ``metrics.csv`` with one row per variant (uncorrected baseline plus
    each factors file), saves every corrected-CF frame (``cor_cf_*.csv``, plus
    ``unc_cf.csv``) so runs can be diffed at frame level, and returns the
    evaluation run directory. Handles both obs levels: turbine-level clusters
    the test fleet against the training fleet; country-level reuses the grid
    points' own cluster assignments and scores the capacity-weighted country
    aggregate.

    Args:
        score_zones: Optional zonal observation source used for an extra
            per-zone metric. A zonal run supplies its own; pass one explicitly
            to score a NATIONALLY trained run zone by zone, which is what makes
            the two comparable. Without it the national fit is only ever judged
            on the national aggregate, which is its own training objective.
    """
    is_country = spec.obs_level == "country"
    year = int(year if year is not None else spec.test_years[0])
    train_run_dir = Path(train_run_dir)

    source = source if source is not None else resolve_source(spec, "test")
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

    # A zonal source can also be scored zone by zone. The national metric is the
    # joint optimiser's own objective, so it favours a national fit by
    # construction; the per-zone metric scores what a zonal fit actually
    # targets. Both are reported, distinguished by the "scope" column.
    zone_source = score_zones if score_zones is not None else (
        source if spec.source == "entsoe-zonal" else None
    )
    obs_zonal = zone_source.load_observations() if zone_source is not None else None

    def _skill_rows(sim_cf: pd.DataFrame, variant: str, num_clu, time_res) -> list[dict]:
        head = {"variant": variant, "num_clu": num_clu, "time_res": time_res}
        if not is_country:
            tidy = collapse_pseudo_replicates(
                _tidy_eval_frame(sim_cf, obs_cf, turb_info), spec
            )
            return [{**head, "scope": "fleet", **skill_metrics(tidy)}]

        out = [{**head, "scope": "national", **_country_skill(sim_cf, obs_cf, turb_info)}]
        if obs_zonal is not None:
            # Scored on the grid's own zone assignments, not the run's cluster
            # count, so a 1-cluster run is still judged zone by zone.
            out.append(
                {**head, "scope": "per-zone", **_zonal_skill(sim_cf, obs_zonal, turb_info)}
            )
        return out

    _, unc_cf = wind.simulate_wind(reanalysis, turb_info, power_curves)
    unc_cf.to_csv(run_dir / "unc_cf.csv", index=False)
    rows.extend(_skill_rows(unc_cf, "uncorrected", 1, "none"))

    for factors_path in sorted(train_run_dir.glob("factors_*.csv")):
        time_res, num_clu_str = factors_path.stem.split("_")[1:3]
        num_clu = int(num_clu_str)
        factors = pd.read_csv(factors_path)
        if is_country:
            # Grid points carry their own cluster assignments; no re-clustering
            # runs on the country-level path (mirrors PyVWF.simulate_cf). The
            # same resolution training used has to be reapplied here, or a
            # single-cluster national fit would be merged against a grid still
            # carrying its per-zone clusters and every factor would come out
            # NaN.
            clus_info = assign_country_clusters(turb_info, num_clu)
        else:
            train_fleet = pd.read_csv(train_run_dir / f"train_turb_info_{num_clu}.csv")
            clus_info = cluster_turbines(
                num_clu, train_fleet, False, turb_info,
                min_cluster_size=spec.min_cluster_size,
            )
        _, cor_cf = model.apply(
            reanalysis, clus_info, power_curves, factors, time_res, seasons=spec.seasons
        )
        cor_cf.to_csv(run_dir / f"cor_cf_{time_res}_{num_clu}.csv", index=False)
        # Skill alone hides a bad fit: a region can score as a corrected win
        # while carrying an implausible scalar or an offset that never
        # converged, so the fit diagnostics travel with every corrected row.
        quality = fit_quality(factors)
        if quality["n_implausible_scalar"] or quality["n_failed_offset"]:
            warnings.warn(
                f"{spec.code} {time_res} k={num_clu}: "
                f"{quality['n_implausible_scalar']} implausible scalar(s) "
                f"(max {quality['max_scalar']:.3g}), "
                f"{quality['n_failed_offset']} failed offset fit(s) in "
                f"cluster(s) {quality['degenerate_clusters']}. "
                "The skill metric can still look good; see "
                "docs/findings/method-hourly-resolution.md."
            )
        rows.extend([
            {**row, **quality}
            for row in _skill_rows(cor_cf, spec.correction_model, num_clu, time_res)
        ])

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


def _zone_aggregate(sim_cf: pd.DataFrame, members: pd.DataFrame) -> pd.Series:
    """Capacity-weighted monthly mean CF over one set of grid points."""
    cap = members.assign(ID=members["ID"].astype(str)).set_index("ID")["capacity"]
    valid = [c for c in sim_cf.columns if c != "time" and str(c) in cap.index]
    if not valid:
        return pd.Series(dtype=float)

    caps = cap[[str(c) for c in valid]].to_numpy(float)
    vals = sim_cf[valid].to_numpy(float)
    present = ~np.isnan(vals)
    wsum = np.where(present, caps, 0.0).sum(axis=1)
    agg = np.where(wsum > 0, np.where(present, vals * caps, 0.0).sum(axis=1) / wsum, np.nan)

    frame = pd.DataFrame({"time": pd.to_datetime(sim_cf["time"]), "cf_sim": agg})
    return frame.groupby(frame["time"].dt.to_period("M"))["cf_sim"].mean()


def _error_metrics(merged: pd.DataFrame) -> dict:
    """MBE, MAE, RMSE and correlation over paired sim/obs columns."""
    if merged.empty:
        return {"mbe": float("nan"), "mae": float("nan"), "rmse": float("nan"),
                "pearson_r": float("nan"), "n_months": 0}
    diff = merged["cf_sim"] - merged["cf_obs"]
    r = (
        float(np.corrcoef(merged["cf_sim"], merged["cf_obs"])[0, 1])
        if len(merged) > 1 else float("nan")
    )
    return {
        "mbe": float(diff.mean()),
        "mae": float(diff.abs().mean()),
        "rmse": float(np.sqrt((diff**2).mean())),
        "pearson_r": r,
        "n_months": int(len(merged)),
    }


def _zonal_skill(
    sim_cf: pd.DataFrame, obs_zonal: pd.DataFrame, turb_info: pd.DataFrame
) -> dict:
    """Each zone's simulated aggregate against that zone's own observation.

    The national metric scores the capacity-weighted country aggregate, which is
    exactly what the joint country optimiser targets, so it favours the national
    fit by construction: an estimator judged on its own objective tends to win.
    This scores the quantity a zonal fit actually targets. Errors from every
    zone are pooled into one set of statistics, so a country's zonal score is
    comparable across cluster counts.

    Args:
        sim_cf: Wide (time x grid ID) simulated capacity factors.
        obs_zonal: DatetimeIndexed observations with ``capacity_factor`` and
            ``cluster``.
        turb_info: Grid points with ``ID``, ``capacity`` and ``cluster``.

    Returns:
        Pooled metrics plus ``n_zones``.
    """
    obs = obs_zonal.copy()
    if not isinstance(obs.index, pd.DatetimeIndex):
        obs.index = pd.to_datetime(obs.index, utc=True, format="mixed")
    if obs.index.tz is not None:
        obs.index = obs.index.tz_convert("UTC").tz_localize(None)

    pairs = []
    for cluster, members in turb_info.groupby("cluster"):
        zone_obs = obs[obs["cluster"] == cluster]
        if zone_obs.empty:
            continue
        sim_m = _zone_aggregate(sim_cf, members)
        if sim_m.empty:
            continue
        obs_m = (
            zone_obs.groupby(pd.DatetimeIndex(zone_obs.index).to_period("M"))["capacity_factor"]
            .mean()
            .rename("cf_obs")
        )
        merged = pd.concat([sim_m, obs_m], axis=1).dropna()
        merged["cluster"] = cluster
        pairs.append(merged)

    if not pairs:
        return {**_error_metrics(pd.DataFrame()), "n_zones": 0}

    pooled = pd.concat(pairs)
    return {**_error_metrics(pooled), "n_zones": int(pooled["cluster"].nunique())}


def _country_skill(
    sim_cf: pd.DataFrame, obs_country: pd.DataFrame, turb_info: pd.DataFrame
) -> dict:
    """Capacity-weighted country aggregate vs the observed country series.

    Grid-level simulated CF is collapsed to one capacity-weighted country CF
    per timestep (NaN-skipping, reweighting on the present grid points), then
    both sides are compared as monthly means, matching the legacy
    country-level metric.
    """
    grid_cols = [c for c in sim_cf.columns if c != "time"]
    cap = turb_info.assign(ID=turb_info["ID"].astype(str)).set_index("ID")["capacity"]
    valid = [c for c in grid_cols if str(c) in cap.index]
    caps = cap[[str(c) for c in valid]].to_numpy(float)

    sim = sim_cf.copy()
    sim["time"] = pd.to_datetime(sim["time"])
    vals = sim[valid].to_numpy(float)
    present = ~np.isnan(vals)
    wsum = np.where(present, caps, 0.0).sum(axis=1)
    country = np.where(wsum > 0, np.where(present, vals * caps, 0.0).sum(axis=1) / wsum, np.nan)
    sim_country = pd.DataFrame({"time": sim["time"], "cf_sim": country})
    sim_country["ym"] = sim_country["time"].dt.to_period("M")
    sim_m = sim_country.groupby("ym")["cf_sim"].mean()

    obs = obs_country.copy()
    obs["time"] = pd.to_datetime(obs["time"])
    obs["ym"] = obs["time"].dt.to_period("M")
    obs_m = obs.groupby("ym")["obs"].mean().rename("cf_obs")

    return _error_metrics(pd.concat([sim_m, obs_m], axis=1).dropna())


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
