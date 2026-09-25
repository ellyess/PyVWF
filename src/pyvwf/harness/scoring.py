"""Scoring for harness evaluate runs: paired frames, scopes and error metrics.

Split from ``pyvwf.harness.driver``, which calls these to score each variant of a
run on the rows every variant can score. The fleet scope's metrics come from
``pyvwf.harness.skill``; the national and per-zone scopes pair capacity-weighted
simulated aggregates with the observed series here.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pyvwf.harness.skill import (
    restrict_to_common_rows,
    skill_metrics,
    summarise_exclusions,
)
from pyvwf.metrics import weighted_mean


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


def _record_off_curve(variants: list[dict], code: str) -> dict:
    """Each variant's off-curve record, for the manifest, with a warning if any."""
    record = {v["label"]: v.get("tail", {}) for v in variants}
    hit = {
        label: r
        for label, r in record.items()
        if r and (r["off_curve_below_share"] or r["off_curve_above_share"] or r["no_speed_share"])
    }
    if hit:
        worst = max(
            hit,
            key=lambda k: (
                hit[k]["off_curve_below_share"]
                + hit[k]["off_curve_above_share"]
                + hit[k]["no_speed_share"]
            ),
        )
        r = hit[worst]
        warnings.warn(
            f"{code}: {len(hit)} variant(s) have simulated values the power curves could "
            f"not convert; worst {worst}: {r['off_curve_below_share']:.2%} below the "
            f"curve, {r['off_curve_above_share']:.2%} above it, {r['no_speed_share']:.2%} "
            f"with no speed, by capacity, and {r['unit_months_partly_missing']} unit-months "
            "scored on only some of their steps. They are missing, not zero; see the "
            "off_curve block in the manifest."
        )
    return record


#: Per scope: the columns that identify a row across variants, the weight
#: column (None for the unweighted monthly aggregates) and the unit column.
_SCOPE_KEYS: dict[str, tuple[list[str], str | None, str | None]] = {
    "fleet": (["ID", "year", "month"], "capacity", "ID"),
    "national": (["ym"], None, None),
    "per-zone": (["cluster", "ym"], None, "cluster"),
}


SCORING_EXCLUSIONS_NAME = "scoring_exclusions.csv"


def _score_on_common_rows(
    variants: list[dict], code: str, run_dir: Path
) -> tuple[list[dict], dict]:
    """Score every variant of one run on the rows all of them can score.

    Scored one at a time, a variant with no value for some units (a corrected
    variant whose cluster failed to fit) was compared with the uncorrected
    variant on a different set of rows. The rows dropped were the hard ones,
    so the comparison flattered the correction. Each scope is now restricted
    to its common complete rows before any metric is computed. The rows
    excluded, and the variants that lacked them, are written to
    ``scoring_exclusions.csv``. The excluded share goes into ``metrics.csv``,
    and a summary into the manifest.

    Args:
        variants: One dict per variant, in output order, with ``label`` (the
            name used in the exclusions file), ``head`` (the leading metrics
            columns), ``extra`` (trailing columns, such as fit quality) and
            ``pairs`` (scope name to paired frame), and optionally ``tail``
            (columns appended after ``extra``, such as the off-curve record).
        code: Region code, for warnings.
        run_dir: Where the exclusions file is written.

    Returns:
        The metrics rows, and the per-scope summary for the manifest.
    """
    scopes = list(variants[0]["pairs"])
    scored: dict[str, dict[str, pd.DataFrame]] = {}
    summaries: dict[str, dict] = {}
    exclusion_tables = []
    for scope in scopes:
        keys, weight, unit = _SCOPE_KEYS[scope]
        frames = {v["label"]: v["pairs"][scope] for v in variants}
        scored[scope], excluded = restrict_to_common_rows(frames, keys, weight=weight)
        summaries[scope] = summarise_exclusions(frames, excluded, keys, weight=weight, unit=unit)
        if len(excluded):
            exclusion_tables.append(excluded.assign(scope=scope))
            summary = summaries[scope]
            warnings.warn(
                f"{code} {scope}: {summary['n_rows_excluded']} row(s), "
                f"{summary['excluded_share']:.1%} of what any variant could score, "
                "lack a value in some variant and are excluded from every "
                f"variant's score; see {run_dir / SCORING_EXCLUSIONS_NAME}."
            )

    # Written even when empty, so a missing file never has to be interpreted.
    columns = ["scope", "ID", "year", "month", "cluster", "ym", "capacity", "missing_in"]
    exclusions = pd.concat(exclusion_tables, ignore_index=True) if exclusion_tables else None
    (
        exclusions[[c for c in columns if c in exclusions.columns]]
        if exclusions is not None
        else pd.DataFrame(columns=columns)
    ).to_csv(run_dir / SCORING_EXCLUSIONS_NAME, index=False)

    rows = []
    for v in variants:
        for scope in scopes:
            frame = scored[scope][v["label"]]
            if scope == "fleet":
                metrics = skill_metrics(frame)
            elif scope == "per-zone":
                metrics = _zonal_metrics(frame)
            else:
                metrics = _error_metrics(frame)
            rows.append({**v["head"], "scope": scope, **metrics, **v["extra"], **v.get("tail", {})})
    return rows, summaries


def _zone_aggregate(sim_cf: pd.DataFrame, members: pd.DataFrame) -> pd.Series:
    """Capacity-weighted monthly mean CF over one set of grid points."""
    cap = members.assign(ID=members["ID"].astype(str)).set_index("ID")["capacity"]
    valid = [c for c in sim_cf.columns if c != "time" and str(c) in cap.index]
    if not valid:
        return pd.Series(dtype=float)

    caps = cap[[str(c) for c in valid]].to_numpy(float)
    vals = sim_cf[valid].to_numpy(float)
    agg = weighted_mean(vals, caps, axis=1)

    frame = pd.DataFrame({"time": pd.to_datetime(sim_cf["time"]), "cf_sim": agg})
    return frame.groupby(frame["time"].dt.to_period("M"))["cf_sim"].mean()


def _error_metrics(merged: pd.DataFrame) -> dict:
    """MBE, MAE, RMSE and correlation over paired sim/obs columns."""
    if merged.empty:
        return {
            "mbe": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "pearson_r": float("nan"),
            "n_months": 0,
        }
    diff = merged["cf_sim"] - merged["cf_obs"]
    r = (
        float(np.corrcoef(merged["cf_sim"], merged["cf_obs"])[0, 1])
        if len(merged) > 1
        else float("nan")
    )
    return {
        "mbe": float(diff.mean()),
        "mae": float(diff.abs().mean()),
        "rmse": float(np.sqrt((diff**2).mean())),
        "pearson_r": r,
        "n_months": int(len(merged)),
    }


def _zonal_skill(sim_cf: pd.DataFrame, obs_zonal: pd.DataFrame, turb_info: pd.DataFrame) -> dict:
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
    return _zonal_metrics(_zonal_pairs(sim_cf, obs_zonal, turb_info).dropna())


def _zonal_pairs(
    sim_cf: pd.DataFrame, obs_zonal: pd.DataFrame, turb_info: pd.DataFrame
) -> pd.DataFrame:
    """Per-zone monthly simulated and observed CF, one row per (zone, month).

    Rows keep a missing side as NaN, so the caller can score several
    conditions on the same rows (``restrict_to_common_rows``).
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
        merged = pd.concat([sim_m, obs_m], axis=1)
        merged["cluster"] = cluster
        pairs.append(merged)

    if not pairs:
        return pd.DataFrame(columns=["cluster", "ym", "cf_sim", "cf_obs"])
    pooled = pd.concat(pairs).rename_axis("ym").reset_index()
    return pooled[["cluster", "ym", "cf_sim", "cf_obs"]]


def _zonal_metrics(pooled: pd.DataFrame) -> dict:
    if pooled.empty:
        return {**_error_metrics(pd.DataFrame()), "n_zones": 0}
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
    return _error_metrics(_country_pairs(sim_cf, obs_country, turb_info).dropna())


def _country_pairs(
    sim_cf: pd.DataFrame, obs_country: pd.DataFrame, turb_info: pd.DataFrame
) -> pd.DataFrame:
    """Monthly national simulated and observed CF, one row per month (``ym``).

    Rows keep a missing side as NaN, so the caller can score several
    conditions on the same months (``restrict_to_common_rows``).
    """
    grid_cols = [c for c in sim_cf.columns if c != "time"]
    cap = turb_info.assign(ID=turb_info["ID"].astype(str)).set_index("ID")["capacity"]
    valid = [c for c in grid_cols if str(c) in cap.index]
    caps = cap[[str(c) for c in valid]].to_numpy(float)

    sim = sim_cf.copy()
    sim["time"] = pd.to_datetime(sim["time"])
    vals = sim[valid].to_numpy(float)
    country = weighted_mean(vals, caps, axis=1)
    sim_country = pd.DataFrame({"time": sim["time"], "cf_sim": country})
    sim_country["ym"] = sim_country["time"].dt.to_period("M")
    sim_m = sim_country.groupby("ym")["cf_sim"].mean()

    obs = obs_country.copy()
    obs["time"] = pd.to_datetime(obs["time"])
    obs["ym"] = obs["time"].dt.to_period("M")
    obs_m = obs.groupby("ym")["obs"].mean().rename("cf_obs")

    return pd.concat([sim_m, obs_m], axis=1).rename_axis("ym").reset_index()
