"""Standard skill metrics for the validation harness.

All functions take one tidy frame with columns ``ID``, ``cf_sim``,
``cf_obs``, ``capacity`` and (where noted) ``time`` or ``month``. Legacy
``vwf.metrics`` is untouched; the harness reports from here.

Granularity honesty (design §2): regions whose rows pseudo-replicate a
coarser observation unit (UK: farm generation equally pre-split across
turbine rows) must be collapsed to independent stations with
:func:`collapse_pseudo_replicates` before distribution comparisons, and
``n_units`` always counts independent units, not rows.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from vwf.harness.regions import RegionSpec

_REQUIRED = ("ID", "cf_sim", "cf_obs", "capacity")


def _check_columns(df: pd.DataFrame, extra: tuple[str, ...] = ()) -> None:
    missing = [c for c in (*_REQUIRED, *extra) if c not in df.columns]
    if missing:
        raise ValueError(f"skill frame is missing required columns {missing}")


def station_ids(ids: pd.Series, spec: RegionSpec | None = None) -> pd.Series:
    """Map row IDs to independent-station IDs.

    Uses ``spec.station_id_regex`` (one capture group = the station part)
    when set; IDs that do not match, or specs without a regex, map to
    themselves.
    """
    ids = ids.astype(str)
    if spec is None or not spec.station_id_regex:
        return ids
    pattern = re.compile(spec.station_id_regex)

    def _one(value: str) -> str:
        match = pattern.match(value)
        return match.group(1) if match else value

    return ids.map(_one)


def collapse_pseudo_replicates(df: pd.DataFrame, spec: RegionSpec) -> pd.DataFrame:
    """Collapse pseudo-replicated rows to one row per independent station.

    No-op for regions without ``pseudo_replicated_rows``. For regions with it
    (UK): rows sharing a station carry the SAME observation (farm generation
    equally pre-split), so per group-and-time we keep ``cf_obs`` as-is,
    average ``cf_sim`` capacity-weighted, and sum ``capacity``.

    The frame is grouped per time step when a ``time`` or ``month``/``year``
    column is present, else across the whole frame.
    """
    _check_columns(df)
    if not spec.pseudo_replicated_rows:
        return df.copy()

    df = df.copy()
    df["ID"] = station_ids(df["ID"], spec)
    time_cols = [c for c in ("time", "year", "month") if c in df.columns]

    # Identity guard: the collapse assumes every row of a (station, time)
    # group carries the SAME observation (equal-split pseudo-replicates, the
    # verified UK structure). Divergent obs within a group means the rows are
    # NOT replicates of one measurement; averaging them away here would
    # silently fabricate an observation, so refuse instead.
    obs_group = df.groupby(["ID", *time_cols])["cf_obs"]
    spread = obs_group.max() - obs_group.min()
    divergent = spread[spread > 1e-9]
    if not divergent.empty:
        raise ValueError(
            f"collapse_pseudo_replicates: {len(divergent)} station-time group(s) have "
            f"divergent cf_obs across their rows (first: {divergent.index[0]!r}, "
            f"spread {float(divergent.iloc[0]):.3g}). These rows are not pseudo-"
            "replicates of one observation; check station_id_regex or the data."
        )

    # Vectorised capacity-weighted collapse (groupby.apply(include_groups=...)
    # would need pandas >= 2.2; the project supports >= 2.0).
    df["_sim_x_cap"] = df["cf_sim"] * df["capacity"]
    grouped = df.groupby(["ID", *time_cols], as_index=False).agg(
        cf_obs=("cf_obs", "first"),
        cf_sim=("cf_sim", "mean"),  # fallback where total capacity is zero
        capacity=("capacity", "sum"),
        _sim_x_cap=("_sim_x_cap", "sum"),
    )
    positive = grouped["capacity"] > 0
    grouped.loc[positive, "cf_sim"] = (
        grouped.loc[positive, "_sim_x_cap"] / grouped.loc[positive, "capacity"]
    )
    return grouped.drop(columns=["_sim_x_cap"])


def skill_metrics(df: pd.DataFrame, *, weighted: bool = True) -> dict[str, float]:
    """Compute the standard skill table for one (region, variant) pair.

    Args:
        df: Tidy frame (``ID``, ``cf_sim``, ``cf_obs``, ``capacity``); one row
            per unit per time step. Collapse pseudo-replicates first where
            the region requires it.
        weighted: Capacity-weight MBE/MAE/RMSE and the EMD. Pearson r is
            always unweighted (documented choice: r describes co-variability,
            not fleet-aggregate error).

    Returns:
        Dict with ``mbe``, ``mae``, ``rmse``, ``pearson_r``, ``emd``,
        ``n_units`` (independent IDs) and ``n_samples`` (rows). NaN pairs are
        dropped; an empty frame raises.
    """
    _check_columns(df)
    data = df.dropna(subset=["cf_sim", "cf_obs", "capacity"])
    if data.empty:
        raise ValueError("skill frame has no complete (cf_sim, cf_obs, capacity) rows")

    sim = data["cf_sim"].to_numpy(dtype=float)
    obs = data["cf_obs"].to_numpy(dtype=float)
    diff = sim - obs
    weights = data["capacity"].to_numpy(dtype=float) if weighted else np.ones_like(diff)
    if weights.sum() <= 0:
        raise ValueError("capacity weights sum to zero")

    mbe = float(np.average(diff, weights=weights))
    mae = float(np.average(np.abs(diff), weights=weights))
    rmse = float(np.sqrt(np.average(diff**2, weights=weights)))
    if len(data) > 1 and np.std(sim) > 0 and np.std(obs) > 0:
        pearson = float(np.corrcoef(sim, obs)[0, 1])
    else:
        pearson = float("nan")
    emd = float(wasserstein_distance(obs, sim, u_weights=weights, v_weights=weights))

    return {
        "mbe": mbe,
        "mae": mae,
        "rmse": rmse,
        "pearson_r": pearson,
        "emd": emd,
        "n_units": int(data["ID"].nunique()),
        "n_samples": int(len(data)),
    }


def seasonal_cycle_rmse(df: pd.DataFrame, *, weighted: bool = True) -> float:
    """RMSE of the mean monthly climatology (12 sim-vs-obs differences).

    Requires a ``month`` column, or a ``time`` column to derive it from.
    """
    _check_columns(df)
    df = df.copy()
    if "month" not in df.columns:
        if "time" not in df.columns:
            raise ValueError("seasonal_cycle_rmse needs a 'month' or 'time' column")
        df["month"] = pd.to_datetime(df["time"]).dt.month

    data = df.dropna(subset=["cf_sim", "cf_obs", "capacity"]).copy()
    if data.empty:
        raise ValueError("seasonal_cycle_rmse: no complete rows")

    weights = data["capacity"] if weighted else pd.Series(1.0, index=data.index)
    data["_w"] = weights
    data["_obs_x_w"] = data["cf_obs"] * weights
    data["_sim_x_w"] = data["cf_sim"] * weights
    monthly = data.groupby("month")[["_obs_x_w", "_sim_x_w", "_w"]].sum()
    diff = (monthly["_sim_x_w"] - monthly["_obs_x_w"]) / monthly["_w"]
    return float(np.sqrt((diff**2).mean()))
