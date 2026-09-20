"""Standard skill metrics for the validation harness.

All functions take one tidy frame with columns ``ID``, ``cf_sim``,
``cf_obs``, ``capacity`` and (where noted) ``time`` or ``month``. Legacy
``vwf.metrics`` is untouched; the harness reports from here.

Granularity honesty (design §2): regions whose rows pseudo-replicate a
coarser observation unit (UK: farm generation equally pre-split across
turbine rows) must be collapsed to independent stations with
:func:`collapse_pseudo_replicates` before distribution comparisons, and
``n_units`` always counts independent units, not rows.

Conditions compared with each other are scored on the same rows: see
:func:`restrict_to_common_rows`.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from vwf.harness.regions import RegionSpec
from vwf.metrics import weighted_mean, weighted_mean_by

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
    #
    # The weights run over the rows that HAVE a simulated value, and the
    # weighted sum keeps NaN when none does (``min_count=1``). A row without a
    # value belongs to a cluster whose factor was refused; dividing by the
    # station's whole capacity instead would scale the station's capacity
    # factor down by the missing share, and a station with no value at all
    # would come out as exactly zero rather than missing, because an empty
    # pandas sum is 0.0. That is what happened to four UK stations, 48
    # station-months, when the accepted-years rule refused two clusters: they
    # scored a corrected CF of 0.0 against an observed 0.38 and stayed in the
    # scored rows, because only a NaN leaves the common rows. ``capacity``
    # stays the station's whole capacity: it weights the station in the fleet
    # metric, and a capacity factor does not depend on how much of the station
    # carries a value.
    keys = ["ID", *time_cols]
    grouped = df.groupby(keys, as_index=False).agg(
        cf_obs=("cf_obs", "first"),
        cf_sim=("cf_sim", "mean"),  # replaced below; the shape is what is kept
        capacity=("capacity", "sum"),
    )
    station_cf = weighted_mean_by(df, "cf_sim", "capacity", keys).rename("_cf")
    grouped = grouped.merge(station_cf.reset_index(), on=keys, how="left")
    grouped["cf_sim"] = grouped.pop("_cf")
    return grouped


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

    mbe = weighted_mean(diff, weights)
    mae = weighted_mean(np.abs(diff), weights)
    rmse = float(np.sqrt(weighted_mean(diff**2, weights)))
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


def restrict_to_common_rows(
    frames: dict[str, pd.DataFrame],
    keys: list[str],
    *,
    weight: str | None = "capacity",
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Restrict every condition to the rows that every condition can score.

    Each scoring call drops its own incomplete rows. Scored one at a time, two
    conditions are therefore compared on different rows whenever one of them
    cannot simulate some units. A corrected condition has no value for units
    in a cluster whose offset fit failed, so its score silently excluded
    exactly the units the correction failed on, while the uncorrected score
    kept them. That is not a comparison. Here a row is scored only if it is
    complete (``cf_sim`` and ``cf_obs`` present, and the weight when given) in
    every condition.

    Args:
        frames: Condition name to paired frame. Every frame carries ``keys``,
            ``cf_sim`` and ``cf_obs``, and ``weight`` when given.
        keys: Columns identifying a row across conditions, for example
            ``["ID", "year", "month"]`` or ``["ym"]``.
        weight: Column weighting the excluded share, or None to count rows.

    Returns:
        The frames restricted to the common complete rows, and a table of the
        rows excluded from comparison: those complete in at least one condition
        but not in all. It has the key columns, ``weight`` when given, and
        ``missing_in``, the conditions that could not score the row, joined by
        ";". A row complete in no condition (a missing observation) is not
        listed, since no condition was ever scored on it.
    """
    required = ["cf_sim", "cf_obs"] + ([weight] if weight else [])
    complete = {
        name: frame.dropna(subset=required).drop_duplicates(subset=keys)[keys]
        for name, frame in frames.items()
    }
    key_index = {
        name: pd.MultiIndex.from_frame(rows.astype(object)) for name, rows in complete.items()
    }
    names = list(frames)
    common = key_index[names[0]]
    union = key_index[names[0]]
    for name in names[1:]:
        common = common.intersection(key_index[name])
        union = union.union(key_index[name])

    restricted = {}
    for name, frame in frames.items():
        idx = pd.MultiIndex.from_frame(frame[keys].astype(object))
        keep = idx.isin(common) & frame[required].notna().all(axis=1).to_numpy()
        restricted[name] = frame.loc[keep].reset_index(drop=True)

    excluded_keys = union.difference(common)
    columns = [*keys, *([weight] if weight else []), "missing_in"]
    if len(excluded_keys) == 0:
        return restricted, pd.DataFrame(columns=columns)

    excluded = excluded_keys.to_frame(index=False)
    excluded.columns = keys
    excluded["missing_in"] = [
        ";".join(n for n in names if key not in key_index[n]) for key in excluded_keys
    ]
    if weight:
        # A row's weight is the same in every condition that has it.
        weights = pd.concat(
            [f.dropna(subset=required)[[*keys, weight]] for f in frames.values()]
        ).drop_duplicates(subset=keys)
        weights[keys] = weights[keys].astype(object)
        excluded = excluded.astype({k: object for k in keys}).merge(weights, on=keys, how="left")
    return restricted, excluded[columns]


def summarise_exclusions(
    frames: dict[str, pd.DataFrame],
    excluded: pd.DataFrame,
    keys: list[str],
    *,
    weight: str | None = "capacity",
    unit: str | None = "ID",
) -> dict:
    """Summarise :func:`restrict_to_common_rows` for the run record.

    Returns the number of rows any condition could score, the number scored,
    the excluded share of those rows (by ``weight``, or by count when
    ``weight`` is None), and the units with every row excluded.
    """
    required = ["cf_sim", "cf_obs"] + ([weight] if weight else [])
    union = pd.concat(
        [f.dropna(subset=required)[[*keys, *([weight] if weight else [])]] for f in frames.values()]
    ).drop_duplicates(subset=keys)
    n_union, n_excluded = len(union), len(excluded)
    if weight:
        total = float(union[weight].sum())
        share = float(excluded[weight].sum()) / total if total > 0 else 0.0
    else:
        share = n_excluded / n_union if n_union else 0.0
    dropped_units: list[str] = []
    if unit and n_excluded:
        rows_per_unit = union.groupby(unit).size()
        excluded_per_unit = excluded.groupby(unit).size()
        dropped_units = sorted(
            str(u) for u, n in excluded_per_unit.items() if n == rows_per_unit.get(u, 0)
        )
    return {
        "n_rows_scorable": int(n_union),
        "n_rows_scored": int(n_union - n_excluded),
        "n_rows_excluded": int(n_excluded),
        "excluded_share": share,
        "units_wholly_excluded": dropped_units,
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
