"""Shared pieces of the physics-informed run drivers.

``scripts/pinn/e1_loro.py`` (leave one region out) and ``scripts/pinn/loco.py``
(leave one country out) resolve configs, record what each region was built
from, and score conditions on common rows the same way. Keeping one copy is
what makes their records comparable. Nothing here imports torch.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from vwf.harness.driver import _error_metrics
from vwf.harness.skill import (
    collapse_pseudo_replicates, restrict_to_common_rows, skill_metrics,
    summarise_exclusions,
)

CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "regions"
UNIT_KEYS = ["ID", "year", "month"]
MONTH_KEYS = ["year", "month"]


def resolve_configs(codes, overrides: list[str], configs: Path = CONFIGS) -> dict[str, Path]:
    """Config path per region: ``--config CODE=PATH`` where given, else maintained."""
    named = {}
    for item in overrides:
        code, sep, path = item.partition("=")
        if not sep or not path:
            raise SystemExit(f"--config expects CODE=PATH, got {item!r}")
        named[code] = Path(path)
    return {c: named.get(c, configs / f"{c.lower().replace('-', '_')}.toml")
            for c in codes}


def config_record(paths: dict[str, Path]) -> dict:
    """Each config's path and sha256, so the manifest names the exact file."""
    return {c: {"path": str(p),
                "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
            for c, p in paths.items()}


def region_record(r) -> dict[str, Any]:
    """What one region/split's tensors were built from and what was dropped."""
    return {
        **r.era5_record,
        "level": getattr(r, "level", "turbine"),
        "fleet": dict(getattr(r, "fleet_record", {})),
        "units_simulated": int(r.n_units),
        "units_dropped_no_wind": len(r.dropped_ids),
        "capacity_share_dropped_no_wind": float(r.dropped_capacity_share),
        "ids_dropped_no_wind": list(r.dropped_ids[:10]),
        "isolated_cells_filled": int(r.filled_cells),
    }


def score_on_common_rows(conditions: dict, spec) -> tuple[dict, pd.DataFrame, dict]:
    """Score every condition of one holdout on the unit-months all can score.

    Pseudo-replicates are collapsed first, as the harness does before it
    restricts, so the common rows are stations rather than turbine-shaped rows.

    Args:
        conditions: Label to ``(arm, seed, frame)`` with per-unit frames.
        spec: The holdout's region config.

    Returns:
        Metrics per label, the excluded rows, and the exclusion summary.
    """
    pairs = {label: collapse_pseudo_replicates(frame, spec)
             for label, (_, _, frame) in conditions.items()}
    restricted, excluded = restrict_to_common_rows(pairs, UNIT_KEYS, weight="capacity")
    summary = summarise_exclusions(pairs, excluded, UNIT_KEYS, weight="capacity", unit="ID")
    metrics = {label: skill_metrics(frame) for label, frame in restricted.items()}
    return metrics, excluded, summary


def score_national_on_common_months(conditions: dict) -> tuple[dict, pd.DataFrame, dict]:
    """Score national monthly series on the months every condition can score.

    The metrics are the harness's own national ones (MBE, MAE, RMSE, Pearson r
    and the month count, unweighted over months), so a country fold is scored
    the way a scorecard country row is.

    Args:
        conditions: Label to ``(arm, seed, frame)`` with (year, month, cf_sim,
            cf_obs) frames.

    Returns:
        Metrics per label, the excluded months, and the exclusion summary.
    """
    frames = {label: frame for label, (_, _, frame) in conditions.items()}
    restricted, excluded = restrict_to_common_rows(frames, MONTH_KEYS, weight=None)
    summary = summarise_exclusions(frames, excluded, MONTH_KEYS, weight=None, unit=None)
    metrics = {label: _error_metrics(frame) for label, frame in restricted.items()}
    return metrics, excluded, summary
