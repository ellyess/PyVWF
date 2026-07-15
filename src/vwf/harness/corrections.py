"""Correction-model interface for the validation harness.

Alternative correction formulations (Phase 3) register here and are compared
against the affine baseline through one interface. The baseline,
:class:`AffineWindCorrection`, is a thin delegate to the validated legacy
code (:mod:`vwf.correction`, :mod:`vwf.data`, :mod:`vwf.wind`) — it adds no
maths of its own, and a golden regression test pins its output to the legacy
pipeline's bit for bit.

Season handling: every entry point takes an optional ``seasons`` mapping
(season name → month list, from :attr:`vwf.harness.regions.RegionSpec.seasons`).
``None`` reproduces the legacy Northern-Hemisphere behaviour exactly; the
harness always passes the region's explicit definitions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np
import pandas as pd

import vwf.correction as correction
import vwf.wind as wind
from vwf.data import cluster_train_set, format_bc_factors

_CORRECTIONS: dict[str, type["CorrectionModel"]] = {}


def register_correction(cls: type["CorrectionModel"]) -> type["CorrectionModel"]:
    """Class decorator adding a correction model to the registry."""
    if not (isinstance(cls, type) and issubclass(cls, CorrectionModel)):
        raise TypeError(f"{cls!r} is not a CorrectionModel subclass")
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must define a non-empty 'name'")
    existing = _CORRECTIONS.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(f"A correction model named {name!r} is already registered")
    _CORRECTIONS[name] = cls
    return cls


def available_corrections() -> tuple[str, ...]:
    """Names of every registered correction model, sorted."""
    return tuple(sorted(_CORRECTIONS))


def get_correction(name: str) -> "CorrectionModel":
    """Construct a registered correction model by name."""
    try:
        cls = _CORRECTIONS[name]
    except KeyError:
        raise KeyError(
            f"Unknown correction model {name!r}. Registered: {available_corrections()}"
        ) from None
    return cls()


class CorrectionModel(ABC):
    """One correction formulation: fit factors from training data, apply them.

    The contract mirrors the legacy affine pipeline so the baseline can
    delegate without translation:

    - :meth:`fit` consumes the ``train_set`` outputs and returns
      ``(factors, clus_info)`` where ``factors`` is the long-format table the
      apply path merges on (``cluster``, ``<time_res>``, model parameters).
    - :meth:`apply` simulates corrected wind speeds and capacity factors for
      a fleet using a fitted factors table.

    Variants may put whatever parameter columns they need in ``factors``; the
    only fixed columns are ``cluster`` and the time-slice column.
    """

    name: ClassVar[str]

    @abstractmethod
    def fit(
        self,
        gen_cf: pd.DataFrame,
        turb_info: pd.DataFrame,
        reanalysis: Any,
        power_curves: pd.DataFrame,
        *,
        num_clusters: int,
        time_res: str,
        seasons: Mapping[str, Sequence[int]] | None = None,
        obs_level: str = "turbine",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fit correction factors; return ``(factors, clus_info)``."""

    @abstractmethod
    def apply(
        self,
        reanalysis: Any,
        clus_info: pd.DataFrame,
        power_curves: pd.DataFrame,
        factors: pd.DataFrame,
        time_res: str,
        *,
        seasons: Mapping[str, Sequence[int]] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate with corrections; return ``(wind_speed, capacity_factor)``."""


def _override_season_column(
    df: pd.DataFrame, seasons: Mapping[str, Sequence[int]] | None
) -> pd.DataFrame:
    """Re-derive the ``season`` column from explicit definitions.

    The training frame arrives with a ``season`` column already assigned by
    the legacy NH mapping; for a region with its own definitions the column
    is rebuilt from ``month`` before any grouping happens.
    """
    if seasons is None or "season" not in df.columns:
        return df
    if "month" not in df.columns:
        raise ValueError(
            "cannot re-derive the season column from explicit season "
            "definitions: the training frame has no 'month' column"
        )
    df = df.copy()
    month_to_season = {m: name for name, months in seasons.items() for m in months}
    df["season"] = df["month"].map(month_to_season)
    return df


@register_correction
class AffineWindCorrection(CorrectionModel):
    """The validated baseline: ``cor_ws = scalar * unc_ws + offset``.

    Pure delegation. Turbine-level fit = cluster_train_set → find_offset
    (sequential) → format_bc_factors; country-level fit = cluster_train_set →
    find_offsets_country_level (joint) → format_bc_factors; apply =
    wind.simulate_wind. With ``seasons=None`` the turbine-level path is the
    exact legacy code path (pinned by the golden regression test in
    tests/test_harness_corrections.py).
    """

    name: ClassVar[str] = "affine-wind"

    def fit(
        self,
        gen_cf: pd.DataFrame,
        turb_info: pd.DataFrame,
        reanalysis: Any,
        power_curves: pd.DataFrame,
        *,
        num_clusters: int,
        time_res: str,
        seasons: Mapping[str, Sequence[int]] | None = None,
        obs_level: str = "turbine",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if obs_level not in ("turbine", "country"):
            raise ValueError(f"obs_level must be 'turbine' or 'country', got {obs_level!r}")

        gen_cf = _override_season_column(gen_cf, seasons)
        train_bias_df, clus_info = cluster_train_set(
            gen_cf, time_res, num_clusters, turb_info, obs_level=obs_level
        )

        if obs_level == "country":
            # Delegation mirror of PyVWF.train's country branch: one country
            # observation per period, all cluster offsets optimised jointly
            # (correction.find_offsets_country_level; identifiability is RQ4's
            # question, not changed here).
            unique_periods = train_bias_df[["year", time_res]].drop_duplicates()
            offsets_rows: list[dict[str, Any]] = []
            for _, period in unique_periods.iterrows():
                period_data = train_bias_df[
                    (train_bias_df["year"] == period["year"])
                    & (train_bias_df[time_res] == period[time_res])
                ]
                offsets = correction.find_offsets_country_level(
                    year=period["year"],
                    time_slice=period[time_res],
                    obs_country_cf=period_data["obs"].iloc[0],
                    scalars_by_cluster=dict(
                        zip(period_data["cluster"], period_data["scalar"])
                    ),
                    turb_info=clus_info,
                    reanalysis=reanalysis,
                    powerCurveFile=power_curves,
                    seasons=seasons,
                )
                offsets_rows.extend(
                    {
                        "year": period["year"],
                        time_res: period[time_res],
                        "cluster": cluster_id,
                        "offset": offset,
                    }
                    for cluster_id, offset in offsets.items()
                )
            train_bias_df = train_bias_df.drop(columns=["offset"], errors="ignore")
            train_bias_df = train_bias_df.merge(
                pd.DataFrame(offsets_rows), on=["year", time_res, "cluster"], how="left"
            )
            return format_bc_factors(train_bias_df, time_res), clus_info

        # Sequential offset fit, mirroring PyVWF.train(dask_n_workers=0):
        # optimise rows with usable observations, zero-fill obs == 0, keep
        # NaN observations NaN.
        valid = train_bias_df[
            train_bias_df["obs"].notna() & (train_bias_df["obs"] > 0)
        ].copy()
        if len(valid) == 0:
            train_bias_df["offset"] = 0.0
        else:
            valid["offset"] = valid.apply(
                correction.find_offset,
                args=(clus_info, reanalysis, power_curves),
                seasons=seasons,
                axis=1,
            )
            zero_obs = train_bias_df[train_bias_df["obs"] == 0].copy()
            zero_obs["offset"] = 0.0
            nan_obs = train_bias_df[train_bias_df["obs"].isna()].copy()
            nan_obs["offset"] = np.nan
            train_bias_df = pd.concat(
                [valid, zero_obs, nan_obs], ignore_index=True
            ).sort_index()

        factors = format_bc_factors(train_bias_df, time_res)
        return factors, clus_info

    def apply(
        self,
        reanalysis: Any,
        clus_info: pd.DataFrame,
        power_curves: pd.DataFrame,
        factors: pd.DataFrame,
        time_res: str,
        *,
        seasons: Mapping[str, Sequence[int]] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return wind.simulate_wind(
            reanalysis, clus_info, power_curves, factors, time_res, seasons=seasons
        )
