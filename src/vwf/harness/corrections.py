"""Correction-model interface for the validation harness.

Alternative correction formulations (Phase 3) register here and are compared
against the affine baseline through one interface. The baseline,
:class:`AffineWindCorrection`, is a thin delegate to the validated legacy
code (:mod:`vwf.correction`, :mod:`vwf.data`, :mod:`vwf.wind`): it adds no
maths of its own, and a golden regression test pins its output to the legacy
pipeline's bit for bit.

Season handling: every entry point takes an optional ``seasons`` mapping
(season name → month list, from :attr:`vwf.harness.regions.RegionSpec.seasons`).
``None`` reproduces the legacy Northern-Hemisphere behaviour exactly; the
harness always passes the region's explicit definitions.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np
import pandas as pd

import vwf.correction as correction
import vwf.wind as wind
from vwf.data import (
    cluster_train_set,
    country_obs_is_per_cluster,
    format_bc_factors,
)

#: Rows below this count fit sequentially regardless of the worker setting: the
#: dask cluster startup and scatter cost only pays off on large offset fits.
_PARALLEL_OFFSET_MIN_ROWS = 500

#: Physically defensible range for an affine wind scalar. A correction is meant
#: to repair a reanalysis bias of tens of percent, not rebuild the wind field, so
#: a scalar outside this band means the fit is compensating for something the
#: model cannot represent (see docs/findings/hourly_resolution_test.md, where an
#: Atacama cluster fitted a scalar of 80).
#:
#: Calibrated against 373,179 fitted scalars across every region in
#: output/validation (docs/findings/scalar_bounds_calibration.md), and the two
#: bounds are deliberately ASYMMETRIC because they are not equally informative:
#:
#: - The CEILING discriminates sharply. Every healthy region tops out just below
#:   3.0 (UK 2.86, DK 2.68, IT 2.65, AU-NEM 2.65, NZ 2.18), while the regions
#:   with known pathologies run far past it (CL 150, US 66, AR 26, BR 8.2). A
#:   ceiling of 3.0 sits in the gap between those populations.
#: - The FLOOR discriminates weakly, because the country-level path legitimately
#:   fits low scalars when it compensates for the cube-law overshoot (BE 0.314,
#:   ES 0.330, SE 0.359). A floor of 0.3 produced no false positives on this
#:   data, but only 4.7% clear of Belgium, which is too fine a margin to trust
#:   across other years. It is set at 0.2 so it flags only the unambiguous
#:   (US 0.003, DK 0.090, UK 0.096) and stays quiet otherwise: a warning that
#:   fires on legitimate fits is one people learn to ignore.
PLAUSIBLE_SCALAR = (0.2, 3.0)


def fit_quality(factors: pd.DataFrame, scalar_bounds=PLAUSIBLE_SCALAR) -> dict:
    """Summarise whether a fitted factors table is physically believable.

    Skill metrics alone hide a bad fit: Chile scores as a corrected win at
    monthly resolution while containing a wind scalar of 80 and an offset that
    never converged. Those only become visible at finer resolution or in a
    downstream export, which is too late. This reduces a factors table to the
    few numbers that expose it.

    Args:
        factors: A fitted factors table with ``cluster`` and ``scalar``, and
            usually ``offset``.
        scalar_bounds: Inclusive ``(low, high)`` plausible range for the scalar.

    Returns:
        dict with ``n_clusters``, ``n_implausible_scalar``, ``n_failed_offset``,
        ``max_scalar``, ``min_scalar`` and ``degenerate_clusters`` (a sorted,
        comma-joined string, so it survives a CSV round trip).
    """
    if factors is None or factors.empty or "scalar" not in factors.columns:
        return {
            "n_clusters": 0, "n_implausible_scalar": 0, "n_failed_offset": 0,
            "max_scalar": float("nan"), "min_scalar": float("nan"),
            "degenerate_clusters": "",
        }
    low, high = scalar_bounds
    s = pd.to_numeric(factors["scalar"], errors="coerce")
    bad_scalar = s.notna() & ((s < low) | (s > high))
    # A NaN offset where a scalar was fitted means the offset solver failed and
    # nothing raised; those sites silently drop out of any simulation.
    if "offset" in factors.columns:
        off = pd.to_numeric(factors["offset"], errors="coerce")
        failed_offset = off.isna() & s.notna()
    else:
        failed_offset = pd.Series(False, index=factors.index)

    flagged = sorted(
        {int(c) for c in factors.loc[bad_scalar | failed_offset, "cluster"].dropna()}
    )
    return {
        "n_clusters": int(factors["cluster"].nunique()),
        "n_implausible_scalar": int(bad_scalar.sum()),
        "n_failed_offset": int(failed_offset.sum()),
        "max_scalar": float(s.max()) if s.notna().any() else float("nan"),
        "min_scalar": float(s.min()) if s.notna().any() else float("nan"),
        "degenerate_clusters": ",".join(str(c) for c in flagged),
    }


def _fit_offsets(
    valid: pd.DataFrame,
    clus_info: pd.DataFrame,
    reanalysis: Any,
    power_curves: pd.DataFrame,
    seasons: Mapping[str, Sequence[int]] | None,
) -> pd.Series:
    """Fit each valid row's offset with :func:`correction.find_offset`.

    Sequential by default, which is bit-for-bit the legacy
    ``PyVWF.train(dask_n_workers=0)`` path the golden regression test pins.
    Set ``PYVWF_OFFSET_WORKERS`` above 1 to fan the row-wise fit across that
    many worker processes with :mod:`dask.distributed`, mirroring
    ``PyVWF.train``'s parallel branch (``vwf.py``): the shared reanalysis and
    curves are scattered once and broadcast, then ``find_offset`` runs per
    partition. ``find_offset`` is a pure row function, so the parallel result
    is identical, only faster; large sweeps (high cluster counts, monthly
    slices) are where it matters. Falls back to sequential on a small row
    count or any cluster-startup failure.

    Returns:
        A ``Series`` of offsets indexed like ``valid``.
    """
    def _sequential() -> pd.Series:
        return valid.apply(
            correction.find_offset,
            args=(clus_info, reanalysis, power_curves),
            seasons=seasons,
            axis=1,
        )

    workers = int(os.environ.get("PYVWF_OFFSET_WORKERS", "0") or "0")
    if workers <= 1 or len(valid) < _PARALLEL_OFFSET_MIN_ROWS:
        return _sequential()

    try:
        import dask.dataframe as dd
        from dask.distributed import Client, LocalCluster
    except Exception:
        return _sequential()

    def _fit_partition(df, clus_arg, reanalysis_arg, power_curves_arg):
        return df.apply(
            correction.find_offset,
            args=(clus_arg, reanalysis_arg, power_curves_arg),
            seasons=seasons,
            axis=1,
        )

    npartitions = max(workers * 4, 1)
    try:
        cluster = LocalCluster(
            n_workers=workers, threads_per_worker=1, processes=True
        )
        client = Client(cluster)
        try:
            ddf = dd.from_pandas(valid, npartitions=npartitions)
            clus_arg = client.scatter(clus_info, broadcast=True)
            reanalysis_arg = client.scatter(reanalysis, broadcast=True)
            power_curves_arg = client.scatter(power_curves, broadcast=True)
            ddf["offset"] = ddf.map_partitions(
                _fit_partition,
                clus_arg,
                reanalysis_arg,
                power_curves_arg,
                meta=("offset", "float"),
            )
            # compute preserves the row index, so the caller's index-aligned
            # assignment lands each offset on its own row.
            return ddf.compute()["offset"]
        finally:
            client.close()
            cluster.close()
    except Exception:
        return _sequential()

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
        min_cluster_size: int = 1,
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
        min_cluster_size: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if obs_level not in ("turbine", "country"):
            raise ValueError(f"obs_level must be 'turbine' or 'country', got {obs_level!r}")

        gen_cf = _override_season_column(gen_cf, seasons)
        train_bias_df, clus_info = cluster_train_set(
            gen_cf, time_res, num_clusters, turb_info, obs_level=obs_level,
            min_cluster_size=min_cluster_size,
        )

        if obs_level == "country" and country_obs_is_per_cluster(
            train_bias_df, time_res
        ):
            # Each cluster has its own observation (a zonal source), so each
            # offset is determined by one constraint. That is the turbine-level
            # problem, so it takes the turbine-level solver rather than the
            # joint optimiser, and the result is an estimate rather than
            # wherever L-BFGS-B stopped on an under-determined surface.
            return self._fit_per_cluster_offsets(
                train_bias_df, clus_info, reanalysis, power_curves, time_res, seasons
            )

        if obs_level == "country":
            # Delegation mirror of PyVWF.train's country branch: one country
            # observation per period, all cluster offsets optimised jointly
            # (correction.find_offsets_country_level). Whether those
            # offsets are identifiable at all is an open question studied
            # elsewhere; nothing here changes the estimator.
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
            valid["offset"] = _fit_offsets(
                valid, clus_info, reanalysis, power_curves, seasons
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

    @staticmethod
    def _fit_per_cluster_offsets(
        train_bias_df: pd.DataFrame,
        clus_info: pd.DataFrame,
        reanalysis: Any,
        power_curves: pd.DataFrame,
        time_res: str,
        seasons: Mapping[str, Sequence[int]] | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Solve each cluster's offset against its own observation.

        The same rule the turbine-level path uses: optimise rows with a usable
        observation, zero-fill ``obs == 0``, keep NaN observations NaN.
        ``correction.find_offset`` expects a ``time_slice`` field, which the
        turbine frame carries and this one names after ``time_res``.
        """
        frame = train_bias_df.copy()
        if time_res != "time_slice":
            frame["time_slice"] = frame[time_res]

        valid = frame[frame["obs"].notna() & (frame["obs"] > 0)].copy()
        if len(valid) == 0:
            frame["offset"] = 0.0
        else:
            valid["offset"] = _fit_offsets(
                valid, clus_info, reanalysis, power_curves, seasons
            )
            zero_obs = frame[frame["obs"] == 0].copy()
            zero_obs["offset"] = 0.0
            nan_obs = frame[frame["obs"].isna()].copy()
            nan_obs["offset"] = np.nan
            frame = pd.concat([valid, zero_obs, nan_obs], ignore_index=True).sort_index()

        if time_res != "time_slice":
            frame = frame.drop(columns=["time_slice"])
        return format_bc_factors(frame, time_res), clus_info

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


@register_correction
class ScalarOnlyWindCorrection(AffineWindCorrection):
    """``cor_ws = scalar * unc_ws``: the affine model with the offset held at 0.

    This is the control for the country-level identifiability question. An
    N-cluster country run fits N scalars, which are identified (each is a ratio
    of observed to simulated capacity factor), and N offsets, which against a
    single national observation are under-determined by N-1 and come out
    wherever L-BFGS-B stopped. Any improvement the N-cluster run shows over the
    single-cluster baseline could therefore come entirely from the scalars.

    Pinning the offsets to zero separates the two. If this model matches the
    full affine model, the offsets contribute nothing and the country-level
    joint fit is doing no work worth the ambiguity. If the full model is
    clearly better, the offsets carry real information despite being
    under-determined, and the identifiability problem is worth solving properly
    rather than sidestepping.

    Fitting reuses the parent's path, so the scalars are identical and the two
    models differ in exactly one term. That is the point: it is a control, not
    an alternative estimator.
    """

    name: ClassVar[str] = "scalar-only"

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
        min_cluster_size: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # cluster_train_set already produces the scalars and an offset column of
        # zeros; short-circuiting before any offset solver keeps the scalars
        # bit-identical to the affine model's and skips the expensive fit.
        if obs_level not in ("turbine", "country"):
            raise ValueError(f"obs_level must be 'turbine' or 'country', got {obs_level!r}")

        gen_cf = _override_season_column(gen_cf, seasons)
        train_bias_df, clus_info = cluster_train_set(
            gen_cf, time_res, num_clusters, turb_info, obs_level=obs_level,
            min_cluster_size=min_cluster_size,
        )
        train_bias_df = train_bias_df.copy()
        train_bias_df["offset"] = 0.0
        factors = format_bc_factors(train_bias_df, time_res)
        # format_bc_factors zeroes the offset wherever the scalar was NaN, but
        # nothing guarantees it elsewhere; assert rather than trust.
        assert (factors["offset"] == 0).all(), "scalar-only produced a nonzero offset"
        return factors, clus_info


@register_correction
class ScaledAffineWindCorrection(AffineWindCorrection):
    """Affine wind correction plus a per-cluster availability factor.

    The affine model corrects wind speed: ``cor_ws = scalar * unc_ws + offset``.
    That works when the whole error is a wind-speed bias. It cannot represent a
    loss that reduces output regardless of wind: a farm curtailed 15% of the
    time, or with 92% availability, produces less at every wind speed, and no
    transform of the wind speed says so. Forcing the affine model to absorb such
    a loss through the offset distorts the temporal shape, because the offset
    acts through the convex power curve.

    This adds a third parameter per cluster, an availability ``a`` in ``(0, 1]``,
    that multiplies the affine-corrected capacity factor:
    ``cf = a * curve(scalar * unc_ws + offset)``. It is fitted after the affine
    part, from the residual level the affine correction leaves, and clipped to
    ``(0, 1]`` so it only ever represents a loss (a value above 1 would be the
    wind correction's job, not availability's). One factor per cluster, not per
    slice: availability is a farm property, and a per-slice factor would just
    re-absorb the seasonal wind signal the affine part is there to model.

    This is a genuine model extension for fleets where part of the gap is
    generation loss rather than reanalysis wind error (Patagonia, the Brazilian
    Nordeste), not a tuning knob. See docs/findings/south_america_spatial_bias.md.
    """

    name: ClassVar[str] = "scaled-affine"

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
        min_cluster_size: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        factors, clus_info = super().fit(
            gen_cf, turb_info, reanalysis, power_curves,
            num_clusters=num_clusters, time_res=time_res,
            seasons=seasons, obs_level=obs_level,
            min_cluster_size=min_cluster_size,
        )

        # Simulate the affine-corrected CF on the training winds, aggregate to
        # one capacity-weighted series per cluster, and set each cluster's
        # availability to the ratio that best matches its observed level. This
        # is the residual level the affine correction could not reach with wind
        # alone, which is exactly what a multiplicative loss represents.
        _, cor_cf = super().apply(
            reanalysis, clus_info, power_curves, factors, time_res, seasons=seasons
        )
        avail = self._fit_availability(cor_cf, gen_cf, clus_info, obs_level)
        factors = factors.merge(avail, on="cluster", how="left")
        factors["avail"] = factors["avail"].fillna(1.0)
        return factors, clus_info

    @staticmethod
    def _fit_availability(
        cor_cf: pd.DataFrame,
        gen_cf: pd.DataFrame,
        clus_info: pd.DataFrame,
        obs_level: str,
    ) -> pd.DataFrame:
        """Per-cluster ``obs_level / affine_corrected_level``, clipped to (0, 1]."""
        cluster_of = (
            clus_info.assign(ID=clus_info["ID"].astype(str))
            .set_index("ID")["cluster"]
        )
        cap_of = (
            clus_info.assign(ID=clus_info["ID"].astype(str))
            .set_index("ID")["capacity"]
        )

        # Corrected sim: wide (time x ID) -> capacity-weighted mean CF per cluster.
        sim = cor_cf.melt(id_vars="time", var_name="ID", value_name="cf")
        sim["ID"] = sim["ID"].astype(str)
        sim["cluster"] = sim["ID"].map(cluster_of)
        sim["cap"] = sim["ID"].map(cap_of)
        sim = sim.dropna(subset=["cf", "cluster", "cap"])
        sim_level = sim.groupby("cluster").apply(
            lambda g: (g["cf"] * g["cap"]).sum() / g["cap"].sum(),
            include_groups=False,
        )

        # Observed level per cluster, on the same capacity-weighted basis.
        obs = gen_cf.copy()
        obs["ID"] = obs["ID"].astype(str)
        obs["cluster"] = obs["ID"].map(cluster_of)
        obs["cap"] = obs["ID"].map(cap_of)
        obs = obs.dropna(subset=["obs", "cluster", "cap"])
        obs_level_by = obs.groupby("cluster").apply(
            lambda g: (g["obs"] * g["cap"]).sum() / g["cap"].sum(),
            include_groups=False,
        )

        avail = (obs_level_by / sim_level).clip(upper=1.0)
        # A cluster with no usable corrected sim (all NaN) keeps availability 1,
        # i.e. no loss applied, rather than dropping out.
        avail = avail.replace([np.inf, -np.inf], np.nan)
        return avail.rename("avail").reset_index()

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
        cor_ws, cor_cf = super().apply(
            reanalysis, clus_info, power_curves, factors, time_res, seasons=seasons
        )
        if "avail" not in factors.columns:
            return cor_ws, cor_cf

        # Scale each grid point's corrected CF by its cluster's availability.
        avail_of = (
            factors.drop_duplicates("cluster").set_index("cluster")["avail"]
        )
        cluster_of = (
            clus_info.assign(ID=clus_info["ID"].astype(str))
            .set_index("ID")["cluster"]
        )
        scaled = cor_cf.copy()
        for col in scaled.columns:
            if col == "time":
                continue
            cluster = cluster_of.get(str(col))
            factor = avail_of.get(cluster, 1.0) if cluster is not None else 1.0
            if pd.notna(factor):
                scaled[col] = scaled[col] * factor
        return cor_ws, scaled
