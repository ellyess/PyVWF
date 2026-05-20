"""Distribution-aware bias correction for PyVWF.

The default PyVWF correction scheme is a per-cluster, per-time-slice linear map

    w_corrected = alpha * w_uncorrected + beta,

where ``alpha`` and ``beta`` are tuned so that the *mean* simulated capacity
factor matches the *mean* observed capacity factor over the training window.
This corrects the first moment of the distribution but leaves higher moments
(variance, skew, tail behaviour) untouched. Reanalysis-derived wind is known to
mis-represent variability [Staffell & Pfenninger 2016], so two time series can
share a mean while disagreeing badly on ramps, calm spells, and high-wind tails
- exactly the features that matter for energy-system studies (storage sizing,
balancing, capacity-credit analysis).

This module adds *distributional* bias correction, ported from the climate
downscaling literature:

* **Empirical Quantile Mapping (EQM)** - learn a monotonic transfer function
  that maps each quantile of the modelled distribution onto the corresponding
  quantile of the observed distribution. This corrects the entire marginal
  distribution, not just its mean.

* **Quantile Delta Mapping (QDM)** [Cannon, Sobie & Murdock 2015, J. Climate
  28(17)] - a trend-preserving variant. Rather than overwriting the model
  distribution, it applies the reference-period model-vs-observation quantile
  *deltas* to the target-period model values, so any change the model itself
  produces between reference and target periods is retained. This matters when
  the correction trained on one period is applied to another (e.g. a different
  test year or a future climate scenario).

Both estimators operate on plain 1-D arrays, so they can correct wind speeds or
capacity factors interchangeably. Helpers at the bottom apply them per cluster
and per time-slice, mirroring the structure of :mod:`vwf.correction`.

Caveat - bounded/saturated variables: quantile mapping degenerates where the
model distribution saturates against a hard bound (e.g. capacity factor pinned
at 1.0 during sustained rated wind), because the inverse model CDF is undefined
on the resulting flat region. Prefer mapping wind speed, or applying QM at a
temporal resolution (e.g. monthly means) where saturation is rare.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

__all__ = [
    "QuantileMapper",
    "empirical_quantile_mapping",
    "quantile_delta_mapping",
    "fit_quantile_correction_table",
    "apply_quantile_correction",
    "fit_quantile_factor_frame",
    "apply_quantile_factor_frame",
]


def _clean(x) -> np.ndarray:
    """Return finite values as a 1-D float array."""
    arr = np.asarray(x, dtype=float).ravel()
    return arr[np.isfinite(arr)]


@dataclass
class QuantileMapper:
    """Empirical quantile-mapping transfer function.

    Fit a monotonic mapping ``T`` such that ``T(model)`` shares the marginal
    distribution of ``obs``. Calling :meth:`transform` applies ``T`` to new
    values.

    Args:
        n_quantiles: Number of evenly spaced probability knots used to estimate
            the model and observed quantile functions.
        kind: ``"additive"`` corrects by adding ``obs_q - model_q`` deltas;
            ``"multiplicative"`` corrects by the ratio ``obs_q / model_q``.
            Additive is the safe default for wind speed; multiplicative can be
            useful for strictly positive, ratio-scaled variables.
        extrapolate: How to handle values outside the fitted model range.
            ``"constant"`` holds the edge delta/ratio (recommended);
            ``"clip"`` clamps inputs to the fitted range before mapping.
    """

    n_quantiles: int = 100
    kind: str = "additive"
    extrapolate: str = "constant"

    # Learned state.
    _p: np.ndarray = field(default=None, repr=False)
    _model_q: np.ndarray = field(default=None, repr=False)
    _obs_q: np.ndarray = field(default=None, repr=False)
    fitted: bool = field(default=False, repr=False)

    def __post_init__(self):
        if self.kind not in ("additive", "multiplicative"):
            raise ValueError("kind must be 'additive' or 'multiplicative'")
        if self.extrapolate not in ("constant", "clip"):
            raise ValueError("extrapolate must be 'constant' or 'clip'")
        if self.n_quantiles < 2:
            raise ValueError("n_quantiles must be >= 2")

    @classmethod
    def from_quantiles(cls, p, model_q, obs_q, kind="additive", extrapolate="constant"):
        """Rebuild a fitted mapper from stored quantile knots (deserialisation).

        Args:
            p: Probability knots in ``[0, 1]``.
            model_q: Model quantile values at ``p``.
            obs_q: Observed quantile values at ``p``.
            kind / extrapolate: Same meaning as the constructor arguments.
        """
        p = np.asarray(p, dtype=float)
        obj = cls(n_quantiles=len(p), kind=kind, extrapolate=extrapolate)
        obj._p = p
        obj._model_q = np.maximum.accumulate(np.asarray(model_q, dtype=float))
        obj._obs_q = np.asarray(obs_q, dtype=float)
        obj.fitted = True
        return obj

    def quantiles(self):
        """Return the fitted ``(p, model_q, obs_q)`` knots (for serialisation)."""
        if not self.fitted:
            raise RuntimeError("QuantileMapper must be fitted before quantiles()")
        return self._p.copy(), self._model_q.copy(), self._obs_q.copy()

    def fit(self, model, obs) -> "QuantileMapper":
        """Estimate the transfer function from paired training samples.

        ``model`` and ``obs`` need not be the same length or time-aligned; only
        their marginal distributions are used.
        """
        m = _clean(model)
        o = _clean(obs)
        if m.size == 0 or o.size == 0:
            raise ValueError("fit requires non-empty finite model and obs arrays")
        self._p = np.linspace(0.0, 1.0, self.n_quantiles)
        self._model_q = np.quantile(m, self._p)
        self._obs_q = np.quantile(o, self._p)
        # Guarantee a strictly non-decreasing model quantile function so that
        # interpolation against it is well defined even with ties.
        self._model_q = np.maximum.accumulate(self._model_q)
        self.fitted = True
        return self

    def _cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative probability of ``x`` within the fitted model distribution."""
        return np.interp(x, self._model_q, self._p, left=0.0, right=1.0)

    def transform(self, x) -> np.ndarray:
        """Apply the fitted correction to ``x``."""
        if not self.fitted:
            raise RuntimeError("QuantileMapper must be fitted before transform")
        x = np.asarray(x, dtype=float)
        finite = np.isfinite(x)
        out = np.array(x, dtype=float, copy=True)

        xv = x[finite]
        if self.extrapolate == "clip":
            xv = np.clip(xv, self._model_q[0], self._model_q[-1])

        p = self._cdf(xv)
        obs_at_p = np.interp(p, self._p, self._obs_q)
        model_at_p = np.interp(p, self._p, self._model_q)

        if self.kind == "additive":
            corrected = xv + (obs_at_p - model_at_p)
        else:  # multiplicative
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(model_at_p > 0, obs_at_p / model_at_p, 1.0)
            corrected = xv * ratio

        out[finite] = corrected
        return out

    def fit_transform(self, model, obs, x=None) -> np.ndarray:
        self.fit(model, obs)
        return self.transform(model if x is None else x)


def empirical_quantile_mapping(model_train, obs_train, model_apply, **kwargs):
    """Convenience one-shot EQM.

    Fit a :class:`QuantileMapper` on ``(model_train, obs_train)`` and apply it
    to ``model_apply``.
    """
    return QuantileMapper(**kwargs).fit(model_train, obs_train).transform(model_apply)


def quantile_delta_mapping(
    model_ref,
    obs_ref,
    model_target,
    n_quantiles: int = 100,
    kind: str = "additive",
):
    """Trend-preserving Quantile Delta Mapping [Cannon et al. 2015].

    Args:
        model_ref: Modelled values over the reference (training) period.
        obs_ref: Observed values over the reference period.
        model_target: Modelled values over the target period to be corrected.
        n_quantiles: Number of probability knots.
        kind: ``"additive"`` or ``"multiplicative"`` delta.

    Returns:
        Corrected ``model_target`` values, preserving the model's own
        reference-to-target change at every quantile while removing the
        reference-period model-vs-observation bias.
    """
    if kind not in ("additive", "multiplicative"):
        raise ValueError("kind must be 'additive' or 'multiplicative'")

    mr = _clean(model_ref)
    o = _clean(obs_ref)
    if mr.size == 0 or o.size == 0:
        raise ValueError("QDM requires non-empty finite reference arrays")

    p = np.linspace(0.0, 1.0, n_quantiles)
    model_ref_q = np.maximum.accumulate(np.quantile(mr, p))
    obs_ref_q = np.quantile(o, p)

    target = np.asarray(model_target, dtype=float)
    finite = np.isfinite(target)
    out = np.array(target, dtype=float, copy=True)
    tv = target[finite]

    # Quantile (rank) of each target value within the *target* model sample.
    tau = _empirical_cdf_rank(tv)

    obs_at_tau = np.interp(tau, p, obs_ref_q)
    model_ref_at_tau = np.interp(tau, p, model_ref_q)
    # Model value at this rank within the target sample (== the value itself).
    if kind == "additive":
        corrected = tv + (obs_at_tau - model_ref_at_tau)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(model_ref_at_tau > 0, obs_at_tau / model_ref_at_tau, 1.0)
        corrected = tv * ratio

    out[finite] = corrected
    return out


def _empirical_cdf_rank(x: np.ndarray) -> np.ndarray:
    """Plotting-position cumulative probability for each element of ``x``."""
    n = x.size
    order = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    # (i + 0.5) / n plotting positions keep ranks strictly inside (0, 1).
    return (order + 0.5) / n


# ---------------------------------------------------------------------------
# Cluster / time-slice aware helpers (mirror vwf.correction structure)
# ---------------------------------------------------------------------------

def fit_quantile_correction_table(
    train_df: pd.DataFrame,
    value_col: str = "sim",
    obs_col: str = "obs",
    group_cols=("cluster", "time_slice"),
    n_quantiles: int = 100,
    kind: str = "additive",
):
    """Fit one :class:`QuantileMapper` per (cluster, time-slice) group.

    Args:
        train_df: Long-format training data with one row per timestamp/sample.
        value_col: Column of modelled values to be corrected.
        obs_col: Column of observed values.
        group_cols: Grouping columns (default cluster + time slice).
        n_quantiles: Knots per mapper.
        kind: ``"additive"`` or ``"multiplicative"``.

    Returns:
        dict mapping each group key (tuple) to a fitted :class:`QuantileMapper`.
    """
    group_cols = list(group_cols)
    mappers: dict = {}
    for key, grp in train_df.groupby(group_cols):
        model = grp[value_col].to_numpy()
        obs = grp[obs_col].to_numpy()
        if _clean(model).size == 0 or _clean(obs).size == 0:
            continue
        mappers[key if len(group_cols) > 1 else (key,)] = QuantileMapper(
            n_quantiles=n_quantiles, kind=kind
        ).fit(model, obs)
    return mappers


def apply_quantile_correction(
    test_df: pd.DataFrame,
    mappers: dict,
    value_col: str = "sim",
    out_col: str = "cor",
    group_cols=("cluster", "time_slice"),
):
    """Apply fitted per-group mappers to test data.

    Rows whose group has no fitted mapper are passed through unchanged.

    Returns:
        Copy of ``test_df`` with a new ``out_col`` column.
    """
    group_cols = list(group_cols)
    df = test_df.copy()
    df[out_col] = df[value_col].astype(float)

    for key, grp in df.groupby(group_cols):
        lookup = key if len(group_cols) > 1 else (key,)
        mapper = mappers.get(lookup)
        if mapper is None:
            continue
        df.loc[grp.index, out_col] = mapper.transform(grp[value_col].to_numpy())
    return df


# ---------------------------------------------------------------------------
# Serialisable factor frame (the QM analogue of vwf.data.format_bc_factors)
# ---------------------------------------------------------------------------

def fit_quantile_factor_frame(
    gen_cf: pd.DataFrame,
    time_res: str,
    value_col: str = "sim",
    obs_col: str = "obs",
    cluster_col: str = "cluster",
    n_quantiles: int = 100,
    kind: str = "additive",
) -> pd.DataFrame:
    """Fit per-(cluster, time-slice) quantile mappers and flatten to a table.

    This is the quantile-mapping counterpart of
    :func:`vwf.data.format_bc_factors`: it consumes the monthly ``gen_cf``
    training frame (one row per timestamp/grid-point with ``sim`` and ``obs``
    capacity factors, a cluster assignment, and a ``time_res`` slice column) and
    produces a long, CSV-serialisable table of quantile knots.

    Returns:
        DataFrame with columns ``[cluster_col, time_res, "p", "model_q",
        "obs_q", "kind", "n_quantiles"]`` - one row per probability knot per
        group. Round-trips through :func:`apply_quantile_factor_frame`.
    """
    if cluster_col not in gen_cf.columns:
        raise KeyError(f"gen_cf must contain a '{cluster_col}' column")
    if time_res not in gen_cf.columns:
        raise KeyError(f"gen_cf must contain the time-resolution column '{time_res}'")

    mappers = fit_quantile_correction_table(
        gen_cf,
        value_col=value_col,
        obs_col=obs_col,
        group_cols=(cluster_col, time_res),
        n_quantiles=n_quantiles,
        kind=kind,
    )

    rows = []
    for (cluster, slice_label), mapper in mappers.items():
        p, model_q, obs_q = mapper.quantiles()
        for pi, mi, oi in zip(p, model_q, obs_q):
            rows.append({
                cluster_col: cluster,
                time_res: slice_label,
                "p": pi,
                "model_q": mi,
                "obs_q": oi,
                "kind": mapper.kind,
                "n_quantiles": n_quantiles,
            })
    return pd.DataFrame(
        rows,
        columns=[cluster_col, time_res, "p", "model_q", "obs_q", "kind", "n_quantiles"],
    )


def apply_quantile_factor_frame(
    sim_df: pd.DataFrame,
    factor_frame: pd.DataFrame,
    time_res: str,
    value_col: str = "sim",
    out_col: str = "cor",
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    """Apply a serialised quantile-factor table to new simulated values.

    Args:
        sim_df: Frame to be corrected; must carry ``cluster_col``, the
            ``time_res`` slice column, and ``value_col``.
        factor_frame: Output of :func:`fit_quantile_factor_frame` (e.g. read
            back from CSV).
        time_res: Name of the time-slice column shared by both frames.
        value_col / out_col / cluster_col: Column names.

    Returns:
        Copy of ``sim_df`` with an ``out_col`` of corrected values. Groups with
        no fitted mapper pass through unchanged.
    """
    mappers: dict = {}
    for (cluster, slice_label), grp in factor_frame.groupby([cluster_col, time_res]):
        grp = grp.sort_values("p")
        kind = grp["kind"].iloc[0] if "kind" in grp.columns else "additive"
        mappers[(cluster, slice_label)] = QuantileMapper.from_quantiles(
            grp["p"].to_numpy(),
            grp["model_q"].to_numpy(),
            grp["obs_q"].to_numpy(),
            kind=kind,
        )

    return apply_quantile_correction(
        sim_df,
        mappers,
        value_col=value_col,
        out_col=out_col,
        group_cols=(cluster_col, time_res),
    )
