"""Plausibility gates for country-level observed capacity-factor series.

A national wind fleet has a physical envelope. Its capacity factor cannot
exceed 1, and over a multi-year sub-daily record it must at some point come
close to the fleet's aggregate peak, which is well above 0.6 for every real
system. A series that never rises that far, or that sits against a clipping
ceiling, is reporting something other than generation over installed capacity.

Nothing downstream detects this. The affine correction absorbs any constant
factor into the scalar and still reports a clean in-sample fit, so a series
that is uniformly four times too small produces plausible-looking factors and
a silently wrong model. These checks are the only place the error surfaces.

The thresholds are deliberately loose. They are a floor on physical
possibility, not a judgement about whether a country's wind resource is well
modelled.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Capacity factor is a fraction of nameplate; above 1 the series is not a CF.
MAX_CF = 1.0

#: Ceiling applied by the ENTSO-E fetcher. Rows sitting on it are saturated,
#: not merely high, so the true value is unknown.
CLIP_CEILING = 1.5

#: A national fleet's sub-daily peak. Every real system exceeds this; a record
#: that does not is understating generation or overstating capacity.
MIN_PEAK_CF = 0.65

#: Long-run national mean CF. Onshore fleets sit near 0.2 to 0.3, offshore
#: heavy ones higher. Outside this band something is wrong with the ratio.
MIN_MEAN_CF = 0.08
MAX_MEAN_CF = 0.50

#: The peak check only means something on a sub-daily series; monthly means
#: legitimately never approach the fleet peak.
MAX_STEP_HOURS_FOR_PEAK_CHECK = 6.0

#: Below this many years a constant capacity register is unremarkable.
FROZEN_CAPACITY_MIN_YEARS = 2.0

#: ENTSO-E publishes installed capacity annually, so a healthy multi-year
#: register has roughly one distinct value per year. Fewer than one per this
#: many years means the register is not tracking the fleet.
CAPACITY_YEARS_PER_UPDATE = 2.0

#: Ratio between the best and worst annual mean CF. Interannual wind
#: variability is roughly plus or minus 15%, and even a fleet upgrading to
#: taller machines and more offshore moves the national figure by well under
#: this. A larger spread means the generation series covers a changing share
#: of the fleet that the capacity denominator counts, which no constant
#: rescaling can repair.
MAX_ANNUAL_CF_RATIO = 1.6

#: Full years needed before the drift test means anything.
DRIFT_MIN_YEARS = 3

#: Total capacity movement over the record below which a coarse register is a
#: register that stopped tracking rather than a genuinely flat fleet.
MIN_CAPACITY_GROWTH = 0.05

#: Fraction of missing capacity factors that stops being incidental.
MAX_MISSING_FRACTION = 0.05


@dataclass
class CountryObsReport:
    """Summary statistics and gate failures for one observed CF series."""

    label: str
    n_rows: int
    step_hours: float | None
    span_years: float | None
    mean_cf: float
    peak_cf: float
    frac_clipped: float
    frac_missing: float
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no gate failed."""
        return not self.issues

    def as_row(self) -> dict:
        """Flat mapping for tabulating many regions together."""
        return {
            "label": self.label,
            "n_rows": self.n_rows,
            "step_hours": self.step_hours,
            "span_years": self.span_years,
            "mean_cf": self.mean_cf,
            "peak_cf": self.peak_cf,
            "frac_clipped": self.frac_clipped,
            "frac_missing": self.frac_missing,
            "ok": self.ok,
            "issues": "; ".join(self.issues),
        }


def _as_datetime_index(index: pd.Index) -> pd.DatetimeIndex | None:
    """Coerce an index to UTC datetimes, or None if it is not time-like.

    ``pd.read_csv(..., parse_dates=True)`` returns a plain object Index when the
    timestamps carry mixed UTC offsets, which every DST-crossing ENTSO-E export
    does. Without this coercion the resolution-dependent checks silently skip
    exactly the files most worth checking.
    """
    if isinstance(index, pd.DatetimeIndex):
        return index.tz_convert("UTC") if index.tz is not None else index
    try:
        return pd.DatetimeIndex(pd.to_datetime(index, utc=True, format="mixed"))
    except (TypeError, ValueError):
        return None


def _median_step_hours(index: pd.DatetimeIndex | None) -> float | None:
    if index is None or len(index) < 3:
        return None
    deltas = index.to_series().sort_index().diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]
    if deltas.empty:
        return None
    return float(deltas.median().total_seconds() / 3600.0)


def _span_years(index: pd.DatetimeIndex | None) -> float | None:
    if index is None or len(index) < 2:
        return None
    return float((index.max() - index.min()).total_seconds() / (365.25 * 24 * 3600))


def check_country_cf(
    obs: pd.DataFrame,
    label: str = "country observations",
    *,
    strict: bool = False,
    warn: bool = True,
) -> CountryObsReport:
    """Check one country-level observed CF series against physical bounds.

    Args:
        obs: Observations with a ``capacity_factor`` column, ideally
            DatetimeIndexed. An optional ``capacity_mw`` column enables the
            frozen-register check.
        label: Name used in messages, e.g. ``"NL train 2015-2021"``.
        strict: Raise :class:`ValueError` instead of warning when a gate fails.
        warn: Emit a :class:`UserWarning` per failed gate. Ignored when
            ``strict`` is set.

    Returns:
        A :class:`CountryObsReport`; ``report.ok`` is False if any gate failed.

    Raises:
        ValueError: If ``capacity_factor`` is missing, or if ``strict`` and any
            gate failed.
    """
    if "capacity_factor" not in obs.columns:
        raise ValueError(f"{label}: no 'capacity_factor' column")

    cf = pd.to_numeric(obs["capacity_factor"], errors="coerce")
    n = len(cf)
    valid = cf.dropna()
    frac_missing = float(1.0 - len(valid) / n) if n else 1.0

    index = _as_datetime_index(obs.index)
    step_hours = _median_step_hours(index)
    span_years = _span_years(index)

    mean_cf = float(valid.mean()) if len(valid) else float("nan")
    peak_cf = float(valid.max()) if len(valid) else float("nan")
    frac_clipped = (
        float((valid >= CLIP_CEILING - 1e-9).mean()) if len(valid) else float("nan")
    )

    issues: list[str] = []

    if not len(valid):
        issues.append("every capacity factor is missing")
    else:
        if frac_missing > MAX_MISSING_FRACTION:
            issues.append(f"{frac_missing:.1%} of capacity factors are missing")

        if peak_cf > MAX_CF:
            issues.append(
                f"peak CF {peak_cf:.3f} exceeds 1; generation and capacity are "
                "not on a consistent basis"
            )

        n_clipped = int((valid >= CLIP_CEILING - 1e-9).sum())
        if n_clipped:
            issues.append(
                f"{n_clipped} row{'' if n_clipped == 1 else 's'} "
                f"({frac_clipped:.2%}) {'sits' if n_clipped == 1 else 'sit'} on "
                f"the {CLIP_CEILING} clip ceiling, so the true value is unknown"
            )

        if (
            step_hours is not None
            and step_hours <= MAX_STEP_HOURS_FOR_PEAK_CHECK
            and peak_cf < MIN_PEAK_CF
        ):
            issues.append(
                f"peak CF {peak_cf:.3f} never reaches {MIN_PEAK_CF} over a "
                f"{step_hours:g} h series; generation is understated relative "
                "to capacity"
            )

        if np.isfinite(mean_cf) and not (MIN_MEAN_CF <= mean_cf <= MAX_MEAN_CF):
            issues.append(
                f"mean CF {mean_cf:.3f} is outside the plausible national band "
                f"[{MIN_MEAN_CF}, {MAX_MEAN_CF}]"
            )

    if index is not None and len(valid) and span_years and span_years >= DRIFT_MIN_YEARS:
        annual = pd.Series(valid.to_numpy(), index=index[cf.notna().to_numpy()])
        annual = annual.resample("YE").mean().dropna()
        if len(annual) >= DRIFT_MIN_YEARS:
            lo, hi = float(annual.min()), float(annual.max())
            if lo > 0 and hi / lo > MAX_ANNUAL_CF_RATIO:
                issues.append(
                    f"annual mean CF ranges {lo:.3f} to {hi:.3f} ({hi / lo:.1f}x) "
                    "across the record, which is more than weather; check "
                    "whether the fleet genuinely improved that much or the "
                    "generation series covers a changing share of the fleet "
                    "the capacity counts"
                )

    if "capacity_mw" in obs.columns:
        cap = pd.to_numeric(obs["capacity_mw"], errors="coerce").dropna()
        if len(cap) and span_years is not None and span_years >= FROZEN_CAPACITY_MIN_YEARS:
            distinct = int(cap.nunique())
            if distinct == 1:
                issues.append(
                    f"installed capacity is constant at {cap.iloc[0]:.0f} MW over "
                    f"{span_years:.1f} years; the register did not update"
                )
            elif distinct * CAPACITY_YEARS_PER_UPDATE < span_years:
                # Few distinct values is only damning when the total movement is
                # also negligible. A genuinely flat fleet (PT grew 15% over
                # seven years) can legitimately be described by two numbers; a
                # register that stopped tracking (IE moved 0.6%) cannot.
                growth = float(cap.max() / cap.min() - 1.0) if cap.min() > 0 else 0.0
                if growth < MIN_CAPACITY_GROWTH:
                    issues.append(
                        f"installed capacity takes only {distinct} distinct "
                        f"values over {span_years:.1f} years and moves "
                        f"{growth:.1%} ({cap.min():.0f} to {cap.max():.0f} MW); "
                        "the register is not tracking the fleet"
                    )

    report = CountryObsReport(
        label=label,
        n_rows=n,
        step_hours=step_hours,
        span_years=span_years,
        mean_cf=mean_cf,
        peak_cf=peak_cf,
        frac_clipped=frac_clipped,
        frac_missing=frac_missing,
        issues=issues,
    )

    if issues:
        message = f"{label}: " + "; ".join(issues)
        if strict:
            raise ValueError(message)
        if warn:
            warnings.warn(message, UserWarning, stacklevel=2)

    return report
