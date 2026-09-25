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

#: Severity tiers for capacity factors above 1, because one hour in seven years
#: and 13.1% of hours are not the same defect and must not read the same. A
#: warning that fires identically on both teaches a reader to ignore it.
#: Ireland's preserved pre-repair series sets the bar for severe: 13.1% of
#: hours above 1 and two calendar months whose MEAN exceeds 1. Belgium's single
#: hour sits nowhere near it.
WARN_SHARE_ABOVE_MAX_CF = 0.001
FAIL_SHARE_ABOVE_MAX_CF = 0.01

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

#: Consecutive calendar years over which an unchanged register is a register
#: that stopped tracking. A national wind fleet growing even slowly moves its
#: register inside three years. **This replaces a test on total movement over
#: the record**, which asked the wrong question: Portugal moves 15.5% and
#: Sweden 19.9%, so both passed it, and both hold one number for five straight
#: years while their fleets grew. When the register moves is the test; how far
#: it moves in the end is not.
#:
#: **The threshold was set from Portugal and Sweden, and found Sweden's zonal
#: series by itself.** SE-BZ was not among the cases it was tuned on: its four
#: bidding zones each hold a frozen capacity over five years, each peaks at
#: exactly 0.900, and the four sum to Sweden's national register exactly, which
#: is the fetcher's ``gen.max() / 0.9`` fallback showing through. A rule that
#: fires on a case it was not built for is the point of setting it on a
#: statistic rather than on the cases.
#:
#: **It reports the wrong number on a stacked zonal frame.** Given one frame
#: holding several zones it takes the median capacity across them, so SE-BZ is
#: reported as 2158 MW unchanged for three years when the truth is four zones
#: frozen for five. The finding is right and the figure is not; checking per
#: zone is an open item.
MAX_UNCHANGED_YEARS = 3

#: Ratio between the best and worst annual mean CF. Interannual wind
#: variability is roughly plus or minus 15%, and even a fleet upgrading to
#: taller machines and more offshore moves the national figure by well under
#: this. A larger spread means the generation series covers a changing share
#: of the fleet that the capacity denominator counts, which no constant
#: rescaling can repair.
MAX_ANNUAL_CF_RATIO = 1.6

#: Full years needed before the drift test means anything.
DRIFT_MIN_YEARS = 3


#: Fraction of missing capacity factors that stops being incidental.
MAX_MISSING_FRACTION = 0.05


def longest_unchanged_run(
    capacity: pd.Series, index: pd.DatetimeIndex
) -> tuple[int, tuple[int, int] | None]:
    """The longest run of consecutive calendar years holding one capacity.

    Args:
        capacity: the register, one value per observation row.
        index: the matching timestamps.

    Returns:
        The run length in years, and its first and last year, or ``None`` when
        there is nothing to report.

    A register is judged on whether it moved while the fleet did, not on how
    far it moved in the end. Portugal holds 4486 MW from 2015 to 2019 and then
    steps to 5181, so its total movement is 15.5% and its register tracked
    nothing for five years.
    """
    if not len(capacity) or index is None or not len(index):
        return 0, None
    yearly = pd.Series(capacity.to_numpy(), index=index).groupby(index.year).median()
    yearly = yearly.dropna()
    if not len(yearly):
        return 0, None
    years = [int(y) for y in yearly.index]
    values = yearly.to_numpy(dtype=float)
    best, best_start = 1, years[0]
    run, start = 1, years[0]
    for i in range(1, len(values)):
        if years[i] == years[i - 1] + 1 and abs(values[i] - values[i - 1]) < 1e-6:
            run += 1
        else:
            run, start = 1, years[i]
        if run > best:
            best, best_start = run, start
    return best, (best_start, best_start + best - 1)


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
    n_clipped: int = 0
    longest_unchanged_years: int = 0
    unchanged_span: str = ""
    failures: list[str] = field(default_factory=list)
    warnings_: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def issues(self) -> list[str]:
        """Every finding, worst first. Kept so existing readers still work."""
        return [*self.failures, *self.warnings_, *self.notes]

    @property
    def ok(self) -> bool:
        """True when no gate FAILED.

        Notes and warnings do not clear it to False. A series with one hour
        above 1 in seven years and a series with 13.1% of hours above 1 both
        used to come back not ok, with the same wording, which is how a reader
        learns to ignore the message.
        """
        return not self.failures

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
            "n_clipped": self.n_clipped,
            "frac_missing": self.frac_missing,
            "longest_unchanged_years": self.longest_unchanged_years,
            "unchanged_span": self.unchanged_span,
            "ok": self.ok,
            "n_failures": len(self.failures),
            "n_warnings": len(self.warnings_),
            "n_notes": len(self.notes),
            "issues": "; ".join(self.issues),
        }


def _months_above_one(cf: pd.Series, index: pd.DatetimeIndex | None) -> list[str]:
    """Calendar months whose MEAN capacity factor exceeds 1.

    A single hour above 1 is what an annual register does to a growing fleet. A
    whole month averaging above 1 is impossible under any correct denominator,
    which makes it the unambiguous signal and the only one in the failure tier
    on its own.
    """
    if index is None or not len(cf):
        return []
    stamps = pd.DatetimeIndex(index[: len(cf)])
    # strftime rather than to_period: a tz-aware index warns that the
    # conversion drops the timezone, and the month label is all this needs.
    monthly = pd.Series(cf.to_numpy()).groupby(stamps.strftime("%Y-%m")).mean()
    return [str(month) for month, value in monthly.items() if value > MAX_CF]


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
    frac_clipped = float((valid >= CLIP_CEILING - 1e-9).mean()) if len(valid) else float("nan")

    failures: list[str] = []
    warnings_: list[str] = []
    notes: list[str] = []

    if not len(valid):
        failures.append("every capacity factor is missing")
    else:
        if frac_missing > MAX_MISSING_FRACTION:
            failures.append(f"{frac_missing:.1%} of capacity factors are missing")

        # Capacity factors above 1, tiered. The count and share travel in every
        # tier's message, because the number is what separates one bad hour
        # from a broken denominator.
        above = valid > MAX_CF
        n_above, share_above = int(above.sum()), float(above.mean())
        monthly_over = _months_above_one(valid, index) if n_above else []
        if n_above:
            where = (
                f"{n_above} of {len(valid)} rows ({share_above:.3%}) exceed 1, peak {peak_cf:.3f}"
            )
            if monthly_over or share_above > FAIL_SHARE_ABOVE_MAX_CF:
                months = (
                    f"; {len(monthly_over)} calendar month"
                    f"{'' if len(monthly_over) == 1 else 's'} have a MEAN "
                    f"above 1 ({', '.join(monthly_over[:4])}"
                    f"{', ...' if len(monthly_over) > 4 else ''})"
                    if monthly_over
                    else ""
                )
                failures.append(
                    f"{where}{months}; generation and capacity are not on a consistent basis"
                )
            elif share_above > WARN_SHARE_ABOVE_MAX_CF:
                warnings_.append(
                    f"{where}; the denominator does not track the fleet within the year"
                )
            else:
                notes.append(
                    f"{where}; an annual register cannot track within-year "
                    "additions, so an isolated hour above 1 is expected"
                )

        n_clipped = int((valid >= CLIP_CEILING - 1e-9).sum())
        if n_clipped:
            message = (
                f"{n_clipped} row{'' if n_clipped == 1 else 's'} "
                f"({frac_clipped:.2%}) {'sits' if n_clipped == 1 else 'sit'} on "
                f"the {CLIP_CEILING} clip ceiling, so the true value is discarded "
                "rather than wrong"
            )
            (failures if frac_clipped > WARN_SHARE_ABOVE_MAX_CF else notes).append(message)

        if (
            step_hours is not None
            and step_hours <= MAX_STEP_HOURS_FOR_PEAK_CHECK
            and peak_cf < MIN_PEAK_CF
        ):
            failures.append(
                f"peak CF {peak_cf:.3f} never reaches {MIN_PEAK_CF} over a "
                f"{step_hours:g} h series; generation is understated relative "
                "to capacity"
            )

        if np.isfinite(mean_cf) and not (MIN_MEAN_CF <= mean_cf <= MAX_MEAN_CF):
            failures.append(
                f"mean CF {mean_cf:.3f} is outside the plausible national band "
                f"[{MIN_MEAN_CF}, {MAX_MEAN_CF}]"
            )

    if index is not None and len(valid) and span_years and span_years >= DRIFT_MIN_YEARS:
        annual = pd.Series(valid.to_numpy(), index=index[cf.notna().to_numpy()])
        annual = annual.resample("YE").mean().dropna()
        if len(annual) >= DRIFT_MIN_YEARS:
            lo, hi = float(annual.min()), float(annual.max())
            if lo > 0 and hi / lo > MAX_ANNUAL_CF_RATIO:
                failures.append(
                    f"annual mean CF ranges {lo:.3f} to {hi:.3f} ({hi / lo:.1f}x) "
                    "across the record, which is more than weather; check "
                    "whether the fleet genuinely improved that much or the "
                    "generation series covers a changing share of the fleet "
                    "the capacity counts"
                )

    unchanged_years, unchanged_span = 0, None
    if "capacity_mw" in obs.columns:
        cap = pd.to_numeric(obs["capacity_mw"], errors="coerce").dropna()
        if len(cap) and span_years is not None and span_years >= FROZEN_CAPACITY_MIN_YEARS:
            if int(cap.nunique()) == 1:
                failures.append(
                    f"installed capacity is constant at {cap.iloc[0]:.0f} MW over "
                    f"{span_years:.1f} years; the register did not update"
                )
            elif index is not None:
                # When the register moved, not how far it moved in the end. The
                # test this replaces asked for total movement over the record
                # and so passed Portugal at 15.5% and Sweden at 19.9%, both of
                # which hold one number for five straight years while their
                # fleets grew.
                kept = index[pd.to_numeric(obs["capacity_mw"], errors="coerce").notna().to_numpy()]
                unchanged_years, span = longest_unchanged_run(cap, kept)
                unchanged_span = span
                if unchanged_years >= MAX_UNCHANGED_YEARS and span is not None:
                    held = float(
                        pd.Series(cap.to_numpy(), index=kept)
                        .groupby(kept.year)
                        .median()
                        .loc[span[0]]
                    )
                    failures.append(
                        f"installed capacity is unchanged at {held:.0f} MW across "
                        f"{unchanged_years} consecutive years ({span[0]} to "
                        f"{span[1]}) of a {span_years:.1f} year record; the "
                        "register is not tracking the fleet"
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
        n_clipped=n_clipped if len(valid) else 0,
        longest_unchanged_years=unchanged_years,
        unchanged_span=""
        if unchanged_span is None
        else f"{unchanged_span[0]} to {unchanged_span[1]}",
        failures=failures,
        warnings_=warnings_,
        notes=notes,
    )

    if report.issues:
        message = f"{label}: " + "; ".join(report.issues)
        if strict and report.failures:
            raise ValueError(message)
        if warn:
            warnings.warn(message, UserWarning, stacklevel=2)

    return report
