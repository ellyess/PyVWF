"""What a harness run records about its own inputs, for the manifest.

Split from ``pyvwf.harness.driver``. Each helper reads one property of a run's
inputs (the observation gates, the curve each unit resolved to, the ERA5
extent, the roughness treatment, the accepted years behind each factor) and
returns the manifest's block for it, writing a file where the record is too
large for the manifest.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

import pyvwf.wind as wind
from pyvwf.data import min_accepted_years
from pyvwf.harness.regions import RegionSpec
from pyvwf.provenance import (
    CURVE_RESOLUTION_NAME,
    curve_resolution,
    summarise_curve_resolution,
)


class CurveSubstitutionError(ValueError):
    """A country-level fleet names a power curve the loaded library lacks.

    Every unit of a country grid carries one representative turbine, so a
    missing curve is never a minor share of the fleet: it moves the whole row
    onto the curve table's first column. That is how all eight country rows ran
    on a 100 kW distributed-wind curve until 2026-09-25.
    """


def _record_observation_quality(source) -> dict:
    """What the country-level observation gates found, for the manifest.

    A row sitting on the fetcher's clip ceiling is a value that was discarded,
    not a value that is wrong, so a metric computed over the series is computed
    over fewer observations than it appears to be. That count stopped at the
    audit script; it now travels with the run, the way the substituted and
    extrapolated shares do.

    Returns an empty dict for a source that runs no such gate, which is every
    turbine-level source.
    """
    report = getattr(source, "obs_report", None)
    if report is None:
        return {}
    return {
        "n_rows": report.n_rows,
        "n_clipped": report.n_clipped,
        "clipped_share": report.frac_clipped,
        "peak_cf": report.peak_cf,
        "mean_cf": report.mean_cf,
        "longest_unchanged_capacity_years": report.longest_unchanged_years,
        "unchanged_capacity_span": report.unchanged_span,
        "ok": report.ok,
        "failures": list(report.failures),
        "warnings": list(report.warnings_),
        "notes": list(report.notes),
    }


def _record_curve_resolution(
    run_dir: Path,
    fleet: pd.DataFrame,
    power_curves: pd.DataFrame,
    code: str,
    *,
    refuse_substitution: bool = False,
) -> dict:
    """Write ``curve_resolution.csv`` for the fleet a run simulates.

    A model missing from the curve table used to be visible only as a one-off
    warning, which is how every country-level row came to be simulated on a
    100 kW fallback curve for two months unnoticed. The record goes in the run
    directory, its summary in the manifest, and the substituted share into
    every metrics row, so it travels with the numbers the way ``fit_quality``
    does.

    With ``refuse_substitution``, which country-level runs pass, any
    substitution raises :class:`CurveSubstitutionError` once the record is
    written, so the run directory still says which keys were missing.
    """
    resolution = curve_resolution(fleet, power_curves)
    resolution.to_csv(run_dir / CURVE_RESOLUTION_NAME, index=False)
    summary = summarise_curve_resolution(resolution)
    if summary["n_models_substituted"] and refuse_substitution:
        raise CurveSubstitutionError(
            f"{code}: the loaded curve library has no curve for "
            f"{sorted(summary['substitutions'])}, so "
            f"{summary['substituted_capacity_share']:.1%} of fleet capacity would be "
            f"simulated on {sorted(set(summary['substitutions'].values()))}. Country "
            "grids name licensed Vestas curves: run with PYVWF_INPUT=input/combined. "
            f"Record: {run_dir / CURVE_RESOLUTION_NAME}."
        )
    if summary["n_models_substituted"]:
        warnings.warn(
            f"{code}: {summary['substituted_capacity_share']:.1%} of fleet capacity "
            f"simulated on a substitute curve {summary['substitutions']}; see "
            f"{run_dir / CURVE_RESOLUTION_NAME}."
        )
    return summary


def _record_era5_extent(reanalysis, fleet: pd.DataFrame, spec: RegionSpec) -> dict:
    """Where the fleet lies against the loaded ERA5 extent, for the run record.

    A run only gets here with units outside the extent if its region opted in
    (``[era5] allow_extrapolation = true``); otherwise ``interpolate_wind``
    has already refused. The record is written either way, with zeros when
    nothing is outside, so an absent value never needs interpreting. Inside the
    loaded extent is a statement about position, not a check of the data in
    those cells; ``wind.off_curve_record`` counts off-curve and missing values.
    """
    coverage = wind.loaded_extent_coverage(reanalysis, fleet)
    record = {
        **coverage,
        "requested_bbox": list(spec.bbox),
        "allow_extrapolation": bool(spec.allow_extrapolation),
        "meaning": (
            "units inside the loaded ERA5 extent are interpolated, not extrapolated; "
            "this does not verify the data in those cells"
        ),
    }
    if coverage["units_outside_loaded_extent"]:
        warnings.warn(
            f"{spec.code}: {coverage['capacity_share_outside_loaded_extent']:.1%} of fleet "
            "capacity lies outside the loaded ERA5 extent and was simulated from "
            "extrapolated winds ([era5] allow_extrapolation = true). Any scorecard row "
            "from this run carries the extrapolation marker and the share."
        )
    return record


def _record_roughness(reanalysis, spec: RegionSpec) -> dict:
    """Which temporal treatment of the roughness the run actually applied.

    Requested and applied are both recorded, because they differ when a region
    asks for a stored field and the ERA5 files carry none. The European files
    carry one annual mean per year; every other region derives it per timestep
    (``docs/design/roughness-temporal-treatment.md``).
    """
    return {
        "requested": spec.roughness,
        "applied": reanalysis.attrs.get("pyvwf_roughness_treatment"),
    }


def _record_accepted_years(factors: pd.DataFrame, time_res: str, n_training_years: int) -> dict:
    """How many accepted years each factor rests on, for the manifest (#28).

    The same counts are the factors table's ``n_years`` column; the manifest
    carries them so a run's record says, without the table, which factors rest
    on fewer than all their training years, which were refused, and which were
    never fitted and carry the identity.
    """
    per_slice: dict[str, dict[str, int]] = {}
    for slice_label, cluster, n in zip(factors[time_res], factors["cluster"], factors["n_years"]):
        per_slice.setdefault(str(slice_label), {})[str(cluster)] = int(n)
    refused = factors["scalar"].isna() & factors["offset"].isna()
    unfitted = (factors["n_years"] == 0) & ~refused
    partial = (factors["n_years"] < n_training_years) & ~refused & ~unfitted
    return {
        "training_years": n_training_years,
        "min_accepted_years": min_accepted_years(n_training_years),
        "n_partial": int(partial.sum()),
        "n_refused": int(refused.sum()),
        "n_unfitted": int(unfitted.sum()),
        "per_factor": per_slice,
    }
