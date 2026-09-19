"""Harness driver: train, evaluate, and transfer runs for one region config.

The thin CLI in ``scripts/analysis/validate_region.py`` wraps these functions; the
logic lives here so it is importable and tested.

Transfer semantics are normative (design §7): collapse the source region's
factors to ONE capacity-weighted (scalar, offset) per time-slice, apply
uniformly to the target, and match seasonal slices by season NAME under the
TARGET's season definitions. Nothing else: no spatial matching schemes on
this branch. The driver also enforces the approved pair set: transfer runs
must have AU-NEM on exactly one side.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

import vwf.wind as wind
from vwf.clustering import cluster_turbines
from vwf.config import PyVWFPaths
from vwf.data import (
    assign_country_clusters,
    min_accepted_years,
    train_set,
    val_obs_and_fleet,
    val_set,
)
from vwf.harness.corrections import fit_quality, get_correction
from vwf.provenance import (
    CURVE_RESOLUTION_NAME,
    curve_resolution,
    summarise_curve_resolution,
    write_manifest_safe,
)
from vwf.harness.regions import RegionSpec
from vwf.harness.skill import (
    collapse_pseudo_replicates,
    restrict_to_common_rows,
    skill_metrics,
    summarise_exclusions,
)
from vwf.sources import (
    EntsoeFileSource,
    EntsoeZonalFileSource,
    ObservationSource,
    get_source,
)

#: The transfer pair set that has been validated: AU-NEM against Europe, in
#: either direction. Other pairings are untested rather than unsupported.
TRANSFER_HUB = "AU-NEM"


def check_transfer_pair(source_code: str, target_code: str) -> None:
    """Reject transfer pairs outside the approved AU↔Europe set."""
    codes = {source_code.upper(), target_code.upper()}
    if source_code.upper() == target_code.upper():
        raise ValueError(f"transfer requires two different regions, got {codes}")
    if TRANSFER_HUB not in codes:
        raise ValueError(
            f"transfer pair ({source_code}, {target_code}) is outside the approved "
            f"set: {TRANSFER_HUB} must be on exactly one side (design §7). "
            "Other pairs are out of scope on this branch."
        )


def resolve_source(
    spec: RegionSpec, split: Literal["train", "test"] = "train"
) -> ObservationSource:
    """Resolve the region's observation source from the registry by name.

    Country-level file-backed regions ("entsoe-country", "entsoe-zonal") read a
    different file per split, so the source is built per split; turbine-level
    sources ignore the split and are resolved by country from the registry.
    """
    if spec.source in ("entsoe-country", "entsoe-zonal"):
        cls = EntsoeFileSource if spec.source == "entsoe-country" else EntsoeZonalFileSource
        return cls(spec.code, split, spec.train_years, spec.test_years[0])
    return get_source(spec.source, spec.code)


def load_obs_and_fleet(spec: RegionSpec, year: int, source: ObservationSource | None = None):
    """The observations and fleet :func:`run_evaluate` scores, without ERA5.

    For analyses that re-score a run's recorded frames. The whole fleet is
    returned (mode ``"all"``) with its own curves, as the scorecard rows were
    evaluated.

    Args:
        spec: The region.
        year: The test year.
        source: The adapter, resolved for the test split when omitted.

    Returns:
        Tuple of observations and turbine metadata, as :func:`vwf.data.val_obs_and_fleet`.
    """
    source = source if source is not None else resolve_source(spec, "test")
    return val_obs_and_fleet(spec.code, year, obs_level=spec.obs_level, source=source)


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
    run_dir: Path, fleet: pd.DataFrame, power_curves: pd.DataFrame, code: str
) -> dict:
    """Write ``curve_resolution.csv`` for the fleet a run simulates.

    A model missing from the curve table used to be visible only as a one-off
    warning, which is how every country-level row came to be simulated on a
    100 kW fallback curve for two months unnoticed. The record goes in the run
    directory, its summary in the manifest, and the substituted share into
    every metrics row, so it travels with the numbers the way ``fit_quality``
    does. Recording only: a substitution never stops a run.
    """
    resolution = curve_resolution(fleet, power_curves)
    resolution.to_csv(run_dir / CURVE_RESOLUTION_NAME, index=False)
    summary = summarise_curve_resolution(resolution)
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


def _era5_dir(spec: RegionSpec) -> Path:
    return PyVWFPaths.INPUT_ROOT / spec.era5_path


def _run_dir(out_root: str | Path, spec: RegionSpec, mode: str, run_name: str | None) -> Path:
    stamp = run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(out_root) / spec.code / f"{mode}-{stamp}"


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


def _record_accepted_years(factors: pd.DataFrame, time_res: str, n_training_years: int) -> dict:
    """How many accepted years each factor rests on, for the manifest (#28).

    The same counts are the factors table's ``n_years`` column; the manifest
    carries them so a run's record says, without the table, which factors rest
    on fewer than all their training years and which were refused.
    """
    per_slice: dict[str, dict[str, int]] = {}
    for slice_label, cluster, n in zip(factors[time_res], factors["cluster"], factors["n_years"]):
        per_slice.setdefault(str(slice_label), {})[str(cluster)] = int(n)
    refused = factors.loc[factors["scalar"].isna() & factors["offset"].isna(), "cluster"]
    return {
        "training_years": n_training_years,
        "min_accepted_years": min_accepted_years(n_training_years),
        "n_partial": int((factors["n_years"] < n_training_years).sum() - len(refused)),
        "n_refused": int(len(refused)),
        "per_factor": per_slice,
    }


def run_train(
    spec: RegionSpec,
    out_root: str | Path,
    *,
    source: ObservationSource | None = None,
    calc_z0: bool = True,
    mode: str = "all",
    run_name: str | None = None,
) -> Path:
    """Train correction factors for every (cluster, slice) combo in the config.

    Returns the run directory, containing ``factors_<slice>_<n>.csv``, the
    training fleet, and ``run_manifest.json``.
    """
    source = source if source is not None else resolve_source(spec)
    gen_cf, turb_info, reanalysis, power_curves = train_set(
        spec.code,
        calc_z0,
        mode,
        obs_level=spec.obs_level,
        source=source,
        era5_dir=_era5_dir(spec),
        bbox=spec.bbox,
        allow_extrapolation=spec.allow_extrapolation,
        roughness=spec.roughness,
    )

    model = get_correction(spec.correction_model)
    run_dir = _run_dir(out_root, spec, "train", run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    curves = _record_curve_resolution(run_dir, turb_info, power_curves, spec.code)
    era5_extent = _record_era5_extent(reanalysis, turb_info, spec)
    fit_record: dict[str, dict] = {}
    accepted_years: dict[str, dict] = {}

    for num_clu in spec.cluster_list:
        for time_res in spec.time_slices:
            factors, clus_info = model.fit(
                gen_cf,
                turb_info,
                reanalysis,
                power_curves,
                num_clusters=num_clu,
                time_res=time_res,
                seasons=spec.seasons,
                obs_level=spec.obs_level,
                min_cluster_size=spec.min_cluster_size,
            )
            factors.to_csv(run_dir / f"factors_{time_res}_{num_clu}.csv", index=False)
            accepted_years[f"{time_res}_{num_clu}"] = _record_accepted_years(
                factors, time_res, spec.train_years[1] - spec.train_years[0] + 1
            )
            clus_info.to_csv(run_dir / f"train_turb_info_{num_clu}.csv", index=False)
            # What the fitted pairs do to the speeds they were fitted on: a pair
            # that sends training days off the curve drops them from its own
            # objective. Recorded beside the dagger; it does not set it.
            diagnostics = wind.fit_diagnostics(
                reanalysis,
                clus_info,
                factors,
                time_res,
                power_curves,
                seasons=spec.seasons,
                years=spec.train_years,
            )
            diagnostics.to_csv(run_dir / f"fit_diagnostics_{time_res}_{num_clu}.csv", index=False)
            fit_record[f"{time_res}_{num_clu}"] = {
                k: v
                for k, v in fit_quality(factors, diagnostics=diagnostics).items()
                if k.startswith("max_") and k.endswith("_share")
            }

    write_manifest_safe(
        run_dir,
        spec,
        extra={
            "run_mode": "train",
            "fleet_mode": mode,
            "curve_resolution": curves,
            "era5_extent": era5_extent,
            "fit_diagnostics": fit_record,
            "accepted_years": accepted_years,
            "observation_quality": _record_observation_quality(source),
            "era5_roughness": _record_roughness(reanalysis, spec),
        },
    )
    return run_dir


def run_evaluate(
    spec: RegionSpec,
    train_run_dir: str | Path,
    out_root: str | Path,
    *,
    year: int | None = None,
    source: ObservationSource | None = None,
    calc_z0: bool = True,
    mode: str = "all",
    run_name: str | None = None,
    score_zones: ObservationSource | None = None,
) -> Path:
    """Evaluate a trained run against a held-out year.

    Writes ``metrics.csv`` with one row per variant (uncorrected baseline plus
    each factors file), saves every corrected-CF frame (``cor_cf_*.csv``, plus
    ``unc_cf.csv``) so runs can be diffed at frame level, and returns the
    evaluation run directory. Handles both obs levels: turbine-level clusters
    the test fleet against the training fleet; country-level reuses the grid
    points' own cluster assignments and scores the capacity-weighted country
    aggregate.

    Args:
        score_zones: Optional zonal observation source used for an extra
            per-zone metric. A zonal run supplies its own; pass one explicitly
            to score a NATIONALLY trained run zone by zone, which is what makes
            the two comparable. Without it the national fit is only ever judged
            on the national aggregate, which is its own training objective.
    """
    is_country = spec.obs_level == "country"
    year = int(year if year is not None else spec.test_years[0])
    train_run_dir = Path(train_run_dir)

    source = source if source is not None else resolve_source(spec, "test")
    obs_cf, turb_info, reanalysis, power_curves = val_set(
        spec.code,
        calc_z0,
        mode,
        year_test=year,
        obs_level=spec.obs_level,
        source=source,
        era5_dir=_era5_dir(spec),
        bbox=spec.bbox,
        allow_extrapolation=spec.allow_extrapolation,
        roughness=spec.roughness,
    )

    model = get_correction(spec.correction_model)
    run_dir = _run_dir(out_root, spec, f"evaluate-{year}", run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    curves = _record_curve_resolution(run_dir, turb_info, power_curves, spec.code)
    era5_extent = _record_era5_extent(reanalysis, turb_info, spec)

    # A zonal source can also be scored zone by zone. The national metric is the
    # joint optimiser's own objective, so it favours a national fit by
    # construction; the per-zone metric scores what a zonal fit actually
    # targets. Both are reported, distinguished by the "scope" column.
    zone_source = (
        score_zones
        if score_zones is not None
        else (source if spec.source == "entsoe-zonal" else None)
    )
    obs_zonal = zone_source.load_observations() if zone_source is not None else None

    def _pairs(sim_cf: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if not is_country:
            return {
                "fleet": collapse_pseudo_replicates(
                    _tidy_eval_frame(sim_cf, obs_cf, turb_info), spec
                )
            }
        out = {"national": _country_pairs(sim_cf, obs_cf, turb_info)}
        if obs_zonal is not None:
            # Scored on the grid's own zone assignments, not the run's cluster
            # count, so a 1-cluster run is still judged zone by zone.
            out["per-zone"] = _zonal_pairs(sim_cf, obs_zonal, turb_info)
        return out

    # Every variant's paired frame is built first and scored afterwards, all on
    # the rows every variant can score (see _score_on_common_rows).
    variants: list[dict] = []

    capacity = turb_info.assign(ID=turb_info["ID"].astype(str)).set_index("ID")["capacity"]
    unc_ws, unc_cf = wind.simulate_wind(reanalysis, turb_info, power_curves)
    unc_cf.to_csv(run_dir / "unc_cf.csv", index=False)
    variants.append(
        {
            "label": "uncorrected",
            "head": {"variant": "uncorrected", "num_clu": 1, "time_res": "none"},
            # The uncorrected row carries the fit-quality columns empty, so the
            # corrected rows' columns keep their place in metrics.csv.
            "extra": dict.fromkeys(fit_quality(pd.DataFrame()), np.nan),
            "tail": wind.off_curve_record(unc_ws, unc_cf, capacity, power_curves),
            "pairs": _pairs(unc_cf),
        }
    )
    del unc_ws, unc_cf

    for factors_path in sorted(train_run_dir.glob("factors_*.csv")):
        time_res, num_clu_str = factors_path.stem.split("_")[1:3]
        num_clu = int(num_clu_str)
        factors = pd.read_csv(factors_path)
        if is_country:
            # Grid points carry their own cluster assignments; no re-clustering
            # runs on the country-level path (mirrors PyVWF.simulate_cf). The
            # same resolution training used has to be reapplied here, or a
            # single-cluster national fit would be merged against a grid still
            # carrying its per-zone clusters and every factor would come out
            # NaN.
            clus_info = assign_country_clusters(turb_info, num_clu)
        else:
            train_fleet = pd.read_csv(train_run_dir / f"train_turb_info_{num_clu}.csv")
            clus_info = cluster_turbines(
                num_clu,
                train_fleet,
                False,
                turb_info,
                min_cluster_size=spec.min_cluster_size,
            )
        cor_ws, cor_cf = model.apply(
            reanalysis, clus_info, power_curves, factors, time_res, seasons=spec.seasons
        )
        cor_cf.to_csv(run_dir / f"cor_cf_{time_res}_{num_clu}.csv", index=False)
        # Skill alone hides a bad fit: a region can score as a corrected win
        # while carrying an implausible scalar or an offset that never
        # converged, so the fit diagnostics travel with every corrected row.
        diagnostics_path = train_run_dir / f"fit_diagnostics_{time_res}_{num_clu}.csv"
        diagnostics = pd.read_csv(diagnostics_path) if diagnostics_path.is_file() else None
        quality = fit_quality(factors, diagnostics=diagnostics)
        if quality["n_implausible_scalar"] or quality["n_failed_offset"]:
            warnings.warn(
                f"{spec.code} {time_res} k={num_clu}: "
                f"{quality['n_implausible_scalar']} implausible scalar(s) "
                f"(max {quality['max_scalar']:.3g}), "
                f"{quality['n_failed_offset']} failed offset fit(s) in "
                f"cluster(s) {quality['degenerate_clusters']}. "
                "The skill metric can still look good; see "
                "docs/findings/method-hourly-resolution.md."
            )
        variants.append(
            {
                "label": f"{time_res}_{num_clu}",
                "head": {
                    "variant": spec.correction_model,
                    "num_clu": num_clu,
                    "time_res": time_res,
                },
                "extra": quality,
                "tail": wind.off_curve_record(cor_ws, cor_cf, capacity, power_curves),
                "pairs": _pairs(cor_cf),
            }
        )
        del cor_ws, cor_cf

    rows, scoring = _score_on_common_rows(variants, spec.code, run_dir)
    off_curve = _record_off_curve(variants, spec.code)
    metrics_df = pd.DataFrame(rows)
    # Every variant simulates the same fleet on the same table.
    metrics_df["substituted_capacity_share"] = curves["substituted_capacity_share"]
    metrics_df["excluded_share"] = metrics_df["scope"].map(
        {scope: summary["excluded_share"] for scope, summary in scoring.items()}
    )
    metrics_df["extrapolated_capacity_share"] = era5_extent["capacity_share_outside_loaded_extent"]
    observation_quality = _record_observation_quality(source)
    metrics_df["observations_clipped_share"] = observation_quality.get("clipped_share", 0.0)
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)
    write_manifest_safe(
        run_dir,
        spec,
        extra={
            "run_mode": "evaluate",
            "observation_quality": observation_quality,
            "evaluation_year": year,
            "trained_from": str(train_run_dir),
            "curve_resolution": curves,
            "common_row_scoring": scoring,
            "era5_extent": era5_extent,
            "era5_roughness": _record_roughness(reanalysis, spec),
            "off_curve": off_curve,
        },
    )
    return run_dir


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
    present = ~np.isnan(vals)
    wsum = np.where(present, caps, 0.0).sum(axis=1)
    agg = np.where(wsum > 0, np.where(present, vals * caps, 0.0).sum(axis=1) / wsum, np.nan)

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
    present = ~np.isnan(vals)
    wsum = np.where(present, caps, 0.0).sum(axis=1)
    country = np.where(wsum > 0, np.where(present, vals * caps, 0.0).sum(axis=1) / wsum, np.nan)
    sim_country = pd.DataFrame({"time": sim["time"], "cf_sim": country})
    sim_country["ym"] = sim_country["time"].dt.to_period("M")
    sim_m = sim_country.groupby("ym")["cf_sim"].mean()

    obs = obs_country.copy()
    obs["time"] = pd.to_datetime(obs["time"])
    obs["ym"] = obs["time"].dt.to_period("M")
    obs_m = obs.groupby("ym")["obs"].mean().rename("cf_obs")

    return pd.concat([sim_m, obs_m], axis=1).rename_axis("ym").reset_index()


def collapse_factors(
    factors: pd.DataFrame, cluster_capacity: pd.Series, time_res: str
) -> pd.DataFrame:
    """Collapse per-cluster factors to one (scalar, offset) per time-slice.

    The collapse is CAPACITY-weighted over source clusters (design §7.1):
    weights are the installed capacity behind each cluster in the SOURCE
    region's training fleet, never an unweighted mean.

    A cluster with no factor to apply (a refused factor or a failed offset,
    NaN in either parameter) is left out of both the sums and the weights. A
    NaN term drops out of a pandas sum, so keeping its weight would pull the
    collapsed scalar and offset toward zero by that cluster's capacity share.
    A slice where no cluster has a factor collapses to NaN.
    """
    merged = factors.copy()
    merged["_w"] = merged["cluster"].map(cluster_capacity)
    if merged["_w"].isna().any():
        missing = sorted(int(c) for c in merged.loc[merged["_w"].isna(), "cluster"].unique())
        raise ValueError(
            f"no capacity weight for cluster(s) {missing}: the factors table and "
            "the source training fleet disagree"
        )
    usable = merged["scalar"].notna() & merged["offset"].notna()
    merged["_w"] = merged["_w"].where(usable, 0.0)
    merged["_ws"] = merged["scalar"].where(usable, 0.0) * merged["_w"]
    merged["_wo"] = merged["offset"].where(usable, 0.0) * merged["_w"]
    grouped = merged.groupby(time_res, as_index=False)[["_ws", "_wo", "_w"]].sum()
    collapsed = pd.DataFrame(
        {
            "cluster": 0,
            time_res: grouped[time_res],
            "scalar": grouped["_ws"] / grouped["_w"].where(grouped["_w"] > 0),
            "offset": grouped["_wo"] / grouped["_w"].where(grouped["_w"] > 0),
        }
    )
    return collapsed


def run_transfer(
    source_spec: RegionSpec,
    source_run_dir: str | Path,
    target_spec: RegionSpec,
    out_root: str | Path,
    *,
    year: int | None = None,
    target_source: ObservationSource | None = None,
    calc_z0: bool = True,
    mode: str = "all",
    run_name: str | None = None,
) -> Path:
    """Apply a source region's collapsed correction to a target region.

    Design §7, exactly: capacity-weighted collapse per slice, uniform
    application, season-name matching under the TARGET's definitions (the
    collapsed factors carry season NAMES; applying them with
    ``seasons=target_spec.seasons`` is what maps AU winter-trained factors
    onto the target's winter months).
    """
    check_transfer_pair(source_spec.code, target_spec.code)
    if target_spec.obs_level != "turbine":
        raise NotImplementedError(
            "Country-level transfer targets land with the Europe re-runs (Phase 2)."
        )
    source_run_dir = Path(source_run_dir)
    year = int(year if year is not None else target_spec.test_years[0])

    target_source = target_source if target_source is not None else resolve_source(target_spec)
    obs_cf, turb_info, reanalysis, power_curves = val_set(
        target_spec.code,
        calc_z0,
        mode,
        year_test=year,
        obs_level=target_spec.obs_level,
        source=target_source,
        era5_dir=_era5_dir(target_spec),
        bbox=target_spec.bbox,
        allow_extrapolation=target_spec.allow_extrapolation,
        roughness=target_spec.roughness,
    )

    model = get_correction(target_spec.correction_model)
    run_dir = _run_dir(out_root, target_spec, f"transfer-from-{source_spec.code}", run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    curves = _record_curve_resolution(run_dir, turb_info, power_curves, target_spec.code)
    era5_extent = _record_era5_extent(reanalysis, turb_info, target_spec)

    capacity = turb_info.assign(ID=turb_info["ID"].astype(str)).set_index("ID")["capacity"]

    def _variant(sim_ws: pd.DataFrame, sim_cf: pd.DataFrame, variant: str, time_res) -> dict:
        tidy = collapse_pseudo_replicates(_tidy_eval_frame(sim_cf, obs_cf, turb_info), target_spec)
        return {
            "label": variant,
            "head": {"variant": variant, "time_res": time_res},
            "extra": {},
            "tail": wind.off_curve_record(sim_ws, sim_cf, capacity, power_curves),
            "pairs": {"fleet": tidy},
        }

    # As in run_evaluate: every variant is scored on the rows all of them can score.
    variants = []
    unc_ws, unc_cf = wind.simulate_wind(reanalysis, turb_info, power_curves)
    variants.append(_variant(unc_ws, unc_cf, "uncorrected", "none"))

    # Uniform application: every target site is cluster 0.
    target_info = turb_info.copy()
    target_info["cluster"] = 0

    for factors_path in sorted(source_run_dir.glob("factors_*.csv")):
        time_res, num_clu_str = factors_path.stem.split("_")[1:3]
        factors = pd.read_csv(factors_path)
        source_fleet = pd.read_csv(source_run_dir / f"train_turb_info_{num_clu_str}.csv")
        cluster_capacity = source_fleet.groupby("cluster")["capacity"].sum()
        collapsed = collapse_factors(factors, cluster_capacity, time_res)
        collapsed.to_csv(run_dir / f"collapsed_factors_{time_res}_{num_clu_str}.csv", index=False)
        cor_ws, cor_cf = model.apply(
            reanalysis,
            target_info,
            power_curves,
            collapsed,
            time_res,
            seasons=target_spec.seasons,  # season-NAME matching, design §7.3
        )
        variants.append(
            _variant(cor_ws, cor_cf, f"transfer-from-{source_spec.code}-{num_clu_str}", time_res)
        )

    rows, scoring = _score_on_common_rows(variants, target_spec.code, run_dir)
    off_curve = _record_off_curve(variants, target_spec.code)
    pd.DataFrame(rows).drop(columns="scope").assign(
        substituted_capacity_share=curves["substituted_capacity_share"],
        excluded_share=scoring["fleet"]["excluded_share"],
        extrapolated_capacity_share=era5_extent["capacity_share_outside_loaded_extent"],
    ).to_csv(run_dir / "metrics.csv", index=False)
    write_manifest_safe(
        run_dir,
        target_spec,
        extra={
            "run_mode": "transfer",
            "curve_resolution": curves,
            "transfer_source_region": source_spec.code,
            "transfer_source_run": str(source_run_dir),
            "transfer_semantics": "capacity-weighted-collapse, uniform, season-name-matched",
            "evaluation_year": year,
            "common_row_scoring": scoring,
            "era5_extent": era5_extent,
            "era5_roughness": _record_roughness(reanalysis, target_spec),
            "off_curve": off_curve,
        },
    )
    return run_dir


# ---------------------------------------------------------------------------
# Public entry points for analyses that re-score recorded runs. Each is the
# function run_train or run_evaluate itself calls, so a study scores exactly as
# the harness does. The private names stay the harness's own.
# ---------------------------------------------------------------------------

#: Per scope: the columns that identify a row across variants, the weight
#: column (None for the unweighted monthly aggregates) and the unit column.
SCOPE_KEYS = _SCOPE_KEYS


def era5_dir(spec: RegionSpec) -> Path:
    """The ERA5 directory a region's runs load: its ``era5_path`` under the input root."""
    return _era5_dir(spec)


def tidy_eval_frame(
    sim_cf: pd.DataFrame, obs_cf: pd.DataFrame, turb_info: pd.DataFrame
) -> pd.DataFrame:
    """Simulated and observed monthly CFs per unit, as ``run_evaluate`` scores them."""
    return _tidy_eval_frame(sim_cf, obs_cf, turb_info)


def country_pairs(
    sim_cf: pd.DataFrame, obs_country: pd.DataFrame, turb_info: pd.DataFrame
) -> pd.DataFrame:
    """Monthly national simulated and observed CF, one row per month (``ym``).

    A missing side stays NaN, so several conditions can be scored on the same
    months (``restrict_to_common_rows``).
    """
    return _country_pairs(sim_cf, obs_country, turb_info)


def country_skill(sim_cf: pd.DataFrame, obs_country: pd.DataFrame, turb_info: pd.DataFrame) -> dict:
    """Capacity-weighted national CF against the observed series, as monthly means."""
    return _country_skill(sim_cf, obs_country, turb_info)


def error_metrics(merged: pd.DataFrame) -> dict:
    """MBE, MAE, RMSE and correlation over paired ``cf_sim`` and ``cf_obs`` columns."""
    return _error_metrics(merged)


def score_on_common_rows(variants: list[dict], code: str, run_dir: Path) -> tuple[list[dict], dict]:
    """Score every variant of one run on the rows all of them can score.

    As ``run_evaluate`` does: each scope is restricted to its common complete
    rows before any metric is computed, and the rows excluded are written to
    ``scoring_exclusions.csv`` under ``run_dir``.
    """
    return _score_on_common_rows(variants, code, run_dir)
