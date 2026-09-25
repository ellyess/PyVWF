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

import json
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
    train_set,
    val_obs_and_fleet,
    val_set,
)
from vwf.harness.corrections import fit_quality, get_correction
from vwf.provenance import (
    write_manifest_safe,
)
from vwf.harness.regions import RegionSpec
from vwf.harness.skill import (
    collapse_pseudo_replicates,
)
from vwf.sources import (
    EntsoeFileSource,
    EntsoeZonalFileSource,
    ObservationSource,
    get_source,
)

# The run records and the scoring moved to their own modules on 2026-09-25.
# Every name is imported back, as an explicit re-export where the driver does
# not call it, because tests and scripts import them from here.
from vwf.harness.records import (
    CurveSubstitutionError as CurveSubstitutionError,
    _record_accepted_years as _record_accepted_years,
    _record_curve_resolution,
    _record_era5_extent,
    _record_observation_quality as _record_observation_quality,
    _record_roughness,
)
from vwf.harness.scoring import (
    SCORING_EXCLUSIONS_NAME as SCORING_EXCLUSIONS_NAME,
    _SCOPE_KEYS,
    _country_pairs,
    _country_skill,
    _error_metrics,
    _record_off_curve,
    _score_on_common_rows,
    _tidy_eval_frame,
    _zonal_pairs,
    _zonal_skill as _zonal_skill,
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


def _era5_dir(spec: RegionSpec) -> Path:
    return PyVWFPaths.INPUT_ROOT / spec.era5_path


def _run_dir(out_root: str | Path, spec: RegionSpec, mode: str, run_name: str | None) -> Path:
    """The directory a new run writes to, refused if a run already holds it.

    A reused ``run_name`` used to write into the earlier run's directory, so
    files the new run does not write (a factors file of another cluster count,
    say) survived beside a new manifest that did not describe them.

    Raises:
        FileExistsError: If the directory exists and is not empty.
    """
    stamp = run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(out_root) / spec.code / f"{mode}-{stamp}"
    if run_dir.is_dir() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"{run_dir} already holds a run. Choose another run name, or remove "
            "that directory first."
        )
    return run_dir


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
        # The config's window is the one fitted, not the adapter's default;
        # before 2026-09-24 the two only agreed because every config matched.
        train_years=spec.train_years if spec.obs_level == "turbine" else None,
    )

    model = get_correction(spec.correction_model)
    run_dir = _run_dir(out_root, spec, "train", run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    curves = _record_curve_resolution(
        run_dir,
        turb_info,
        power_curves,
        spec.code,
        refuse_substitution=spec.obs_level == "country",
    )
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


def _check_train_run(spec: RegionSpec, train_run_dir: Path) -> None:
    """Refuse to evaluate a training run that does not belong to ``spec``.

    Evaluation scores every ``factors_*.csv`` it finds, on the rows every
    variant can score, so a factors file left from another configuration
    both adds a variant and can move the rows the reported one is scored on.
    And a run trained under another region, correction model or season
    mapping would be applied to the wrong clusters or months without error.

    Raises:
        ValueError: If a factors file lies outside the spec's cluster counts and
            time slices, or the training manifest names a different region,
            correction model or seasons.
    """
    expected = {(ts, k) for ts in spec.time_slices for k in spec.cluster_list}
    unexpected = []
    for path in sorted(train_run_dir.glob("factors_*.csv")):
        time_res, num_clu = path.stem.split("_")[1:3]
        if (time_res, int(num_clu)) not in expected:
            unexpected.append(path.name)
    if unexpected:
        raise ValueError(
            f"{train_run_dir} holds factors outside this config's cluster_list "
            f"{list(spec.cluster_list)} and time_slices {list(spec.time_slices)}: "
            f"{unexpected}. Evaluate with the config the run was trained with."
        )

    manifest_path = train_run_dir / "run_manifest.json"
    if not manifest_path.exists():
        warnings.warn(
            f"{train_run_dir} has no run_manifest.json; its region, correction model "
            "and seasons cannot be checked against the config",
            stacklevel=3,
        )
        return
    manifest = json.loads(manifest_path.read_text())
    recorded = {
        "region code": (manifest.get("region") or {}).get("code"),
        "correction model": (manifest.get("correction") or {}).get("model"),
        "seasons": {k: list(v) for k, v in (manifest.get("seasons") or {}).items()} or None,
    }
    wanted = {
        "region code": spec.code,
        "correction model": spec.correction_model,
        "seasons": {k: list(v) for k, v in spec.seasons.items()},
    }
    mismatched = [
        f"{name}: run {recorded[name]!r}, config {wanted[name]!r}"
        for name in wanted
        if recorded[name] is not None and recorded[name] != wanted[name]
    ]
    if mismatched:
        raise ValueError(
            f"{train_run_dir} was not trained under this config: " + "; ".join(mismatched)
        )


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
    _check_train_run(spec, train_run_dir)

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
    curves = _record_curve_resolution(
        run_dir, turb_info, power_curves, spec.code, refuse_substitution=is_country
    )
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
            # runs on the country-level path. The
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


def collapse_factors(
    factors: pd.DataFrame, cluster_capacity: pd.Series, time_res: str
) -> pd.DataFrame:
    """Collapse per-cluster factors to one (scalar, offset) per time-slice.

    The collapse is CAPACITY-weighted over source clusters (design §7.1):
    weights are the installed capacity behind each cluster in the SOURCE
    region's training fleet, never an unweighted mean.

    Only fitted clusters enter the sums and the weights. A refused factor or a
    failed offset (NaN in either parameter) is left out: a NaN term drops out
    of a pandas sum, so keeping its weight would pull the collapsed scalar and
    offset toward zero by that cluster's capacity share. An unfitted cluster
    (``n_years`` 0, carrying the identity) is left out too, since the identity
    is not a correction learned from the source. A slice where no cluster is
    fitted collapses to NaN.
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
    if "n_years" in merged.columns:
        usable &= merged["n_years"] > 0
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
