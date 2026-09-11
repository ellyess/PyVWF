"""How much a scorecard row's result depends on dropping off-curve days. Read-only.

A corrected speed outside the power curve table (below 0 m/s, or above its last
speed, 40 m/s) gives a missing capacity factor, not zero output, and a monthly
or national mean skips it (``missing_value_audit.py``). Physically, both give
zero output: below cut-in, or above cut-out. This measures what the scores
would be if those days counted as zero. It does not change the simulation.

For each variant of one scorecard row it recomputes the corrected speeds and
capacity factors through the harness's own ``apply``, on the evaluation's ERA5,
and checks them against the saved ``cor_cf_*.csv`` frame, NaN for NaN. It then
scores every variant three ways:

- ``published``: the row's ``metrics.csv``, scored before the common-row fix;
- ``common``: the fixed harness's common-row scoring of the saved frames;
- ``zero_fill``: the same, after every off-curve corrected value is set to 0.
  Values missing for another reason (a failed offset, or a missing input) stay
  missing, because no corrected speed exists for them.

For each, it reports RMSE, MBE, correlation and the mean simulated CF over the
scored rows (capacity-weighted per unit-month at turbine level, the mean
national month at country level), with the mean observed CF on the same rows.

Fixed before any row was run, on 2026-09-11: the three bases, the metrics, the
zero-fill rule (both directions, nothing else), the reproduction check, and
every variant of all 17 scorecard rows.

Added after the first row (ES) was run, on the same day. ES's uncorrected
values were missing on the same days as many of its off-curve corrected ones,
because the uncorrected speed is itself off the curve: the hub-height profile
``ws100 * ln(h/z0) / ln(100/z0)``, with z0 estimated daily from the 10 m and
100 m winds, goes negative when z0 lies between hub height and 100 m, and
blows up as z0 approaches 100 m. Two measurements were added:

- ``zero_fill_both``: as ``zero_fill``, and every off-curve uncorrected value
  is also set to 0. The uncorrected speeds are recomputed and checked against
  ``unc_cf.csv`` first. Zero output is not physical for a speed the profile
  broke, so this basis only bounds how much the uncorrected side depends on
  those days;
- per row, the unit-days where the hub-height profile broke in a contributing
  ERA5 cell, whether or not the speed lands on the curve (hubs at or below
  100 m only; see the comment in the code).

It needs ERA5, the local input root and the git-ignored run tree, so a third
party cannot run it. Usage, from the repository root, one region per process,
with ``PYVWF_INPUT`` as in the row's manifest:

    PYTHONPATH=src python scripts/analysis/off_curve_sensitivity.py <CODE> <out_dir>

Output: ``<CODE>_off_curve_sensitivity.csv``, one row per variant and basis.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import baseline_bootstrap as bb
from vwf.clustering import cluster_turbines
from vwf.data import assign_country_clusters, load_power_curves, val_set
from vwf import wind
from vwf.harness import driver
from vwf.harness.corrections import get_correction
from vwf.harness.regions import load_region
from vwf.harness.skill import collapse_pseudo_replicates, restrict_to_common_rows


def main(code, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = load_region(Path("configs/regions/scorecard") / f"{bb.CONFIGS[code]}.toml")
    ev = next((bb.BACKFILL / code).glob("evaluate-*-backfill"))
    manifest = json.loads((ev / "run_manifest.json").read_text())
    year = int(manifest["evaluation_year"])
    train_dir = Path(manifest["trained_from"])
    is_country = spec.obs_level == "country"
    published = pd.read_csv(ev / "metrics.csv")

    obs_cf, turb_info, reanalysis, power_curves = val_set(
        spec.code, True, "all", year_test=year, obs_level=spec.obs_level,
        source=driver.resolve_source(spec, "test"), era5_dir=driver._era5_dir(spec),
        bbox=spec.bbox,
    )
    top_speed = float(load_power_curves().iloc[:, 0].max())
    model = get_correction(spec.correction_model)

    def pairs(sim_cf):
        if is_country:
            return {"national": driver._country_pairs(sim_cf, obs_cf, turb_info)}
        return {"fleet": collapse_pseudo_replicates(
            driver._tidy_eval_frame(sim_cf, obs_cf, turb_info), spec)}

    unc = pd.read_csv(ev / "unc_cf.csv", parse_dates=["time"])
    ucols = [c for c in unc.columns if c != "time"]
    unc_ws, unc_cf = wind.simulate_wind(reanalysis, turb_info, power_curves)
    unc_ws = unc_ws.set_index("time")
    unc_cf = unc_cf.set_index("time")
    unc_ws.columns = unc_ws.columns.astype(str)
    unc_cf.columns = unc_cf.columns.astype(str)
    if not np.allclose(unc_cf.loc[unc["time"], ucols].to_numpy(), unc[ucols].to_numpy(),
                       equal_nan=True, rtol=0, atol=1e-12):
        raise SystemExit(f"{code}: recomputed uncorrected CF differs from unc_cf.csv")
    uws = unc_ws.loc[unc["time"], ucols].to_numpy()
    unc_values = unc[ucols].to_numpy(copy=True)
    unc_outside = np.isnan(unc_values) & ((uws < 0) | (uws > top_speed))
    unc_values[unc_outside] = 0.0
    unc_filled = unc.copy()
    unc_filled[ucols] = unc_values
    # Unit-days where the hub-height profile broke in at least one contributing
    # ERA5 cell. Speed is profiled per cell and then interpolated, so the unit's
    # own z0 is not the test. For a hub at or below 100 m, a valid log profile
    # gives a hub speed between 0 and the 100 m speed in every cell, and so at
    # the unit; a ratio outside (0, 1] means some cell's profile was invalid.
    # Hubs above 100 m are not tested (the valid range has no fixed upper bound).
    ids = np.asarray(turb_info["ID"], dtype=object)
    at = {
        "lon": xr.DataArray(np.asarray(turb_info["lon"], float), dims="turbine", coords={"turbine": ids}),
        "lat": xr.DataArray(np.asarray(turb_info["lat"], float), dims="turbine", coords={"turbine": ids}),
    }
    ws100 = reanalysis["wnd100m"].interp(**at, kwargs={"fill_value": None}).transpose("time", "turbine")
    ws100 = ws100.to_pandas()
    ws100.columns = ws100.columns.astype(str)
    ws100 = ws100.loc[unc["time"], ucols].to_numpy()
    hub = np.asarray(turb_info.assign(ID=turb_info["ID"].astype(str)).set_index("ID").loc[ucols, "height"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = uws / ws100
    tested = np.broadcast_to(hub[None, :] <= 100.0, ratio.shape) & (ws100 > 0)
    profile_invalid = int((tested & ((ratio <= 0) | (ratio > 1))).sum())
    profile_tested = int(tested.sum())

    base = {"label": "uncorrected", "extra": {}, "pairs": pairs(unc),
            "head": {"variant": "uncorrected", "num_clu": 1, "time_res": "none"}}
    base_filled = {**base, "pairs": pairs(unc_filled)}
    saved_variants, filled_variants, off_curve = [], [], {}
    for path in sorted(ev.glob("cor_cf_*.csv")):
        time_res, num_clu = path.stem.removeprefix("cor_cf_").rsplit("_", 1)
        label = f"{time_res}_{num_clu}"
        factors = pd.read_csv(train_dir / f"factors_{time_res}_{num_clu}.csv")
        if is_country:
            clus_info = assign_country_clusters(turb_info, int(num_clu))
        else:
            fleet = pd.read_csv(train_dir / f"train_turb_info_{num_clu}.csv")
            clus_info = cluster_turbines(int(num_clu), fleet, False, turb_info,
                                         min_cluster_size=spec.min_cluster_size)
        cor_ws, cor_cf = model.apply(reanalysis, clus_info, power_curves, factors, time_res,
                                     seasons=spec.seasons)
        saved = pd.read_csv(path, parse_dates=["time"])
        cols = [c for c in saved.columns if c != "time"]
        cor_cf = cor_cf.set_index("time")
        cor_ws = cor_ws.set_index("time")
        cor_cf.columns = cor_cf.columns.astype(str)
        cor_ws.columns = cor_ws.columns.astype(str)
        cor_cf = cor_cf.loc[saved["time"], cols]
        cor_ws = cor_ws.loc[saved["time"], cols]
        if not np.allclose(cor_cf.to_numpy(), saved[cols].to_numpy(), equal_nan=True, rtol=0, atol=1e-12):
            raise SystemExit(f"{code} {label}: recomputed corrected CF differs from {path.name}")

        ws = cor_ws.to_numpy()
        values = saved[cols].to_numpy(copy=True)
        outside = np.isnan(values) & ((ws < 0) | (ws > top_speed))
        off_curve[label] = int(outside.sum())
        values[outside] = 0.0
        filled = saved.copy()
        filled[cols] = values
        head = {"variant": spec.correction_model, "num_clu": int(num_clu), "time_res": time_res}
        saved_variants.append({"label": label, "extra": {}, "head": head, "pairs": pairs(saved)})
        filled_variants.append({"label": label, "extra": {}, "head": head, "pairs": pairs(filled)})

    rows = []
    bases = (("common", base, saved_variants), ("zero_fill", base, filled_variants),
             ("zero_fill_both", base_filled, filled_variants))
    for basis, base_variant, variants in bases:
        scratch = out_dir / f"{code}_{basis}"
        scratch.mkdir(exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scored, _ = driver._score_on_common_rows([base_variant, *variants], code, scratch)
        frames = {v["label"]: v["pairs"] for v in [base_variant, *variants]}
        scope = "national" if is_country else "fleet"
        # Mean CFs over the rows scored, which _score_on_common_rows does not return.
        keys, weight, _ = driver._SCOPE_KEYS[scope]
        restricted, _ = restrict_to_common_rows(
            {k: f[scope] for k, f in frames.items()}, keys, weight=weight)
        for row, label in zip(scored, frames):
            r = restricted[label]
            w = r[weight] if weight else pd.Series(1.0, index=r.index)
            rows.append({
                "region": code, "basis": basis, "label": label,
                "reported": label == bb.REPORTED[code],
                "rmse": row["rmse"], "mbe": row["mbe"], "pearson_r": row["pearson_r"],
                "mean_sim_cf": float(np.average(r["cf_sim"], weights=w)),
                "mean_obs_cf": float(np.average(r["cf_obs"], weights=w)),
                "n_rows": len(r),
                "off_curve_values_filled": (
                    off_curve.get(label, 0) if basis != "common" and label != "uncorrected"
                    else int(unc_outside.sum()) if basis == "zero_fill_both" else 0),
                "unit_days_profile_invalid": profile_invalid, "unit_days_tested": profile_tested,
            })
    for _, p in published.iterrows():
        label = "uncorrected" if p["variant"] == "uncorrected" else f"{p['time_res']}_{p['num_clu']}"
        rows.append({"region": code, "basis": "published", "label": label,
                     "reported": label == bb.REPORTED[code], "rmse": p["rmse"], "mbe": p["mbe"],
                     "pearson_r": p["pearson_r"]})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / f"{code}_off_curve_sensitivity.csv", index=False)
    rep = out[out["reported"] | (out["label"] == "uncorrected")]
    with pd.option_context("display.width", 200):
        print(rep[["basis", "label", "rmse", "mbe", "pearson_r", "mean_sim_cf", "mean_obs_cf",
                   "n_rows", "off_curve_values_filled"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
