"""Rescore a scorecard row's evaluate frames on common rows. Read-only.

Until the harness was fixed, ``run_evaluate`` scored each variant on its own
complete rows. A corrected variant with no value for some units was therefore
compared with the uncorrected one on a different set of rows. This takes a
row's saved evaluate frames (``unc_cf.csv``, ``cor_cf_*.csv`` under
``output/validation/curve_resolution_backfill_2026-09-11/``) and scores every
variant through the fixed harness's own ``_score_on_common_rows``. It then
compares the result, variant by variant, with the row's ``metrics.csv``.

Fixed before any row was rescored, on 2026-09-11:

- every variant in the run is rescored, not only the reported one, because the
  fixed harness restricts all variants of a run to the rows every one of them
  can score;
- a variant counts as changed if any of RMSE, MBE, MAE, ``n_units``,
  ``n_samples`` or ``n_months`` differs from ``metrics.csv`` by more than 1e-12;
- as a guard, a row with no exclusions must reproduce its ``metrics.csv``
  exactly, or the script stops. This checks that the rescoring path is the
  one that produced the file.

Output per row, in ``<out_dir>``: ``<CODE>_rescore.csv`` (old against new,
per variant and scope) and ``<CODE>/scoring_exclusions.csv`` (as the fixed
harness writes it). ``all_rescore.csv`` joins the rows.

Added after the rescore results, on the same day: for a turbine-level row with
exclusions, the paired resampled gain of the reported configuration on the
common rows (``<CODE>_common_rows_bootstrap.csv``). It uses the same draws,
seed and interval as ``baseline_bootstrap.py``, and the same capacity-effective
count and top-unit shares as ``unit_concentration.py``. The gain intervals in
the UK and NZ notice were computed before the fix, on each variant's own rows;
for rows with no exclusions they are unchanged. CL's first interval, -0.007 to
0.020, was computed on the rows common to the uncorrected and ``fixed_10``
variants only (55 plants), not on the rows common to all three (53 plants),
which the fixed harness scores.

Also added after the rescore results: ``--joint DIR [DIR ...]`` rescores a
set of evaluate runs of one region whose results a document compares across
runs. It was written for the Chile ``min_cluster_size`` runs, whose gates
compare corrected scores from three runs with one uncorrected score. Each run
is rescored on its own common rows, as the fixed harness would score it, and
all runs together on the rows common to every variant of every run. The
uncorrected frames of the runs must be identical. Output:
``<CODE>_joint_rescore.csv``.

It reads the git-ignored run tree and the local input root, so a third party
cannot run it. It is committed so that the figures in the CL correction notice
can be regenerated.

Usage, from the repository root, one region per process, with ``PYVWF_INPUT``
as in the row's manifest:

    PYTHONPATH=src python scripts/analysis/common_row_rescore.py <CODE> <out_dir>
    PYTHONPATH=src python scripts/analysis/common_row_rescore.py <CODE> <out_dir> --joint DIR...
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_bootstrap as bb
from vwf.harness.driver import load_obs_and_fleet
from vwf.harness import driver
from vwf.harness.bootstrap import percentile_interval, resample_counts, weighted_mean, weighted_rmse
from vwf.harness.regions import load_region
from vwf.harness.skill import collapse_pseudo_replicates, restrict_to_common_rows

COMPARED = ("rmse", "mbe", "mae", "n_units", "n_samples", "n_months")


def main(code, out_dir):
    spec = load_region(Path("configs/regions/scorecard") / f"{bb.CONFIGS[code]}.toml")
    ev = next((bb.BACKFILL / code).glob("evaluate-*-backfill"))
    manifest = json.loads((ev / "run_manifest.json").read_text())
    year = int(manifest["evaluation_year"])
    old = pd.read_csv(ev / "metrics.csv")
    obs, turb_info = load_obs_and_fleet(spec, year)

    def pairs(sim_cf):
        if spec.obs_level == "country":
            return {"national": driver._country_pairs(sim_cf, obs, turb_info)}
        return {"fleet": collapse_pseudo_replicates(
            driver._tidy_eval_frame(sim_cf, obs, turb_info), spec)}

    variants = [{
        "label": "uncorrected",
        "head": {"variant": "uncorrected", "num_clu": 1, "time_res": "none"},
        "extra": {},
        "pairs": pairs(pd.read_csv(ev / "unc_cf.csv")),
    }]
    for path in sorted(ev.glob("cor_cf_*.csv")):
        time_res, num_clu = path.stem.removeprefix("cor_cf_").rsplit("_", 1)
        variants.append({
            "label": f"{time_res}_{num_clu}",
            "head": {"variant": spec.correction_model, "num_clu": int(num_clu),
                     "time_res": time_res},
            "extra": {},
            "pairs": pairs(pd.read_csv(path)),
        })

    region_dir = Path(out_dir) / code
    region_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows, scoring = driver._score_on_common_rows(variants, code, region_dir)
    new = pd.DataFrame(rows)

    key = ["variant", "num_clu", "time_res", "scope"]
    merged = old.merge(new, on=key, suffixes=("_old", "_new"), validate="one_to_one")
    if len(merged) != len(old) or len(merged) != len(new):
        raise SystemExit(f"{code}: variants do not line up between old and new scoring")
    compared = [m for m in COMPARED if f"{m}_old" in merged.columns]
    changed = pd.Series(False, index=merged.index)
    for m in compared:
        changed |= (merged[f"{m}_old"] - merged[f"{m}_new"]).abs() > 1e-12
    merged["changed"] = changed
    n_excluded = sum(s["n_rows_excluded"] for s in scoring.values())
    if n_excluded == 0 and changed.any():
        raise SystemExit(f"{code}: no exclusions, yet the rescoring differs from metrics.csv")

    report = merged[key + [c for m in compared for c in (f"{m}_old", f"{m}_new")] + ["changed"]]
    report = report.assign(region=code, n_rows_excluded=n_excluded,
                           excluded_share=[scoring[s]["excluded_share"] for s in merged["scope"]],
                           units_wholly_excluded=[";".join(scoring[s]["units_wholly_excluded"])
                                                  for s in merged["scope"]])
    report.to_csv(Path(out_dir) / f"{code}_rescore.csv", index=False)
    if spec.obs_level == "turbine" and n_excluded:
        frames = {v["label"]: v["pairs"]["fleet"] for v in variants}
        restricted, _ = restrict_to_common_rows(frames, ["ID", "year", "month"])
        common_rows_bootstrap(code, restricted, bb.REPORTED[code], Path(out_dir))
    print(f"{code}: {int(changed.sum())} of {len(merged)} variant rows change; "
          f"{n_excluded} row(s) excluded")


def common_rows_bootstrap(code, restricted, reported, out_dir):
    """Paired resampled gain of the reported configuration, on the common rows."""
    per_unit = {}
    for name in ("uncorrected", reported):
        t = restricted[name]
        d = t["cf_sim"] - t["cf_obs"]
        per_unit[name] = t.assign(
            w=t["capacity"], e=t["capacity"] * d**2, a=t["capacity"] * d.abs()
        ).groupby("ID")[["w", "e", "a"]].sum().sort_index()
    u, c = per_unit["uncorrected"], per_unit[reported]
    if not u.index.equals(c.index):
        raise SystemExit(f"{code}: common rows give different unit sets")
    n = len(u)
    counts = resample_counts(n, seed=bb.SEED, n_draws=bb.N_DRAWS)

    def rmse(g):
        return weighted_rmse(counts, g.e.values, g.w.values)

    def mae(g):
        return weighted_mean(counts, g.a.values, g.w.values)

    w = u.w.to_numpy()
    shares = (c.e / c.e.sum()).sort_values(ascending=False)
    out = {
        "region": code, "reported": reported, "units": n,
        "capacity_effective_n": float(w.sum() ** 2 / (w**2).sum()),
        "uncorrected_rmse": float(np.sqrt(u.e.sum() / u.w.sum())),
        "corrected_rmse": float(np.sqrt(c.e.sum() / c.w.sum())),
        "uncorrected_mbe": float((restricted["uncorrected"].eval("capacity * (cf_sim - cf_obs)")).sum()
                                 / restricted["uncorrected"]["capacity"].sum()),
        "corrected_mbe": float((restricted[reported].eval("capacity * (cf_sim - cf_obs)")).sum()
                               / restricted[reported]["capacity"].sum()),
        "cor_sse_top1": float(shares.iloc[0]), "cor_sse_top5": float(shares.head(5).sum()),
    }
    out["rmse_gain"] = out["uncorrected_rmse"] - out["corrected_rmse"]
    out["rmse_gain_ci_lo"], out["rmse_gain_ci_hi"] = percentile_interval(rmse(u) - rmse(c))
    out["mae_gain"] = float(u.a.sum() / u.w.sum() - c.a.sum() / c.w.sum())
    out["mae_gain_ci_lo"], out["mae_gain_ci_hi"] = percentile_interval(mae(u) - mae(c))
    pd.DataFrame([out]).to_csv(out_dir / f"{code}_common_rows_bootstrap.csv", index=False)
    print(pd.Series(out).to_string())


def joint(code, out_dir, eval_dirs):
    """Rescore several evaluate runs of one region, each alone and all together."""
    spec = load_region(Path("configs/regions/scorecard") / f"{bb.CONFIGS[code]}.toml")
    eval_dirs = [Path(d) for d in eval_dirs]
    years = {json.loads((d / "run_manifest.json").read_text())["evaluation_year"] for d in eval_dirs}
    if len(years) != 1:
        raise SystemExit(f"{code}: runs evaluate different years {years}")
    obs, turb_info = load_obs_and_fleet(spec, int(years.pop()))

    def pairs(sim_cf):
        return {"fleet": collapse_pseudo_replicates(
            driver._tidy_eval_frame(sim_cf, obs, turb_info), spec)}

    unc = pd.read_csv(eval_dirs[0] / "unc_cf.csv")
    for d in eval_dirs[1:]:
        other = pd.read_csv(d / "unc_cf.csv")
        same = other.columns.equals(unc.columns) and np.allclose(
            other.drop(columns="time").to_numpy(float), unc.drop(columns="time").to_numpy(float),
            equal_nan=True, rtol=0, atol=0)
        if not same:
            raise SystemExit(f"{code}: uncorrected frame of {d} differs from {eval_dirs[0]}")
    unc_variant = {"label": "uncorrected", "extra": {}, "pairs": pairs(unc),
                   "head": {"run": "all", "variant": "uncorrected", "num_clu": 1, "time_res": "none"}}

    per_run = {}
    for d in eval_dirs:
        per_run[d.name] = []
        for path in sorted(d.glob("cor_cf_*.csv")):
            time_res, num_clu = path.stem.removeprefix("cor_cf_").rsplit("_", 1)
            per_run[d.name].append({
                "label": f"{d.name}:{time_res}_{num_clu}", "extra": {},
                "head": {"run": d.name, "variant": spec.correction_model,
                         "num_clu": int(num_clu), "time_res": time_res},
                "pairs": pairs(pd.read_csv(path)),
            })

    rows = []
    scratch = Path(out_dir) / f"{code}_joint"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, variants in per_run.items():
            (scratch / name).mkdir(parents=True, exist_ok=True)
            scored, summary = driver._score_on_common_rows([unc_variant, *variants], code, scratch / name)
            old = pd.read_csv(Path([d for d in eval_dirs if d.name == name][0]) / "metrics.csv")
            for row in scored:
                ref = old[(old["variant"] == row["variant"]) & (old["num_clu"] == row["num_clu"])
                          & (old["time_res"] == row["time_res"])].iloc[0]
                rows.append({"basis": f"within:{name}", **row, "run": row["run"] if row["run"] != "all" else name,
                             "rmse_old": ref["rmse"], "n_units_old": ref["n_units"],
                             "n_samples_old": ref["n_samples"],
                             "excluded_share": summary["fleet"]["excluded_share"]})
        pooled = [unc_variant] + [v for vs in per_run.values() for v in vs]
        (scratch / "all").mkdir(parents=True, exist_ok=True)
        scored, summary = driver._score_on_common_rows(pooled, code, scratch / "all")
        for row in scored:
            rows.append({"basis": "joint", **row, "excluded_share": summary["fleet"]["excluded_share"]})
    report = pd.DataFrame(rows)
    report.to_csv(Path(out_dir) / f"{code}_joint_rescore.csv", index=False)
    with pd.option_context("display.width", 220):
        print(report[["basis", "run", "variant", "rmse", "mbe", "n_units", "n_samples",
                      "rmse_old", "n_units_old", "excluded_share"]].round(4).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "--joint":
        joint(sys.argv[1], sys.argv[2], sys.argv[4:])
    else:
        main(sys.argv[1], sys.argv[2])
