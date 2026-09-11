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

It reads the git-ignored run tree and the local input root, so a third party
cannot run it. It is committed so that the figures in the CL correction notice
can be regenerated.

Usage, from the repository root, one region per process, with ``PYVWF_INPUT``
as in the row's manifest:

    PYTHONPATH=src python scripts/analysis/common_row_rescore.py <CODE> <out_dir>
"""
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

import baseline_bootstrap as bb
from vwf.harness import driver
from vwf.harness.regions import load_region
from vwf.harness.skill import collapse_pseudo_replicates

COMPARED = ("rmse", "mbe", "mae", "n_units", "n_samples", "n_months")


def main(code, out_dir):
    spec = load_region(Path("configs/regions/scorecard") / f"{bb.CONFIGS[code]}.toml")
    ev = next((bb.BACKFILL / code).glob("evaluate-*-backfill"))
    manifest = json.loads((ev / "run_manifest.json").read_text())
    year = int(manifest["evaluation_year"])
    old = pd.read_csv(ev / "metrics.csv")
    obs, turb_info = bb.load_obs_and_fleet(spec, year)

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
    print(f"{code}: {int(changed.sum())} of {len(merged)} variant rows change; "
          f"{n_excluded} row(s) excluded")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
