#!/usr/bin/env python3
"""Score the scorecard's affine rows on the physics-informed model's test rows.

``method-physics-informed.md`` compares its zero-shot model with the affine
correction fitted in region. Its "affine, in region" figures were read from the
cluster sweep of 2026-07-24, which the scorecard has since replaced, and they
were scored on the harness's rows with the harness's uncorrected simulation,
not on the rows the physics-informed model was scored on. This puts the current
scorecard rows on the model's own test rows, scored as the model's drivers
score (``pyvwf.pinn.runs.score_on_common_rows``: pseudo-replicates collapsed,
then the capacity-weighted ``skill_metrics`` on the rows every condition can
score), so the two columns differ only in the correction and the simulation
chain behind it.

For each region it reads the cache's test observations and capacities, and the
scorecard row's evaluate run (``unc_cf.csv`` and ``cor_cf_<slice>_<k>.csv``,
daily, averaged to calendar months as the harness does). It writes
``affine_on_pinn_rows.csv``, one row per region (with the uncorrected
simulation also scored alone on every model row it covers, and the affine row
scored there with the uncorrected value wherever it has none), and a ``run_manifest.json``.
It refuses a dirty tree unless ``--allow-dirty`` is given.

Run: PYTHONPATH=src /opt/miniconda3/envs/pyvwf/bin/python scripts/dev/run_locked.py -- \
         /opt/miniconda3/envs/pyvwf/bin/python scripts/studies/physics-informed/affine_on_pinn_rows.py \
         --out output/pinn_affine_rescore_2026-10-16
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from pyvwf.harness.regions import load_region  # noqa: E402
from pyvwf.harness.skill import (  # noqa: E402
    collapse_pseudo_replicates,
    restrict_to_common_rows,
    skill_metrics,
)
from pyvwf.provenance import build_manifest, write_manifest  # noqa: E402

KEYS = ["ID", "year", "month"]
RERUN = "output/pinn_rerun_2026-09-16/cache"
LOCO = "output/pinn_loco_2026-09-16/cache"
REFRESH = "output/validation/refresh_2026-09-20"
C1 = "output/c1_turbine_order_2026-09-24/after"

#: Region: scorecard config stem, test year, corrected slice and k, the cache
#: the model's figure was scored on, and the evaluate run behind the
#: scorecard row (AU-NEM's and NZ's corrected figures come from the
#: 2026-09-24 re-evaluation, per the scorecard's notice of that date).
ROWS = {
    "DK": ("dk_k100", 2020, "season", 100, RERUN, f"{REFRESH}/DK/evaluate-2020-refresh"),
    "DE": ("de_k100", 2019, "fixed", 100, RERUN, f"{REFRESH}/DE/evaluate-2019-refresh"),
    "UK": ("uk_k50", 2019, "fixed", 50, RERUN, f"{REFRESH}/UK/evaluate-2019-refresh"),
    "US": ("us_k250", 2022, "fixed", 250, RERUN, f"{REFRESH}/US/evaluate-2022-refresh"),
    "BR": ("br_k60", 2024, "fixed", 60, RERUN, f"{REFRESH}/BR/evaluate-2024-refresh"),
    "AR": ("ar_k10", 2024, "fixed", 10, LOCO, f"{REFRESH}/AR/evaluate-2024-refresh"),
    "AU-NEM": ("au_nem_k45", 2023, "season", 45, LOCO, f"{C1}/AU-NEM/evaluate-2023-after"),
    "CL": ("cl_k10", 2024, "fixed", 10, LOCO, f"{REFRESH}/CL/evaluate-2024-refresh"),
    "NZ": ("nz_k7", 2024, "fixed", 7, LOCO, f"{C1}/NZ/evaluate-2024-after"),
}


def monthly_long(cf_wide: pd.DataFrame) -> pd.DataFrame:
    """A daily (time x ID) frame as calendar-month means, long, as the harness scores."""
    cf = cf_wide.copy()
    cf["time"] = pd.to_datetime(cf["time"])
    monthly = cf.groupby(pd.Grouper(key="time", freq="ME")).mean().reset_index()
    long = monthly.melt(id_vars=["time"], var_name="ID", value_name="cf_sim")
    long["year"] = long["time"].dt.year
    long["month"] = long["time"].dt.month
    long["ID"] = long["ID"].astype(str)
    return long[[*KEYS, "cf_sim"]]


def scorecard_rmse(run: Path, variant: str, k: int | None, slice_: str | None) -> float:
    """The fleet RMSE the run's own metrics.csv records for one variant."""
    m = pd.read_csv(run / "metrics.csv")
    m = m[(m["scope"] == "fleet") & (m["variant"] == variant)]
    if k is not None:
        m = m[(m["num_clu"] == k) & (m["time_res"] == slice_)]
    if len(m) != 1:
        raise SystemExit(f"{run}: expected one {variant} row, found {len(m)}")
    return float(m["rmse"].iloc[0])


def score_region(code: str, root: Path) -> dict:
    stem, year, slice_, k, cache, run_rel = ROWS[code]
    run = root / run_rel
    spec = load_region(root / "configs" / "regions" / "scorecard" / f"{stem}.toml")
    test = root / cache / f"{code}_test"
    obs = pd.read_csv(test / "obs.csv").dropna(subset=["obs"]).rename(columns={"obs": "cf_obs"})
    obs["ID"] = obs["ID"].astype(str)
    obs = obs[obs["year"] == year]
    cap = pd.read_csv(test / "meta.csv")[["ID", "capacity"]]
    cap["ID"] = cap["ID"].astype(str)
    base = obs.merge(cap, on="ID", how="left")

    sims = {
        "uncorrected": pd.read_csv(run / "unc_cf.csv"),
        "affine": pd.read_csv(run / f"cor_cf_{slice_}_{k}.csv"),
    }
    frames, coverage = {}, {}
    for label, wide in sims.items():
        f = base.merge(monthly_long(wide), on=KEYS, how="left")
        has = f["cf_sim"].notna()
        coverage[label] = (
            float(has.mean()),
            float(f.loc[has, "capacity"].sum() / f["capacity"].sum()),
        )
        frames[label] = collapse_pseudo_replicates(
            f.loc[has, [*KEYS, "cf_sim", "cf_obs", "capacity"]], spec
        )
    restricted, _ = restrict_to_common_rows(frames, KEYS, weight="capacity")
    met = {label: skill_metrics(frame) for label, frame in restricted.items()}
    # The harness's uncorrected simulation on every row the model was scored
    # on, so the two simulation chains' uncorrected errors can be compared on
    # identical rows, apart from the rows the harness has no simulation for.
    unc_all = skill_metrics(frames["uncorrected"])
    # The affine row with a refused or unscored unit falling back to the
    # uncorrected simulation, which is what a user of the correction gets
    # there, scored on every model row: the like-for-like comparison with a
    # model that scores every row.
    fallback = base.merge(monthly_long(sims["affine"]), on=KEYS, how="left")
    unc_long = monthly_long(sims["uncorrected"]).rename(columns={"cf_sim": "cf_unc"})
    fallback = fallback.merge(unc_long, on=KEYS, how="left")
    fallback["cf_sim"] = fallback["cf_sim"].fillna(fallback["cf_unc"])
    fallback = fallback.dropna(subset=["cf_sim"])[[*KEYS, "cf_sim", "cf_obs", "capacity"]]
    aff_fallback = skill_metrics(collapse_pseudo_replicates(fallback, spec))
    return {
        "region": code,
        "test_year": year,
        "config": f"k{k} {slice_}",
        "run": run_rel,
        "cache": cache,
        "model_rows": int(len(obs)),
        "affine_row_share": coverage["affine"][0],
        "affine_capacity_share": coverage["affine"][1],
        "common_rows": int(len(restricted["affine"])),
        "uncorrected_row_share": coverage["uncorrected"][0],
        "uncorrected_rmse_all_model_rows": unc_all["rmse"],
        "affine_fallback_rmse_all_model_rows": aff_fallback["rmse"],
        "affine_fallback_mbe_all_model_rows": aff_fallback["mbe"],
        "uncorrected_rmse": met["uncorrected"]["rmse"],
        "affine_rmse": met["affine"]["rmse"],
        "uncorrected_mbe": met["uncorrected"]["mbe"],
        "affine_mbe": met["affine"]["mbe"],
        "scorecard_uncorrected_rmse": scorecard_rmse(run, "uncorrected", None, None),
        "scorecard_affine_rmse": scorecard_rmse(run, "affine-wind", k, slice_),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--regions", nargs="+", default=list(ROWS))
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    launch = build_manifest()
    if launch["git_dirty"] and not args.allow_dirty:
        raise SystemExit("refusing to run on a dirty tree. Commit first, or pass --allow-dirty.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for code in args.regions:
        row = score_region(code, ROOT)
        rows.append(row)
        print(
            f"[{code}] uncorrected {row['uncorrected_rmse']:.4f} affine {row['affine_rmse']:.4f} "
            f"on {row['common_rows']} rows; affine covers {row['affine_capacity_share']:.1%} "
            f"of the model's test capacity",
            flush=True,
        )
    table = pd.DataFrame(rows)
    table.to_csv(out / "affine_on_pinn_rows.csv", index=False)
    inputs = {
        str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        for code in args.regions
        for p in (
            f"{ROWS[code][5]}/metrics.csv",
            f"{ROWS[code][5]}/run_manifest.json",
            f"{ROWS[code][4]}/{code}_test/obs.csv",
        )
    }
    write_manifest(
        out,
        build_manifest(
            extra={"run_mode": "pinn-affine-on-pinn-rows", "argv": sys.argv[1:], "inputs": inputs}
        ),
    )


if __name__ == "__main__":
    main()
