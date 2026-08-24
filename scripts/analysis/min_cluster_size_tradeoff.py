"""Does min_cluster_size buy an honest fit, and what does it cost? (Chile)

Chile's shipped k=10 correction contains a wind scalar of 80.2 on a ONE-plant
cluster and an offset that never converged, while still scoring as a corrected
win at monthly resolution (docs/findings/method-hourly-resolution.md). This asks
whether refusing to fit tiny clusters removes that, and what it costs in skill.

REVISED HYPOTHESIS. The obvious story ("degenerate fits come from small
clusters, so forbid small clusters") is NOT what the cross-region survey shows
(docs/findings/method-scalar-bounds.md). Across 373,179 fitted scalars the
RATE of implausible scalars is roughly flat in cluster size (0.19% to 0.42%) and
is actually highest for clusters above 20 sites. What cluster size predicts is
SEVERITY: the worst scalar is 126 to 150 for clusters of five sites or fewer,
against 7 to 12 for larger ones. So min_cluster_size should be expected to cap
how bad a fit can get, not to eliminate bad fits.

PRE-SPECIFIED GATES (written before any run):

  G1 (severity capped). The maximum fitted scalar drops below 10, from the
     baseline 80.2. This is the survey's prediction made falsifiable.

  G2 (the correction still earns its place). Corrected held-out RMSE still beats
     the UNCORRECTED baseline for the same run. If merging makes the correction
     worse than doing nothing, it is not a fix.

  G3 (skill not materially bought at too high a price). Corrected RMSE degrades
     by less than 10% against the min_cluster_size=1 baseline.

PREDICTION (recorded before running): G1 passes. G2 passes. G3 is genuinely
uncertain and is the point of the experiment: merging dissolves the Atacama's
dedicated cluster into its neighbours, so those plants lose the (absurd but
level-matching) correction that was flattering the monthly metric. A G3 failure
would NOT mean the feature is wrong; it would mean Chile's monthly score was
partly resting on a fit nobody should trust, which is worth knowing either way.

    PYVWF_INPUT=input/combined PYTHONPATH=src python \\
        scripts/analysis/min_cluster_size_tradeoff.py
"""
from __future__ import annotations

import dataclasses
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYVWF_INPUT", "input/combined")
sys.path.insert(0, "src")

import pandas as pd  # noqa: E402

from vwf.harness.corrections import fit_quality  # noqa: E402
from vwf.harness.driver import run_evaluate, run_train  # noqa: E402
from vwf.harness.regions import load_region  # noqa: E402
from vwf.sources import get_source  # noqa: E402

warnings.simplefilter("ignore")

NUM_CLU = 10
TIME_RES = "fixed"
SIZES = (1, 3, 5)
OUT = Path("output/min_cluster_size")


def main() -> int:
    base = load_region(Path("configs/regions/cl.toml"))
    rows = []

    for m in SIZES:
        spec = dataclasses.replace(
            base, cluster_list=(NUM_CLU,), time_slices=(TIME_RES,),
            min_cluster_size=m,
        )
        tag = f"min{m}"
        print(f"=== min_cluster_size={m} ===", flush=True)
        train_dir = run_train(spec, OUT, source=get_source(spec.source, spec.code),
                              run_name=tag)
        eval_dir = run_evaluate(spec, train_dir, OUT,
                                source=get_source(spec.source, spec.code),
                                run_name=tag)

        factors = pd.read_csv(train_dir / f"factors_{TIME_RES}_{NUM_CLU}.csv")
        fleet = pd.read_csv(train_dir / f"train_turb_info_{NUM_CLU}.csv")
        q = fit_quality(factors)
        met = pd.read_csv(Path(eval_dir) / "metrics.csv")
        unc = met[met["variant"] == "uncorrected"].iloc[0]
        cor = met[met["variant"] == spec.correction_model].iloc[0]

        rows.append({
            "min_cluster_size": m,
            "clusters_kept": int(fleet["cluster"].nunique()),
            "smallest_cluster": int(fleet.groupby("cluster").size().min()),
            "max_scalar": q["max_scalar"],
            "n_implausible": q["n_implausible_scalar"],
            "n_failed_offset": q["n_failed_offset"],
            "unc_rmse": unc["rmse"], "cor_rmse": cor["rmse"],
            "unc_mbe": unc["mbe"], "cor_mbe": cor["mbe"],
            "cor_r": cor["pearson_r"],
        })

    t = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print("RAW TABLE: Chile k=10 fixed, held-out 2024, by min_cluster_size")
    print("=" * 96)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    base_row = t[t["min_cluster_size"] == 1].iloc[0]
    best = t[t["min_cluster_size"] > 1].sort_values("cor_rmse").iloc[0]
    g1 = best["max_scalar"] < 10
    g2 = best["cor_rmse"] < best["unc_rmse"]
    g3 = best["cor_rmse"] < base_row["cor_rmse"] * 1.10

    print(f"\nBaseline (min=1): max scalar {base_row['max_scalar']:.3g}, "
          f"corrected RMSE {base_row['cor_rmse']:.4f}")
    print(f"Best merged (min={int(best['min_cluster_size'])}): max scalar "
          f"{best['max_scalar']:.3g}, corrected RMSE {best['cor_rmse']:.4f}")
    print("\nPRE-SPECIFIED GATES")
    print(f"  G1 severity capped (<10):     {best['max_scalar']:.3g}"
          f"   [{'PASS' if g1 else 'FAIL'}]")
    print(f"  G2 beats uncorrected:         {best['cor_rmse']:.4f} vs "
          f"{best['unc_rmse']:.4f}   [{'PASS' if g2 else 'FAIL'}]")
    print(f"  G3 within 10% of baseline:    {best['cor_rmse']:.4f} vs "
          f"{base_row['cor_rmse'] * 1.10:.4f} allowed   [{'PASS' if g3 else 'FAIL'}]")

    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "cl_tradeoff.csv", index=False)
    print(f"\nwrote {OUT/'cl_tradeoff.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
