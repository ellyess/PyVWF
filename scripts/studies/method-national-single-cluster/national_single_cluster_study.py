"""Does a single national fit lose anything against the grid's own structure?

Registered in ``docs/findings/method-national-single-cluster-prereg.md``. Seven
country-level configurations: BE, ES, FR, IE, IT, NL and NO. Portugal and
Sweden are excluded before any result, both for a defective capacity register,
Portugal's repairable from the Global Wind Power Tracker and Sweden's derived
from its own generation.

**A country-level configuration has no cluster count to select.**
``vwf.data.assign_country_clusters`` accepts 1, or the number of clusters the
grid points already carry, and refuses everything else, because no clustering
step runs on that path. So there are two candidates, and the
one-standard-error rule reduces to: **take the grid's own count only if its
mean fold score beats one cluster's by more than one standard error of the
better mean.**

The candidate count is read from the **maintained** grid each configuration
loads, by resolving it the way ``EntsoeFileSource`` resolves it rather than by
listing files (``AGENTS.md``). Reading it from the control-point pool gave the
wrong answer for Norway and Portugal, the pool being built on the uniform
grids.

Protocol, metric rule and held constants are the parent study's, so the two are
comparable: forward chaining inside the training years, both metrics with the
smaller count taken on a disagreement, a refit on all training years and one
untouched test year.

**Both conservatisms favour one cluster**, forward chaining giving early folds
less data and the rule preferring the simpler model, and here that is also the
answer the standing caveat predicts. Every result saying one cluster is enough
carries that beside it.

Usage, from the repository root:

    PYVWF_INPUT=input/combined PYVWF_OFFSET_WORKERS=4 PYTHONPATH=src python \\
        scripts/studies/method-national-single-cluster/national_single_cluster_study.py <out_dir> [CODE ...]
"""
import importlib.util
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "cluster_selection_study", REPO / "scripts" / "studies" / "method-cluster-selection" / "cluster_selection_study.py")
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)

from vwf.harness import regions                      # noqa: E402
from vwf.harness.driver import resolve_source        # noqa: E402

#: Registered included set. PT and SE are excluded before any result.
INCLUDED = ("be", "es", "fr", "ie", "it", "nl", "no")
METRICS = ("rmse", "mae")


def grid_clusters(spec) -> int:
    """How many clusters this configuration's maintained grid defines."""
    source = resolve_source(spec, "train")
    return int(source.load_metadata()["cluster"].nunique())


def main(out_dir: str, *only: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wanted = [s.lower() for s in only] or list(INCLUDED)
    fold_rows, selections = [], []

    for stem in wanted:
        spec = regions.load_region(REPO / "configs" / "regions" / f"{stem}.toml")
        label = spec.code
        own = grid_clusters(spec)
        if own <= 1:
            print(f"{label}: the grid defines {own} cluster, so there is nothing "
                  "to compare; skipped and recorded.", flush=True)
            continue
        candidates = (1, own)
        started = time.monotonic()
        print(f"\n=== {label}: candidates {candidates}, train {spec.train_years}, "
              f"test {spec.test_years}", flush=True)

        plan = study.folds(spec.train_years)
        for train_years, year in plan:
            print(f"  fold: train {train_years} validate {year}", flush=True)
            metrics = study.evaluate_at(spec, out, "all", candidates, train_years,
                                        year, f"fold-{year}")
            for _, row in metrics[metrics["variant"] != "uncorrected"].iterrows():
                fold_rows.append({"row": label, "fold_year": year,
                                  "num_clu": int(row["num_clu"]),
                                  "rmse": float(row["rmse"]), "mae": float(row["mae"])})

        scores = pd.DataFrame([r for r in fold_rows if r["row"] == label])
        picks = {m: study.one_standard_error(scores, m) for m in METRICS}
        selected = min(picks[m][0] for m in METRICS)
        final = study.evaluate_at(spec, out, "all", candidates, spec.train_years,
                                  int(spec.test_years[0]), "final")
        final.to_csv(out / f"final_{label}.csv", index=False)
        at = {int(r["num_clu"]): r for _, r in
              final[final["variant"] != "uncorrected"].iterrows()}
        unc = final[final["variant"] == "uncorrected"].iloc[0]

        selections.append({
            "row": label, "folds": len(plan), "grid_clusters": own,
            "selected": selected, "selected_by_rmse": picks["rmse"][0],
            "selected_by_mae": picks["mae"][0],
            "metrics_disagreed": picks["rmse"][0] != picks["mae"][0],
            "test_mae_uncorrected": float(unc["mae"]),
            "test_mae_one": float(at[1]["mae"]),
            "test_mae_own": float(at[own]["mae"]),
            "minutes": round((time.monotonic() - started) / 60.0, 1)})
        pd.DataFrame([r for r in fold_rows if r["row"] == label]).to_csv(
            out / f"fold_scores_{label}.csv", index=False)
        pd.DataFrame([selections[-1]]).to_csv(out / f"selection_{label}.csv", index=False)
        print(f"  selected {selected} of {candidates}; "
              f"{selections[-1]['minutes']} minutes", flush=True)

    frame = pd.DataFrame(selections)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n=== every fold score, before any rule")
        print(pd.DataFrame(fold_rows).round(5).to_string(index=False))
        print("\n=== selections")
        print(frame.round(5).to_string(index=False))
        frame["own_beats_one"] = frame["test_mae_one"] - frame["test_mae_own"]
        print("\n=== N-G2 and N-G3, on the test year")
        print(frame[["row", "grid_clusters", "selected", "test_mae_one",
                     "test_mae_own", "own_beats_one"]].round(5).to_string(index=False))
        chose_own = int((frame["selected"] != 1).sum())
        print(f"  N-G2: the grid's own count selected in {chose_own} of {len(frame)} "
              "(needs 3)")
        survived = frame[(frame["selected"] != 1) & (frame["own_beats_one"] > 0.002)]
        print(f"  N-G3: of those, {len(survived)} beat one cluster on the test year "
              "by more than 0.002")
        print(f"  metrics disagreed in {int(frame['metrics_disagreed'].sum())} of "
              f"{len(frame)}")
    print("\nForward chaining and the one-standard-error rule both favour one "
          "cluster, and so does the standing caveat, so a row selecting one is "
          "not by itself evidence that one is right.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], *sys.argv[2:])
