"""Select each turbine-level configuration's cluster count without leakage.

Registered in ``docs/findings/method-cluster-selection-prereg.md``, with the
2026-09-15 amendments: forward chaining rather than leave-one-training-year-out,
the five turbine-level configurations only, and the one-standard-error rule.

Per configuration:

1. **Forward chaining inside the training years.** Fold *i* trains on a
   contiguous prefix and validates on the next training year, so no fold ever
   trains on a year after the one it validates. The test year is untouched.
2. **The one-standard-error rule**, applied separately to RMSE and to MAE: take
   the smallest cluster count whose mean fold score is at or below the best
   mean plus one standard error of that best.
3. **Smaller count wins a disagreement** between the two metrics, fixed before
   any sweep.
4. Refit at the selection on all training years, and report the single
   untouched test year against both registered baselines, B1 the chapter's own
   count and B2 a fixed ``k=100`` for every configuration.

**Two conservatisms compound and the report says so.** Forward chaining gives
early folds less data, which penalises a large cluster count, and the
one-standard-error rule already prefers the smallest defensible one. A
selection at the bottom of its grid is not evidence that the bottom is best.

Read-only apart from the run directories under ``<out_dir>``.

Usage, from the repository root:

    PYVWF_INPUT=input/combined PYVWF_OFFSET_WORKERS=4 PYTHONPATH=src python \\
        scripts/studies/method-cluster-selection/cluster_selection_study.py <out_dir> [label ...]
"""

import dataclasses
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pyvwf.cli.common import make_parser
from pyvwf.harness import driver, regions

ONSHORE_GRID = (1, 10, 25, 50, 100, 200, 500, 1000)
OFFSHORE_GRID = (1, 2, 3, 5, 10, 25, 50, 100)

ERA5_PATH, ROUGHNESS, TIME_SLICE = "era5/EU_2026-09", "derived", "fixed"

#: Study-scoped box for Denmark, 15.4 east rather than dk.toml's 13.5.
BBOX = {"dk": (7.5, 15.4, 54.0, 58.2)}

#: B2, one rule for every turbine-level configuration, fixed in advance.
FIXED_BASELINE = 100

#: label, config stem, fleet mode, and B1, the chapter's own cluster count.
CONFIGURATIONS = (
    ("DE onshore", "de", "onshore", 500),
    ("DK offshore", "dk", "offshore", 2),
    ("DK onshore", "dk", "onshore", 884),
    ("UK offshore", "uk", "offshore", 10),
    ("UK onshore", "uk", "onshore", 300),
)

METRICS = ("rmse", "mae")


def grid_for(mode: str, fleet_size: int) -> tuple[int, ...]:
    base = OFFSHORE_GRID if mode == "offshore" else ONSHORE_GRID
    return tuple(k for k in base if k <= fleet_size)


def folds(train_years: tuple[int, int]) -> list[tuple[tuple[int, int], int]]:
    """Forward chaining: a contiguous prefix, validated on the next year."""
    first, last = int(train_years[0]), int(train_years[1])
    return [((first, year - 1), year) for year in range(first + 1, last + 1)]


def one_standard_error(scores: pd.DataFrame, metric: str) -> tuple[int, int, float]:
    """The smallest count within one standard error of the best mean.

    Args:
        scores: one row per (cluster count, fold) with the metric column.
        metric: ``rmse`` or ``mae``.

    Returns:
        The selected count, the minimising count, and the threshold.
    """
    grouped = scores.groupby("num_clu")[metric]
    means, counts = grouped.mean(), grouped.count()
    # ddof=1: the sample standard deviation, which is undefined on one fold and
    # comes back NaN. A configuration with one fold has no spread to speak of,
    # so the interval collapses to the minimum and the rule picks it.
    errors = grouped.std(ddof=1) / np.sqrt(counts)
    best = int(means.idxmin())
    threshold = float(means[best] + (0.0 if np.isnan(errors[best]) else errors[best]))
    within = means.index[means <= threshold]
    return int(min(within)), best, threshold


def run_tag(mode: str, name: str) -> str:
    """The run name, carrying the fleet mode.

    ``pyvwf.harness.driver._run_dir`` keys a run directory on the region code and
    the run name and on nothing else, so two configurations of one region
    collide. This study varies the fleet mode, which the path does not carry:
    without the mode here, ``DK onshore`` and ``DK offshore`` wrote factors
    into one directory and ``run_evaluate``, which scores every
    ``factors_*.csv`` it finds, scored an eleven-count grid where eight were
    registered. Running one row per process satisfies the memory isolation
    rule and does nothing about this, because the rule is about processes and
    the collision is in the path.
    """
    return f"{mode}-{name}"


def evaluate_at(
    spec,
    out: Path,
    mode: str,
    clusters: tuple[int, ...],
    train_years: tuple[int, int],
    year: int,
    name: str,
) -> pd.DataFrame:
    """Train at every count and score the given year. Returns metrics.csv."""
    fold_spec = dataclasses.replace(
        spec,
        cluster_list=clusters,
        time_slices=(TIME_SLICE,),
        era5_path=ERA5_PATH,
        roughness=ROUGHNESS,
        train_years=train_years,
        test_years=(year,),
        bbox=BBOX.get(spec.code.lower(), spec.bbox),
    )
    tag = run_tag(mode, name)
    train_dir = driver.run_train(fold_spec, out, mode=mode, run_name=tag)
    evaluate_dir = driver.run_evaluate(fold_spec, train_dir, out, mode=mode, run_name=tag)
    metrics = pd.read_csv(evaluate_dir / "metrics.csv")
    # The path fix above should make this impossible. Checked anyway, because a
    # contaminated grid scores as an ordinary result and the first run of this
    # study did exactly that.
    scored = set(metrics.loc[metrics["variant"] != "uncorrected", "num_clu"].astype(int))
    unexpected = sorted(scored - set(clusters))
    if unexpected:
        raise RuntimeError(
            f"{train_dir} holds factors for {unexpected}, which this run did not "
            f"fit; it asked for {list(clusters)}. Another configuration has "
            "written to the same run directory."
        )
    return metrics


def main(out_dir: str, *only: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fold_rows, selections = [], []

    for label, stem, mode, chapter_count in CONFIGURATIONS:
        if only and label not in only:
            continue
        # The frozen config this study ran on, not the maintained one, which moved to
        # era5/EU_2026-09 and derived roughness after it ran (configs/regions/study/).
        spec = regions.load_region(Path("configs/regions/study") / f"{stem}_era5_eu_stored.toml")
        started = time.monotonic()
        print(f"\n=== {label}: train {spec.train_years}, test {spec.test_years}", flush=True)

        plan = folds(spec.train_years)
        grid = grid_for(mode, 10_000)
        print(f"  {len(plan)} forward-chaining folds, {len(grid)} counts {grid}", flush=True)

        for train_years, year in plan:
            print(f"  fold: train {train_years} validate {year}", flush=True)
            metrics = evaluate_at(spec, out, mode, grid, train_years, year, f"fold-{year}")
            fitted = metrics[metrics["variant"] != "uncorrected"]
            for _, row in fitted.iterrows():
                fold_rows.append(
                    {
                        "row": label,
                        "fold_year": year,
                        "num_clu": int(row["num_clu"]),
                        "rmse": float(row["rmse"]),
                        "mae": float(row["mae"]),
                    }
                )

        scores = pd.DataFrame([r for r in fold_rows if r["row"] == label])
        picks = {m: one_standard_error(scores, m) for m in METRICS}
        selected = min(picks[m][0] for m in METRICS)
        disagree = picks["rmse"][0] != picks["mae"][0]
        print(
            f"  selection: rmse {picks['rmse'][0]}, mae {picks['mae'][0]}, "
            f"taken {selected}{' (disagreed, smaller taken)' if disagree else ''}",
            flush=True,
        )

        candidates = sorted({selected, chapter_count, FIXED_BASELINE})
        final = evaluate_at(
            spec, out, mode, tuple(candidates), spec.train_years, int(spec.test_years[0]), "final"
        )
        final.to_csv(out / f"final_{label.replace(' ', '_')}.csv", index=False)
        at = {int(r["num_clu"]): r for _, r in final[final["variant"] != "uncorrected"].iterrows()}
        uncorrected = final[final["variant"] == "uncorrected"].iloc[0]

        selections.append(
            {
                "row": label,
                "folds": len(plan),
                "grid": str(grid),
                "selected": selected,
                "best_rmse_k": picks["rmse"][1],
                "selected_by_rmse": picks["rmse"][0],
                "selected_by_mae": picks["mae"][0],
                "metrics_disagreed": disagree,
                "B1_chapter": chapter_count,
                "B2_fixed": FIXED_BASELINE,
                "test_mae_selected": float(at[selected]["mae"]),
                "test_mae_B1": float(at[chapter_count]["mae"]),
                "test_mae_B2": float(at[FIXED_BASELINE]["mae"]),
                "test_mae_uncorrected": float(uncorrected["mae"]),
                "minutes": round((time.monotonic() - started) / 60.0, 1),
            }
        )
        print(f"  {label} done in {selections[-1]['minutes']} minutes", flush=True)

    # Per row, not one file per invocation. A shell loop running one row per
    # process, which the memory rule asks for, made each invocation overwrite
    # the last and left only the final row's scores on disk.
    frame = pd.DataFrame(selections)
    for label in frame["row"]:
        stem = label.replace(" ", "_")
        scores = pd.DataFrame([r for r in fold_rows if r["row"] == label])
        scores.to_csv(out / f"fold_scores_{stem}.csv", index=False)
        frame[frame["row"] == label].to_csv(out / f"selection_{stem}.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n=== every fold score, before any rule")
        print(pd.DataFrame(fold_rows).round(5).to_string(index=False))
        print("\n=== selections")
        print(frame.round(5).to_string(index=False))
        print("\n=== C-G2, against B2 (k=100), and C-G3, against B1")
        frame["beats_B2"] = frame["test_mae_B2"] - frame["test_mae_selected"]
        frame["beats_B1"] = frame["test_mae_B1"] - frame["test_mae_selected"]
        print(
            frame[
                [
                    "row",
                    "selected",
                    "test_mae_selected",
                    "test_mae_B2",
                    "beats_B2",
                    "test_mae_B1",
                    "beats_B1",
                ]
            ]
            .round(5)
            .to_string(index=False)
        )
        print(
            f"  beats B2 by more than 0.002 in "
            f"{int((frame['beats_B2'] > 0.002).sum())} of {len(frame)}"
        )
        print(
            f"  beats B1 by more than 0.002 in "
            f"{int((frame['beats_B1'] > 0.002).sum())} of {len(frame)}"
        )
        print(f"  metrics disagreed in {int(frame['metrics_disagreed'].sum())} of {len(frame)}")
    print(
        "\nForward chaining and the one-standard-error rule both favour fewer "
        "clusters, so a selection at the bottom of its grid is not evidence "
        "that the bottom is best."
    )
    print(f"written: {len(frame)} per-row score and selection files under {out}")


def cli(argv: list[str] | None = None) -> None:
    """Parse the recorded command line, ``<out_dir> [label ...]``, and run :func:`main`."""
    parser = make_parser(__doc__)
    parser.add_argument("out_dir", help="Directory for the run directories, under output/")
    parser.add_argument(
        "only", nargs="*", metavar="label", help="Rows of CONFIGURATIONS to run (default: all)"
    )
    args = parser.parse_args(argv)
    main(args.out_dir, *args.only)


if __name__ == "__main__":
    cli()
