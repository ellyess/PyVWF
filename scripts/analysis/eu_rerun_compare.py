"""Per-row paired comparison of two or three evaluate runs of one row. Read-only.

One row at a time, this scores each condition on the rows common to all of
them, and reports the paired interval for every difference. It was written for
the European re-run (``docs/findings/method-eu-rerun-prereg.md``), so that the
treatment change was measured rather than asserted, and it is the comparison
driver the curve library study registered as well
(``docs/findings/method-curve-library-prereg.md``, "Tooling this needs"): one
definition of these numbers rather than a second that could drift from it. The
conditions are named on the command line, so the module is not specific to
either study; ``--tag`` names the output files.

Two or three conditions, named on the command line, the first being the
baseline every difference is taken against:

- **Two** for the rows where only the treatment changes (DE, DK, UK, FR, BE,
  IE): ``published`` and ``new``.
- **Three** for Sweden and Norway, where the treatment and the loaded extent
  both change: ``published``, ``oldfiles_derived`` and ``new``. The second
  minus the first is the treatment alone, since only the roughness differs;
  the third minus the second is the extent alone, since only the files differ.
  All three are scored on the rows common to all three, so the two terms add
  to the total by construction rather than by luck.

Spain, Italy and Portugal take no comparison at all. Their published figures
are not results, so there is nothing to difference against, and the plan says
so in advance.

Fixed by the plan and by procedure B of the curve library study, before any
re-run existed: 1,000 paired draws, the seed of that study, units resampled
for a turbine-level row and months for a country-level one, the row's reported
configuration on both sides, and every condition's rebuilt point metrics
reproducing its own ``metrics.csv`` to 1e-12 before anything is resampled.

The frame building and scoring are imported from the roughness study rather
than rewritten. Two builders would be two definitions of the same number, and
the DK figures this re-run extends came from that one.

Usage, from the repository root, one region per process:

    PYTHONPATH=src:scripts/analysis python scripts/analysis/eu_rerun_compare.py \
        SE output/eu_rerun_2026-09-12/analysis \
        published=<dir> oldfiles_derived=<dir> new=<dir>

    PYTHONPATH=src:scripts/analysis python scripts/analysis/eu_rerun_compare.py \
        DK output/curve_library_study_2026-09-13/analysis --tag=T1 \
        T0=<dir> T1=<dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent / "studies" / "method-roughness-treatment")]
import baseline_bootstrap as bb  # noqa: E402
from vwf.harness.driver import load_obs_and_fleet  # noqa: E402
import roughness_treatment_study as rts  # noqa: E402
from vwf.cli.common import make_parser  # noqa: E402
from vwf.harness import driver  # noqa: E402
from vwf.harness.bootstrap import (  # noqa: E402
    percentile_interval,
    resample_counts,
    resample_indices,
    rmse_over_rows,
    unit_sums,
    weighted_rmse,
)
from vwf.harness.regions import load_region  # noqa: E402
from vwf.harness.skill import restrict_to_common_rows  # noqa: E402


def drop_run_exclusions(frame: pd.DataFrame, ev: Path, keys: list[str], scope: str) -> pd.DataFrame:
    """The frame without the rows the run itself excluded from its score.

    A run scores every variant on the rows all of its variants can score, and
    writes the rest to ``scoring_exclusions.csv``. Rebuilding a frame from
    ``unc_cf.csv`` and ``cor_cf_*.csv`` reconstructs every row, including those,
    so the rebuilt score reproduces ``metrics.csv`` only for a run that excluded
    nothing. Every European row and every C1 and C2 run excluded nothing, which
    is why this went unnoticed until the US, whose runs each exclude 11 rows at
    0.019% of capacity.

    Without this, the 1e-12 self-check compares a number scored on all rows
    with one scored on the common rows and refuses a run that is perfectly
    sound. With it, each side is checked against the convention it was written
    under, and the cross-condition restriction that follows is unchanged.
    """
    path = ev / "scoring_exclusions.csv"
    if not path.exists():
        return frame
    excluded = pd.read_csv(path)
    excluded = excluded[excluded["scope"] == scope]
    if excluded.empty or not set(keys) <= set(excluded.columns):
        return frame
    left = frame.assign(**{k: frame[k].astype(str) for k in keys})
    right = excluded[keys].astype(str).drop_duplicates().assign(_excluded=True)
    merged = left.merge(right, on=keys, how="left")
    return frame[merged["_excluded"].isna().to_numpy()]


def _conditions(argv) -> dict[str, Path]:
    out = {}
    for item in argv:
        if "=" not in item:
            raise SystemExit(f"expected <label>=<evaluate dir>, got {item!r}")
        label, path = item.split("=", 1)
        out[label] = Path(path)
    if len(out) < 2:
        raise SystemExit("need at least a baseline and one condition")
    return out


def main(code: str, out_dir: str, argv, tag: str = "rerun") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = _conditions(argv)
    labels = list(runs)
    baseline = labels[0]

    spec = load_region(Path("configs/regions/scorecard") / f"{bb.CONFIGS[code]}.toml")
    is_country = spec.obs_level == "country"
    reported = bb.REPORTED[code]
    year = int(json.loads((runs[baseline] / "run_manifest.json").read_text())["evaluation_year"])
    obs, turb_info = load_obs_and_fleet(spec, year)

    # Every condition must score the same year and the same reported
    # configuration, or the difference is not the one the plan registered.
    treatments = {}
    for label, ev in runs.items():
        manifest = json.loads((ev / "run_manifest.json").read_text())
        if int(manifest["evaluation_year"]) != year:
            raise SystemExit(
                f"{code} {label}: evaluation year {manifest['evaluation_year']} "
                f"differs from the baseline's {year}"
            )
        treatments[label] = (manifest.get("era5_roughness") or {}).get("applied")

    scope = "national" if is_country else "fleet"
    keys, weight, _ = driver.SCOPE_KEYS[scope]

    frames, point = {}, {}
    for label, ev in runs.items():
        metrics = pd.read_csv(ev / "metrics.csv")
        for kind, frame in rts._frames(ev, spec, obs, turb_info, reported).items():
            frame = drop_run_exclusions(frame, ev, keys, scope)
            frames[f"{label}_{kind}"] = frame
            got = rts._score(frame.dropna(subset=["cf_sim", "cf_obs"]), is_country)
            row = (
                metrics[metrics["variant"] == "uncorrected"]
                if kind == "uncorrected"
                else metrics[metrics["variant"] != "uncorrected"]
            )
            if kind != "uncorrected":
                ts, k = reported.rsplit("_", 1)
                row = row[(row["time_res"] == ts) & (row["num_clu"] == int(k))]
            published = float(row.iloc[0]["rmse"])
            point[f"{label}_{kind}"] = got
            if abs(got["rmse"] - published) > 1e-12:
                raise SystemExit(
                    f"{code} {label} {kind}: rebuilt RMSE {got['rmse']} differs "
                    f"from metrics.csv {published}"
                )

    common, excluded = restrict_to_common_rows(frames, keys, weight=weight)
    if is_country:
        n = len(common[f"{baseline}_corrected"])
        idx = resample_indices(n, seed=bb.SEED, n_draws=bb.N_DRAWS)

        def boot(label):
            d = (common[label]["cf_sim"] - common[label]["cf_obs"]).to_numpy()
            return rmse_over_rows(d, idx), np.sqrt((d**2).mean())
    else:
        units = np.array(sorted(common[f"{baseline}_corrected"]["ID"].unique()))
        counts = resample_counts(len(units), seed=bb.SEED, n_draws=bb.N_DRAWS)

        def boot(label):
            g = unit_sums(common[label], units)
            w, e = g["w"].to_numpy(), g["e"].to_numpy()
            return weighted_rmse(counts, e, w), np.sqrt(e.sum() / w.sum())

    draws = {label: boot(label) for label in frames}
    rows = []

    def record(quantity, b, pt, paired=False):
        lo, hi = percentile_interval(b)
        rows.append(
            {
                "region": code,
                "quantity": quantity,
                "estimate": pt,
                "ci_lo": lo,
                "ci_hi": hi,
                "width": hi - lo,
                "consistent_with_zero": bool(lo <= 0 <= hi) if paired else None,
            }
        )

    for label in frames:
        b, pt = draws[label]
        record(f"{label} RMSE", b, pt)
    for kind in ("corrected", "uncorrected"):
        b0, p0 = draws[f"{baseline}_{kind}"]
        for label in labels[1:]:
            b1, p1 = draws[f"{label}_{kind}"]
            record(f"{kind} RMSE, {label} minus {baseline}", b1 - b0, p1 - p0, paired=True)
    if len(labels) == 3:
        mid, last = labels[1], labels[2]
        for kind in ("corrected", "uncorrected"):
            b1, p1 = draws[f"{mid}_{kind}"]
            b2, p2 = draws[f"{last}_{kind}"]
            record(f"{kind} RMSE, {last} minus {mid}", b2 - b1, p2 - p1, paired=True)
    for label in labels:
        (bu, pu), (bc, pc) = draws[f"{label}_uncorrected"], draws[f"{label}_corrected"]
        record(f"{label} correction gain", bu - bc, pu - pc, paired=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / f"{code}_{tag}_comparison.csv", index=False)
    excluded.to_csv(out_dir / f"{code}_{tag}_excluded_rows.csv", index=False)

    scored = len(common[f"{baseline}_corrected"])
    print(
        f"{code}: conditions {labels}, applied roughness {treatments}, "
        f"rows scored {scored}, excluded {len(excluded)}"
    )
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(frame.round(6).to_string(index=False))


def cli(argv: list[str] | None = None) -> None:
    """Parse the recorded command line and run :func:`main`.

    ``<CODE> <out_dir> [--tag=TAG] NAME=DIR...``, with ``--tag`` anywhere.
    """
    parser = make_parser(__doc__)
    parser.add_argument("code", help="Scorecard row, a key of CONFIGS, e.g. SE")
    parser.add_argument("out_dir", help="Directory for the outputs, under output/")
    parser.add_argument(
        "conditions",
        nargs="+",
        metavar="NAME=DIR",
        help="Each condition's evaluate run, the first being the baseline",
    )
    parser.add_argument("--tag", default="rerun", help="Names the output files (default: rerun)")
    args = parser.parse_intermixed_args(argv)
    main(args.code, args.out_dir, args.conditions, tag=args.tag)


if __name__ == "__main__":
    cli()
