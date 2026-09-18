"""The registered roughness comparison: R1 against R0, one row at a time. Read-only.

Runs the gates of ``docs/findings/method-roughness-treatment-prereg.md`` over
two evaluate runs of the same scorecard row, one with the stored annual-mean
roughness (R0) and one with the roughness derived per timestep (R1). Everything
else is held at the row's scorecard configuration, so the runs differ in the
treatment and nothing else.

Fixed by the pre-registration, before any result:

- the primary test is the paired interval for the R1 minus R0 corrected RMSE
  difference, from 1,000 paired draws with the seed of the curve library study,
  resampling units at turbine level and months at country level;
- all four frames (each condition's uncorrected and corrected) are scored on
  the rows common to all of them, because R1 can lose steps R0 never lost;
- G1 (turbine level): indistinguishable if the interval includes zero and is no
  wider than 0.004; resolved if it excludes zero; indeterminate otherwise. The
  0.004 is a judgement fixed in advance, not a measured quantity;
- G2 (country level): the same interval, no absolute floor, reported with its
  width;
- each condition's rebuilt metrics must reproduce its own ``metrics.csv``
  before anything is resampled.

It reads the local run tree and input root, so a third party cannot run it.

Usage, from the repository root, one region per process:

    PYTHONPATH=src python scripts/studies/method-roughness-treatment/roughness_treatment_study.py <CODE> \
        <R0 evaluate dir> <R1 evaluate dir> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))  # the shared tools
import baseline_bootstrap as bb  # noqa: E402
from vwf.cli.common import make_parser  # noqa: E402
from vwf.harness.driver import load_obs_and_fleet  # noqa: E402
from vwf.harness import driver  # noqa: E402
from vwf.harness.bootstrap import (  # noqa: E402
    percentile_interval,
    resample_counts,
    resample_indices,
    rmse_over_rows,
    unit_sums,
    weighted_rmse,
)
from vwf.harness.regions import load_region
from vwf.harness.skill import (
    collapse_pseudo_replicates,
    restrict_to_common_rows,
    skill_metrics,
)

G1_RESOLVED_WIDTH = 0.004


def _frames(ev: Path, spec, obs, turb_info, reported):
    """The uncorrected and reported-corrected paired frames of one run."""
    def pairs(sim_cf):
        if spec.obs_level == "country":
            return driver.country_pairs(sim_cf, obs, turb_info)
        return collapse_pseudo_replicates(
            driver.tidy_eval_frame(sim_cf, obs, turb_info), spec)
    return {
        "uncorrected": pairs(pd.read_csv(ev / "unc_cf.csv")),
        "corrected": pairs(pd.read_csv(ev / f"cor_cf_{reported}.csv")),
    }


def _score(frame, is_country):
    if is_country:
        return driver.error_metrics(frame)
    return skill_metrics(frame)


def main(code, r0_dir, r1_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = load_region(Path("configs/regions/scorecard") / f"{bb.CONFIGS[code]}.toml")
    is_country = spec.obs_level == "country"
    reported = bb.REPORTED[code]
    runs = {"R0": Path(r0_dir), "R1": Path(r1_dir)}
    year = int(json.loads((runs["R0"] / "run_manifest.json").read_text())["evaluation_year"])
    obs, turb_info = load_obs_and_fleet(spec, year)

    # The two runs must differ in the treatment and nothing else.
    treatments = {}
    for name, ev in runs.items():
        m = json.loads((ev / "run_manifest.json").read_text())
        treatments[name] = m["era5_roughness"]["applied"]
    if treatments["R0"] != "stored" or treatments["R1"] != "derived":
        raise SystemExit(f"{code}: expected stored then derived, got {treatments}")

    frames, point, published = {}, {}, {}
    for name, ev in runs.items():
        metrics = pd.read_csv(ev / "metrics.csv")
        for label, frame in _frames(ev, spec, obs, turb_info, reported).items():
            frames[f"{name}_{label}"] = frame
            got = _score(frame.dropna(subset=["cf_sim", "cf_obs"]), is_country)
            row = metrics[metrics["variant"] == "uncorrected"] if label == "uncorrected" else \
                metrics[metrics["variant"] != "uncorrected"]
            if label != "uncorrected":
                ts, k = reported.rsplit("_", 1)
                row = row[(row["time_res"] == ts) & (row["num_clu"] == int(k))]
            published[f"{name}_{label}"] = float(row.iloc[0]["rmse"])
            point[f"{name}_{label}"] = got
            if abs(got["rmse"] - published[f"{name}_{label}"]) > 1e-12:
                raise SystemExit(
                    f"{code} {name} {label}: rebuilt RMSE {got['rmse']} differs from "
                    f"metrics.csv {published[f'{name}_{label}']}"
                )

    keys, weight, _ = driver.SCOPE_KEYS["national" if is_country else "fleet"]
    common, excluded = restrict_to_common_rows(frames, keys, weight=weight)
    if is_country:
        n = len(common["R0_corrected"])
        idx = resample_indices(n, seed=bb.SEED, n_draws=bb.N_DRAWS)

        def boot(label):
            d = (common[label]["cf_sim"] - common[label]["cf_obs"]).to_numpy()
            return rmse_over_rows(d, idx), np.sqrt((d ** 2).mean())
    else:
        units = np.array(sorted(common["R0_corrected"]["ID"].unique()))
        n = len(units)
        counts = resample_counts(n, seed=bb.SEED, n_draws=bb.N_DRAWS)

        def boot(label):
            g = unit_sums(common[label], units)
            w, e = g["w"].to_numpy(), g["e"].to_numpy()
            return weighted_rmse(counts, e, w), np.sqrt(e.sum() / w.sum())

    draws = {label: boot(label) for label in frames}
    rows = []
    for label, (b, pt) in draws.items():
        lo, hi = percentile_interval(b)
        rows.append({"region": code, "quantity": f"{label} RMSE", "estimate": pt,
                     "ci_lo": lo, "ci_hi": hi, "width": hi - lo})
    comparisons = {
        "corrected RMSE, R1 minus R0": (draws["R1_corrected"], draws["R0_corrected"]),
        "uncorrected RMSE, R1 minus R0": (draws["R1_uncorrected"], draws["R0_uncorrected"]),
    }
    verdicts = {}
    for name, ((b1, p1), (b0, p0)) in comparisons.items():
        lo, hi = percentile_interval(b1 - b0)
        rows.append({"region": code, "quantity": name, "estimate": p1 - p0,
                     "ci_lo": lo, "ci_hi": hi, "width": hi - lo,
                     "consistent_with_zero": bool(lo <= 0 <= hi)})
        if name.startswith("corrected"):
            if lo > 0 or hi < 0:
                verdicts["gate"] = "resolved: " + ("R0 better" if p1 - p0 > 0 else "R1 better")
            elif (hi - lo) <= G1_RESOLVED_WIDTH or is_country:
                verdicts["gate"] = "indistinguishable" if not is_country else "consistent with zero"
            else:
                verdicts["gate"] = "indeterminate"
    # The gain each condition buys, for the record.
    for name in ("R0", "R1"):
        (bu, pu), (bc, pc) = draws[f"{name}_uncorrected"], draws[f"{name}_corrected"]
        lo, hi = percentile_interval(bu - bc)
        rows.append({"region": code, "quantity": f"{name} correction gain", "estimate": pu - pc,
                     "ci_lo": lo, "ci_hi": hi, "width": hi - lo,
                     "consistent_with_zero": bool(lo <= 0 <= hi)})

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / f"{code}_roughness_comparison.csv", index=False)
    # P3: what each condition could not simulate, from its own metrics.csv.
    losses = []
    for name, ev in runs.items():
        m = pd.read_csv(ev / "metrics.csv")
        for _, r in m.iterrows():
            losses.append({
                "region": code, "condition": name, "variant": r["variant"],
                "time_res": r["time_res"], "num_clu": r["num_clu"],
                **{c: r[c] for c in (
                    "off_curve_below_share", "off_curve_above_share", "no_speed_share",
                    "unit_months_wholly_missing", "unit_months_partly_missing",
                    "max_below_zero_share", "max_above_curve_share",
                    "max_period_dropped_share") if c in m.columns},
            })
    pd.DataFrame(losses).to_csv(out_dir / f"{code}_roughness_losses.csv", index=False)
    excluded.to_csv(out_dir / f"{code}_roughness_excluded_rows.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(f"{code}: rows scored {len(common['R0_corrected'])}, "
              f"excluded {len(excluded)}; gate: {verdicts.get('gate')}")
        print(out.to_string(index=False))


def cli(argv: list[str] | None = None) -> None:
    """Parse the recorded command line, ``<CODE> <R0 dir> <R1 dir> <out_dir>``, and run :func:`main`."""
    parser = make_parser(__doc__)
    parser.add_argument("code", help="Scorecard row, a key of CONFIGS, e.g. DK")
    parser.add_argument("r0_dir", help="The R0 evaluate run (annual-mean roughness)")
    parser.add_argument("r1_dir", help="The R1 evaluate run (per-timestep roughness)")
    parser.add_argument("out_dir", help="Directory for the outputs, under output/")
    args = parser.parse_args(argv)
    main(args.code, args.r0_dir, args.r1_dir, args.out_dir)


if __name__ == "__main__":
    cli()
