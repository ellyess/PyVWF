"""The European re-run's per-row comparison: published against new. Read-only.

One row at a time, this scores the published evaluate run and its re-run on the
rows common to both, and reports the paired interval for the difference. It is
the measurement the re-run plan (``docs/findings/method-eu-rerun-prereg.md``)
requires, so that the treatment change is measured rather than asserted.

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
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_bootstrap as bb
import roughness_treatment_study as rts
from vwf.harness import driver
from vwf.harness.regions import load_region
from vwf.harness.skill import restrict_to_common_rows


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


def main(code: str, out_dir: str, argv) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = _conditions(argv)
    labels = list(runs)
    baseline = labels[0]

    spec = load_region(Path("configs/regions/scorecard") / f"{bb.CONFIGS[code]}.toml")
    is_country = spec.obs_level == "country"
    reported = bb.REPORTED[code]
    year = int(json.loads((runs[baseline] / "run_manifest.json").read_text())["evaluation_year"])
    obs, turb_info = bb.load_obs_and_fleet(spec, year)

    # Every condition must score the same year and the same reported
    # configuration, or the difference is not the one the plan registered.
    treatments = {}
    for label, ev in runs.items():
        manifest = json.loads((ev / "run_manifest.json").read_text())
        if int(manifest["evaluation_year"]) != year:
            raise SystemExit(f"{code} {label}: evaluation year {manifest['evaluation_year']} "
                             f"differs from the baseline's {year}")
        treatments[label] = (manifest.get("era5_roughness") or {}).get("applied")

    frames, point = {}, {}
    for label, ev in runs.items():
        metrics = pd.read_csv(ev / "metrics.csv")
        for kind, frame in rts._frames(ev, spec, obs, turb_info, reported).items():
            frames[f"{label}_{kind}"] = frame
            got = rts._score(frame.dropna(subset=["cf_sim", "cf_obs"]), is_country)
            row = metrics[metrics["variant"] == "uncorrected"] if kind == "uncorrected" \
                else metrics[metrics["variant"] != "uncorrected"]
            if kind != "uncorrected":
                ts, k = reported.rsplit("_", 1)
                row = row[(row["time_res"] == ts) & (row["num_clu"] == int(k))]
            published = float(row.iloc[0]["rmse"])
            point[f"{label}_{kind}"] = got
            if abs(got["rmse"] - published) > 1e-12:
                raise SystemExit(f"{code} {label} {kind}: rebuilt RMSE {got['rmse']} differs "
                                 f"from metrics.csv {published}")

    keys, weight, _ = driver._SCOPE_KEYS["national" if is_country else "fleet"]
    common, excluded = restrict_to_common_rows(frames, keys, weight=weight)
    rng = np.random.default_rng(bb.SEED)
    if is_country:
        n = len(common[f"{baseline}_corrected"])
        idx = rng.integers(0, n, size=(bb.N_DRAWS, n))

        def boot(label):
            d = (common[label]["cf_sim"] - common[label]["cf_obs"]).to_numpy()
            return np.sqrt((d[idx] ** 2).mean(axis=1)), np.sqrt((d ** 2).mean())
    else:
        units = np.array(sorted(common[f"{baseline}_corrected"]["ID"].unique()))
        counts = np.stack([
            np.bincount(r, minlength=len(units))
            for r in rng.integers(0, len(units), size=(bb.N_DRAWS, len(units)))
        ]).astype(float)

        def boot(label):
            t = common[label]
            g = t.assign(w=t["capacity"], e=t["capacity"] * (t["cf_sim"] - t["cf_obs"]) ** 2)
            g = g.groupby("ID")[["w", "e"]].sum().reindex(units, fill_value=0.0)
            w, e = g["w"].to_numpy(), g["e"].to_numpy()
            return np.sqrt((counts @ e) / (counts @ w)), np.sqrt(e.sum() / w.sum())

    draws = {label: boot(label) for label in frames}
    rows = []

    def record(quantity, b, pt, paired=False):
        lo, hi = bb.ci(b)
        rows.append({"region": code, "quantity": quantity, "estimate": pt,
                     "ci_lo": lo, "ci_hi": hi, "width": hi - lo,
                     "consistent_with_zero": bool(lo <= 0 <= hi) if paired else None})

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
    frame.to_csv(out_dir / f"{code}_rerun_comparison.csv", index=False)
    excluded.to_csv(out_dir / f"{code}_rerun_excluded_rows.csv", index=False)

    scored = len(common[f"{baseline}_corrected"])
    print(f"{code}: conditions {labels}, applied roughness {treatments}, "
          f"rows scored {scored}, excluded {len(excluded)}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(frame.round(6).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
