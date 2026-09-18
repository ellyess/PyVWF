#!/usr/bin/env python3
"""Leave one turbine-level region out: does a terrain correction transfer?

The registered design is docs/findings/method-physics-informed-turbine-prereg.md.
Each fold holds out one of the nine turbine-level regions and trains on the
other eight. Every arm is scored on the fold's test year per unit, and the
primary metric is the spatial part of the error (``vwf.pinn.runs.level_spatial``),
which a national series cannot see.

Arms:

  uncorrected   ERA5 through the power curves with no correction.
  no-terrain    fitted on the pool; speed-up fixed at zero, shear offset global.
  relief-only   fitted on the pool; the relief term with one global strength.
  full          fitted on the pool; terrain features set the strength and shear.
  gwa-ratio     fitted on the pool; the speed-up is the Global Wind Atlas ratio,
                not learned, from scripts/pinn/gwa_ratio.py.
  in-region     fitted on the fold's own training years.

The fleet head is the same in every fitted arm. A secondary scoring fits
no-terrain, relief-only and full on all nine regions and applies them to the
nine country-level European folds, as national series.

Run: PYVWF_INPUT=input/combined PYTHONPATH=src /opt/anaconda3/bin/python -u \
         scripts/pinn/turbine_loro.py --cache <caches> --gwa <ratios> --out <dir> \
         --config CODE=PATH ...
     Add --report-only to recompute the summary and gates from the CSVs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

REGIONS = ["DE", "UK", "US", "BR", "AR", "AU-NEM", "CL", "DK", "NZ"]
GATED = ["DE", "UK", "US", "BR", "AR", "AU-NEM", "CL"]
CONTROL = "DK"
COUNTRY_FOLDS = ["FR", "BE", "ES", "IE", "IT", "NO", "SE", "PT", "NL"]
GATED_COUNTRY = ["FR", "BE", "ES", "IE", "IT", "NO"]
FLEET = ("log_capdens_50km", "is_offshore", "log_height")
ARMS = ["no-terrain", "relief-only", "full", "gwa-ratio", "in-region"]
#: Pool arms' switches: terrain_off, relief_off, fixed_speedup.
SWITCHES = {
    "no-terrain": (True, True, False),
    "relief-only": (True, False, False),
    "full": (False, False, False),
    "gwa-ratio": (True, True, True),
    "in-region": (False, False, False),
}


def margin(sd_a: float, sd_b: float) -> float:
    """The registered noise margin between two arms' seed means."""
    sds = [x for x in (sd_a, sd_b) if np.isfinite(x)]
    pooled = np.sqrt(np.mean([x ** 2 for x in sds])) if sds else 0.0
    return max(0.002, 2.0 * pooled)


def arm_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Per fold and arm: seed means and seed SDs of each metric."""
    return (raw.groupby(["fold", "arm"])
               .agg(spatial=("spatial_rmse", "mean"), spatial_sd=("spatial_rmse", "std"),
                    level=("level_rmse", "mean"), rmse=("rmse", "mean"),
                    rmse_sd=("rmse", "std"), mbe=("mbe", "mean"),
                    n_units=("n_units", "first"), seeds=("seed", "nunique"))
               .reset_index())


def gates(raw: pd.DataFrame, gated: list[str], control: str) -> dict:
    """T1 to T5 of the registration."""
    t = arm_table(raw).set_index(["fold", "arm"])

    def beats(f, better, worse, col="spatial"):
        a, b = t.loc[(f, better)], t.loc[(f, worse)]
        m = margin(a[f"{col}_sd"], b[f"{col}_sd"])
        return bool(b[col] - a[col] > m), float(b[col] - a[col]), m

    out: dict = {"folds": {}}
    counts = {"T1": 0, "T2": 0, "T5": 0, "T4_below": 0, "T4_worse10": 0}
    for f in gated:
        t1, d1, m1 = beats(f, "relief-only", "no-terrain")
        t2, d2, m2 = beats(f, "full", "relief-only")
        t5, d5, m5 = beats(f, "full", "gwa-ratio")
        unc, full = t.loc[(f, "uncorrected")], t.loc[(f, "full")]
        below = bool(full.rmse < unc.rmse)
        worse10 = bool(full.rmse > 1.10 * unc.rmse)
        counts["T1"] += t1
        counts["T2"] += t2
        counts["T5"] += t5
        counts["T4_below"] += below
        counts["T4_worse10"] += worse10
        out["folds"][f] = {"T1": [t1, d1, m1], "T2": [t2, d2, m2], "T5": [t5, d5, m5],
                           "T4_below": below, "T4_worse10": worse10}
    n = len(gated)
    out["T1"] = {"count": counts["T1"], "of": n, "pass": counts["T1"] >= 5}
    out["T2"] = {"count": counts["T2"], "of": n, "pass": counts["T2"] >= 5}
    out["T4"] = {"below": counts["T4_below"], "worse_by_10pct": counts["T4_worse10"], "of": n,
                 "pass": counts["T4_below"] >= 5 and counts["T4_worse10"] == 0}
    out["T5"] = {"count": counts["T5"], "of": n, "pass": counts["T5"] >= 5}
    if control in set(t.index.get_level_values(0)):
        base = t.loc[(control, "no-terrain")]
        checks = {}
        for arm in ("full", "relief-only"):
            a = t.loc[(control, arm)]
            m = margin(a.spatial_sd, base.spatial_sd)
            checks[arm] = [bool(abs(a.spatial - base.spatial) <= m),
                           float(a.spatial - base.spatial), m]
        out["T3"] = {"arms": checks, "pass": all(v[0] for v in checks.values())}
    return out


def report(out: Path, tag: str) -> None:
    raw = pd.read_csv(out / f"turbine_{tag}_raw.csv")
    pd.set_option("display.width", 220)
    t = arm_table(raw)
    for col, title in (("spatial", "Spatial RMSE"), ("level", "Level RMSE"),
                       ("rmse", "Per-unit RMSE")):
        print(f"\n### {title}, test year, mean over seeds\n")
        print(t.pivot(index="fold", columns="arm", values=col).round(5).to_string())
    g = gates(raw, [f for f in GATED if f in set(raw.fold)], CONTROL)
    for name in ("T1", "T2", "T3", "T4", "T5"):
        if name in g:
            print(f"\n{name}: {g[name]}")
    with open(out / f"turbine_{tag}_gates.json", "w", encoding="utf-8") as fh:
        json.dump(g, fh, indent=2)
        fh.write("\n")
    sec = out / f"turbine_{tag}_secondary.csv"
    if sec.exists():
        s = pd.read_csv(sec)
        print("\n### Secondary: national RMSE on country folds, mean over seeds\n")
        print(s.groupby(["fold", "arm"]).rmse.mean().unstack().round(5).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=REGIONS)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 42])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--config", action="append", default=[], metavar="CODE=PATH")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--gwa", required=True, help="directory written by gwa_ratio.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--no-secondary", action="store_true")
    ap.add_argument("--country-folds", nargs="+", default=COUNTRY_FOLDS)
    ap.add_argument("--registration", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if args.report_only:
        report(out, args.tag)
        return

    import torch

    from vwf.harness.provenance import build_manifest, write_manifest
    from vwf.harness.regions import load_region
    from vwf.harness.skill import collapse_pseudo_replicates, restrict_to_common_rows
    from vwf.pinn.runs import (
        UNIT_KEYS, config_record, level_spatial, region_record, resolve_configs,
        score_national_on_common_months,
    )
    from vwf.pinn.train import (
        attach_fixed_speedup, fit, load_regions, predict_frame, predict_national,
    )

    if build_manifest()["git_dirty"] and not args.allow_dirty:
        raise SystemExit("refusing to run on a dirty tree. Commit first, or pass --allow-dirty.")
    out.mkdir(parents=True, exist_ok=True)
    hidden = args.hidden or None
    secondary = not args.no_secondary
    codes = list(args.regions)
    extra = list(args.country_folds) if secondary else []
    config_paths = resolve_configs(codes + extra, args.config)
    specs = {c: load_region(config_paths[c]) for c in codes}
    for c, spec in specs.items():
        if spec.code != c or spec.obs_level != "turbine":
            raise SystemExit(f"{config_paths[c]} is {spec.code}/{spec.obs_level}, "
                             f"not a turbine-level {c}")

    def ratios(code, split):
        table = pd.read_csv(Path(args.gwa) / f"{code}_{split}.csv", dtype={"ID": str})
        return table.set_index("ID")["ratio"]

    train = {c: attach_fixed_speedup(load_regions([c], "train", args.cache, quiet=True)[0],
                                     ratios(c, "train")) for c in codes}
    test = {c: attach_fixed_speedup(load_regions([c], "test", args.cache, quiet=True)[0],
                                    ratios(c, "test")) for c in codes}
    countries = ({c: load_regions([c], "test", args.cache, quiet=True)[0] for c in extra}
                 if secondary else {})
    gwa_record = json.loads((Path(args.gwa) / "gwa_record.json").read_text(encoding="utf-8"))
    print("caches loaded:", {c: r.n_units for c, r in train.items()}, flush=True)

    write_manifest(out, build_manifest(extra={
        "run_mode": "pinn-turbine-loro", "tag": args.tag, "registration": args.registration,
        "argv": sys.argv[1:], "torch_version": torch.__version__, "cache": args.cache,
        "gwa": args.gwa, "gwa_record": gwa_record,
        "configs": config_record(config_paths),
        "settings": {"regions": codes, "gated": GATED, "control": CONTROL, "arms": args.arms,
                     "seeds": args.seeds, "epochs": args.epochs, "hidden": hidden,
                     "profile": "power", "density": False, "wake": False,
                     "bound_scale": 1.0, "fleet_columns": list(FLEET),
                     "secondary": secondary, "country_folds": extra},
        "regions": {c: {"train": region_record(train[c]), "test": region_record(test[c])}
                    for c in codes},
    }))

    def fitted(regions, arm, seed):
        terrain_off, relief_off, fixed = SWITCHES[arm]
        return fit(regions, hidden=hidden, physics=True, profile="power",
                   epochs=args.epochs, seed=seed, verbose=False, fleet_columns=FLEET,
                   terrain_off=terrain_off, relief_off=relief_off, fixed_speedup=fixed)

    rows, physics, records = [], [], {}
    for fold in codes:
        t0 = time.time()
        te, spec = test[fold], specs[fold]
        pool = [c for c in codes if c != fold]
        print(f"\n=== fold {fold} (pool {'+'.join(pool)}) ===", flush=True)
        conditions = {"uncorrected": ("uncorrected", -1, predict_frame(te, None, None))}
        for arm in args.arms:
            regions = [train[fold]] if arm == "in-region" else [train[c] for c in pool]
            for seed in args.seeds:
                model, std, hist = fitted(regions, arm, seed)
                conditions[f"{arm}/seed{seed}"] = (arm, seed, predict_frame(te, model, std))
                with torch.no_grad():
                    rep = model.report(std.terrain(te), std.fleet(te), std.relief(te), te.capdens)
                physics.append(dict(fold=fold, arm=arm, seed=seed, final_loss=hist[-1], **rep))

        pairs = {k: collapse_pseudo_replicates(frame, spec)
                 for k, (_, _, frame) in conditions.items()}
        restricted, excluded = restrict_to_common_rows(pairs, UNIT_KEYS, weight="capacity")
        for label, (arm, seed, _) in conditions.items():
            f = restricted[label]
            m = level_spatial(f)
            m["mbe"] = float(np.average(f.cf_sim - f.cf_obs, weights=f.capacity))
            rows.append(dict(fold=fold, arm=arm, seed=seed, **m))
        records[fold] = {"pool": pool, "rows_excluded": int(len(excluded))}
        raw = pd.DataFrame(rows)
        raw.to_csv(out / f"turbine_{args.tag}_raw.csv", index=False)
        pd.DataFrame(physics).to_csv(out / f"turbine_{args.tag}_physics.csv", index=False)
        with open(out / f"turbine_{args.tag}_record.json", "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2)
            fh.write("\n")
        sub = raw[raw.fold == fold].groupby("arm", sort=False)
        for arm, g in sub:
            print(f"  {arm:12s} spatial {g.spatial_rmse.mean():.5f}  level {g.level_rmse.mean():.5f}"
                  f"  per-unit {g.rmse.mean():.5f}", flush=True)
        print(f"  [{time.time() - t0:.0f}s]", flush=True)

    if secondary:
        sec = []
        print("\n=== secondary: all nine regions, applied to country folds ===", flush=True)
        for arm in ("no-terrain", "relief-only", "full"):
            for seed in args.seeds:
                model, std, _ = fitted([train[c] for c in codes], arm, seed)
                conds = {c: (arm, seed, predict_national(countries[c], model, std))
                         for c in extra}
                for c, (a, s, frame) in conds.items():
                    metrics, _, _ = score_national_on_common_months({"x": (a, s, frame)})
                    sec.append(dict(fold=c, arm=arm, seed=seed, **metrics["x"]))
        for c in extra:
            frame = predict_national(countries[c], None, None)
            metrics, _, _ = score_national_on_common_months({"x": ("uncorrected", -1, frame)})
            sec.append(dict(fold=c, arm="uncorrected", seed=-1, **metrics["x"]))
        pd.DataFrame(sec).to_csv(out / f"turbine_{args.tag}_secondary.csv", index=False)

    report(out, args.tag)


if __name__ == "__main__":
    main()
