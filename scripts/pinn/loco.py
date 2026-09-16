#!/usr/bin/env python3
"""Leave one country out: does the physics-informed model transfer to countries
it was never trained on?

The registered design is ``docs/findings/method-physics-informed-loco-prereg.md``.
Each fold holds one country out. Every arm is scored on the fold's test year as
the national monthly capacity factor, the quantity a country-level scorecard
row reports, and turbine-level folds are also scored per unit on common rows.

Arms, all fitted with the same settings:

  uncorrected   ERA5 through the power curves with no correction.
  transfer      fitted on the pool, never on the fold.
  features-off  the same pool with every head's inputs set to zero, so each
                learned quantity is one global constant (the speed-up keeps its
                relief pin). If this matches ``transfer``, what transfers is
                constants, not anything tied to place.
  in-country    fitted on the fold's own training years. Not a transfer arm:
                the ceiling the transfer arms are read against.

Flagged folds are scored like the others and are never in any pool.

Run: PYVWF_INPUT=input/combined PYTHONPATH=src /opt/anaconda3/bin/python -u \
         scripts/pinn/loco.py --cache <cache root> --out <dir> --config CODE=PATH ...
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

GATED = ["DK", "DE", "UK", "FR", "BE", "ES", "IE", "IT", "NO"]
FLAGGED = ["SE", "PT", "NL"]
LOCO_FLEET = ("log_capdens_50km", "is_offshore", "log_height")
ARMS = ["transfer", "features-off", "in-country"]


def recovery_ratio(mse_unc: float, mse_arm: float, mse_in: float) -> float:
    """Share of the in-country arm's error reduction a transfer arm recovers.

    ``(MSE_uncorrected - MSE_arm) / (MSE_uncorrected - MSE_in-country)``.
    Undefined, returned as NaN, when the in-country arm does not beat
    uncorrected: there is then no reduction to recover.
    """
    if not (mse_in < mse_unc):
        return float("nan")
    return (mse_unc - mse_arm) / (mse_unc - mse_in)


def fold_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Per fold and arm: mean national RMSE, seed SD, and mean squared error."""
    nat = raw[raw["scope"] == "national"].copy()
    nat["mse"] = nat["rmse"] ** 2
    return (nat.groupby(["fold", "arm"])
               .agg(rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
                    mse=("mse", "mean"), mbe=("mbe", "mean"),
                    r=("pearson_r", "mean"), n_months=("n_months", "first"),
                    seeds=("seed", "nunique"))
               .reset_index())


def gates(raw: pd.DataFrame, gated: list[str]) -> dict:
    """G1 to G3 of the registration, computed on the gated folds only."""
    t = fold_table(raw).set_index(["fold", "arm"])
    out: dict = {"folds": {}}
    below = worse10 = recovered = defined = beats = 0
    for f in gated:
        unc = t.loc[(f, "uncorrected")]
        tr = t.loc[(f, "transfer")]
        off = t.loc[(f, "features-off")]
        inc = t.loc[(f, "in-country")]
        ratio = recovery_ratio(unc.mse, tr.mse, inc.mse)
        sd = float(np.sqrt(np.nanmean([tr.rmse_sd ** 2, off.rmse_sd ** 2])))
        margin = max(0.002, 2.0 * sd)
        row = {
            "uncorrected": float(unc.rmse), "transfer": float(tr.rmse),
            "features_off": float(off.rmse), "in_country": float(inc.rmse),
            "transfer_below_uncorrected": bool(tr.rmse < unc.rmse),
            "transfer_worse_by_10pct": bool(tr.rmse > 1.10 * unc.rmse),
            "recovery_ratio": ratio,
            "features_off_minus_transfer": float(off.rmse - tr.rmse),
            "g3_margin": margin,
            "transfer_beats_features_off": bool(off.rmse - tr.rmse > margin),
        }
        out["folds"][f] = row
        below += row["transfer_below_uncorrected"]
        worse10 += row["transfer_worse_by_10pct"]
        if np.isfinite(ratio):
            defined += 1
            recovered += ratio >= 0.5
        beats += row["transfer_beats_features_off"]
    n = len(gated)
    out["G1"] = {"below": below, "worse_by_10pct": worse10, "of": n,
                 "pass": bool(below >= 7 and worse10 == 0)}
    out["G2"] = {"defined": defined, "recovered_at_least_half": int(recovered),
                 "readable": bool(defined >= 5),
                 "pass": bool(defined >= 5 and recovered > defined / 2)}
    out["G3"] = {"beats": beats, "of": n, "pass": bool(beats >= 5)}
    return out


def report(out: Path, tag: str, gated: list[str]) -> None:
    raw = pd.read_csv(out / f"loco_{tag}_raw.csv")
    pd.set_option("display.width", 220)
    table = fold_table(raw)
    print("\n### National monthly RMSE on each fold's test year, mean over seeds\n")
    print(table.pivot(index="fold", columns="arm", values="rmse").round(5).to_string())
    units = raw[raw["scope"] == "units"]
    if len(units):
        print("\n### Per-unit RMSE, turbine-level folds, common rows\n")
        print(units.groupby(["fold", "arm"]).rmse.mean().unstack().round(5).to_string())
    g = gates(raw, [f for f in gated if f in set(table["fold"])])
    for name in ("G1", "G2", "G3"):
        print(f"\n{name}: {g[name]}")
    with open(out / f"loco_{tag}_gates.json", "w", encoding="utf-8") as fh:
        json.dump(g, fh, indent=2)
        fh.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=GATED + FLAGGED)
    ap.add_argument("--pool", nargs="+", default=GATED,
                    help="countries every fold trains on, minus the fold itself")
    ap.add_argument("--extra-pool", nargs="+", default=[],
                    help="regions added to every fold's pool (the world study)")
    ap.add_argument("--flagged", nargs="+", default=FLAGGED,
                    help="folds scored but never trained on, and outside the gates")
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 42])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=0, help="0 = linear heads")
    ap.add_argument("--fleet-columns", nargs="+", default=list(LOCO_FLEET))
    ap.add_argument("--config", action="append", default=[], metavar="CODE=PATH")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="europe")
    ap.add_argument("--registration", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="recompute the summary and gates from an existing run")
    args = ap.parse_args()
    out = Path(args.out)
    gated = [f for f in args.folds if f not in args.flagged]

    if args.report_only:
        report(out, args.tag, gated)
        return

    import torch

    from vwf.harness.provenance import build_manifest, write_manifest
    from vwf.harness.regions import load_region
    from vwf.pinn.runs import (
        config_record, region_record, resolve_configs, score_national_on_common_months,
        score_on_common_rows,
    )
    from vwf.pinn.train import fit, load_regions, off_curve_shares, predict_frame, predict_national

    if build_manifest()["git_dirty"] and not args.allow_dirty:
        raise SystemExit("refusing to run on a dirty tree: every result would be "
                         "unattributable. Commit first, or pass --allow-dirty.")
    overlap = sorted(set(args.flagged) & set(args.pool + args.extra_pool))
    if overlap:
        raise SystemExit(f"flagged folds may not be in any pool: {overlap}")

    out.mkdir(parents=True, exist_ok=True)
    hidden = args.hidden or None
    fleet_columns = tuple(args.fleet_columns)
    codes = list(dict.fromkeys([*args.pool, *args.extra_pool, *args.folds]))
    config_paths = resolve_configs(codes, args.config)
    specs = {c: load_region(config_paths[c]) for c in codes}
    for c, spec in specs.items():
        if spec.code != c:
            raise SystemExit(f"{config_paths[c]} is region {spec.code}, not {c}")
    train_sets = {c: load_regions([c], "train", args.cache, quiet=True)[0] for c in codes}
    test_sets = {c: load_regions([c], "test", args.cache, quiet=True)[0] for c in args.folds}
    print("caches loaded:", {c: f"{r.level[0]}{r.n_units}" for c, r in train_sets.items()},
          flush=True)

    pools = {f: [c for c in [*args.pool, *args.extra_pool] if c != f] for f in args.folds}
    write_manifest(out, build_manifest(extra={
        "run_mode": "pinn-loco",
        "tag": args.tag,
        "registration": args.registration,
        "argv": sys.argv[1:],
        "torch_version": torch.__version__,
        "cache": str(args.cache),
        "configs": config_record(config_paths),
        "settings": {
            "folds": args.folds, "gated": gated, "flagged": args.flagged,
            "pool": args.pool, "extra_pool": args.extra_pool, "pools_by_fold": pools,
            "arms": args.arms, "seeds": args.seeds, "epochs": args.epochs,
            "hidden": hidden, "profile": "power", "density": False, "wake": False,
            "bound_scale": 1.0, "fleet_columns": list(fleet_columns),
        },
        "regions": {
            c: {"train": region_record(train_sets[c]),
                **({"test": region_record(test_sets[c])} if c in test_sets else {})}
            for c in codes
        },
    }))

    rows, predictions, physics, records = [], [], [], {}
    for fold in args.folds:
        t0 = time.time()
        te, spec = test_sets[fold], specs[fold]
        print(f"\n=== fold {fold} ({te.level}; pool {'+'.join(pools[fold])}) ===", flush=True)
        national: dict = {}
        units: dict = {}
        off_curve: dict = {}

        tally: dict = {}
        national["uncorrected"] = ("uncorrected", -1, predict_national(te, None, None, off_curve=tally))
        off_curve["uncorrected"] = off_curve_shares(tally)
        if te.level == "turbine":
            units["uncorrected"] = ("uncorrected", -1, predict_frame(te, None, None))

        arm_specs = {
            "transfer": (pools[fold], False),
            "features-off": (pools[fold], True),
            "in-country": ([fold], False),
        }
        for arm in args.arms:
            train_codes, features_off = arm_specs[arm]
            for seed in args.seeds:
                model, std, hist = fit([train_sets[c] for c in train_codes], hidden=hidden,
                                       physics=True, profile="power", epochs=args.epochs,
                                       seed=seed, verbose=False, fleet_columns=fleet_columns,
                                       features_off=features_off)
                label = f"{arm}/seed{seed}"
                tally = {}
                national[label] = (arm, seed, predict_national(te, model, std, off_curve=tally))
                off_curve[label] = off_curve_shares(tally)
                if te.level == "turbine":
                    units[label] = (arm, seed, predict_frame(te, model, std))
                with torch.no_grad():
                    rep = model.report(std.terrain(te), std.fleet(te), te.relief, te.capdens)
                physics.append(dict(fold=fold, arm=arm, seed=seed, final_loss=hist[-1], **rep))

        metrics, excluded, summary = score_national_on_common_months(national)
        for label, (arm, seed, frame) in national.items():
            rows.append(dict(fold=fold, level=te.level, scope="national", arm=arm, seed=seed,
                             **metrics[label], **off_curve[label]))
            predictions.append(frame.assign(fold=fold, arm=arm, seed=seed))
        record = {"national_common_months": summary,
                  "national_months_excluded": excluded.to_dict("records"),
                  "off_curve": off_curve, "pool": pools[fold]}
        if units:
            u_metrics, u_excluded, u_summary = score_on_common_rows(units, spec)
            for label, (arm, seed, _) in units.items():
                rows.append(dict(fold=fold, level=te.level, scope="units", arm=arm, seed=seed,
                                 **u_metrics[label]))
            record["unit_common_rows"] = u_summary
        records[fold] = record
        del national, units

        raw = pd.DataFrame(rows)
        raw.to_csv(out / f"loco_{args.tag}_raw.csv", index=False)
        pd.concat(predictions, ignore_index=True).to_csv(
            out / f"loco_{args.tag}_national.csv", index=False)
        pd.DataFrame(physics).to_csv(out / f"loco_{args.tag}_physics.csv", index=False)
        with open(out / f"loco_{args.tag}_record.json", "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, default=str)
            fh.write("\n")
        nat = raw[(raw.fold == fold) & (raw.scope == "national")]
        for arm, g in nat.groupby("arm", sort=False):
            print(f"  {arm:13s} national RMSE {g.rmse.mean():.5f} +- {g.rmse.std() if len(g) > 1 else 0:.5f}"
                  f"  MBE {g.mbe.mean():+.5f}  months {int(g.n_months.iloc[0])}", flush=True)
        print(f"  [{time.time() - t0:.0f}s]", flush=True)

    if set(args.arms) == set(ARMS):
        report(out, args.tag, gated)


if __name__ == "__main__":
    main()
