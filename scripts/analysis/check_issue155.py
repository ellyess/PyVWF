"""Issue #155: does scaling lon/lat over the held-out region move the RF transfer results?

The random-forest transfer analyses min-max scale longitude and latitude over
every region in the pool before each leave-one-region-out fold is split, so
the held-out region helps set the span its own features are scaled by. This
recomputes every fold twice on the same inputs. "Published" scales over the
pooled frame exactly as the drivers do. "Train" recomputes the span from the
training regions only and applies it to both the training and the held-out
rows. Nothing else differs. The gate, registered in the issue before the first
run: every compared value agrees within 1e-9.

Parts, one per process:

- ``r1``: scripts/analysis/ml_transfer_retest.py, every suite, feature set and
  target. It first reruns that script and checks its digests against the
  realdata pin of tests/test_pin_ml_transfer.py.
- ``r2``: scripts/analysis/ml_transfer_expanded.py, both pools, uncapped and
  capped. It imports that script without calling its ``main``, which would
  overwrite output/ml_retest/.
- ``pinn``: the physics-informed study's RF baseline
  (``rf_affine_predictions`` in scripts/studies/physics-informed/e1_loro.py),
  per test unit of the 2026-09-16 rerun's fleets.

Run, from the repository root:

    python scripts/dev/run_locked.py -- python scripts/analysis/check_issue155.py r1 output/issue155_check_2026-10-27
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

import ml_transfer_expanded as r2
import ml_transfer_retest as r1
from pyvwf.cli.common import make_parser
from pyvwf.extensions.ml import transfer as T

ROOT = Path(__file__).resolve().parents[2]
TOL = 1e-9
SEEDS = list(T.SEEDS)
PINN_CACHE = ROOT / "output" / "pinn_rerun_2026-09-16" / "cache"
PINN_SEEDS = [0, 1, 2, 3, 42]  # --seeds of the E1 runs (run_manifest.json settings.seeds)
# As pinned by tests/test_pin_ml_transfer.py (REAL_STDOUT_SHA256, REAL_CENTROIDS_SHA256).
PINNED_STDOUT = {
    "1.7.2": "5453fed60ea4ad2414c0b6ea10c26f4d710c8cdf0eb541e0ad07b29ef747f1f8",
    "1.9.1": "08a27e8202d3d2abc78e1d6b28eb71dbc6e09cde44e8f4fefce25b8e382a06db",
}
PINNED_CENTROIDS = "d2c897729da1d59ad597883de95c7bed14f77426e5aef30af59983cfd66c2bea"


def environment() -> dict:
    """The commit, tree state and library versions the comparison ran under."""

    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "status_porcelain": git("status", "--porcelain"),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "seeds": SEEDS,
        "tolerance": TOL,
    }


def pooled_norm(df: pd.DataFrame) -> pd.DataFrame:
    """The drivers' scaling, verbatim: span over the whole frame."""
    df = df.copy()
    for c in ("lon", "lat"):
        df[f"{c}_norm"] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
    return df


def renorm(frame: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Scale ``frame``'s lon/lat by ``ref``'s span."""
    frame = frame.copy()
    for c in ("lon", "lat"):
        lo, hi = ref[c].min(), ref[c].max()
        frame[f"{c}_norm"] = (frame[c] - lo) / (hi - lo)
    return frame


def span_shift(pooled: pd.DataFrame, train: pd.DataFrame) -> dict:
    """How far the training rows' scaled values move between the two spans."""
    return {
        f"max_abs_shift_{c}_norm_train_rows": float(
            (pooled[f"{c}_norm"] - renorm(train, train)[f"{c}_norm"]).abs().max()
        )
        for c in ("lon", "lat")
    }


def loro_records(df, feats, target, mode):
    """Per-seed leave-one-region-out scores under one scaling, and each fold's span shift."""
    recs, shifts = [], []
    for region in df.region.unique():
        tr, te = df[df.region != region], df[df.region == region]
        if mode == "train":
            shifts.append({"holdout": region, **span_shift(tr, tr)})
            tr, te = renorm(tr, tr), renorm(te, tr)
        for s in SEEDS:
            r2_, mae = T.rf_eval(tr, te, feats, target, s)
            recs.append(dict(holdout=region, seed=s, r2=r2_, mae=mae))
    return pd.DataFrame(recs), pd.DataFrame(shifts)


def aggregate(recs):
    """Seed means per held-out region, as :func:`pyvwf.extensions.ml.transfer.loro` reports."""
    return (
        recs.groupby("holdout")
        .agg(r2_mean=("r2", "mean"), r2_std=("r2", "std"), mae_mean=("mae", "mean"))
        .reset_index()
    )


def feats_name(feats):
    """The feature set's name in :mod:`pyvwf.extensions.ml.transfer`."""
    for name in ("SET_A", "SET_B", "SET_C"):
        if list(feats) == list(getattr(T, name)):
            return name
    return "+".join(feats)


def compare_loro(label, df, feats, target):
    """Both scalings for one suite, feature set and target: one summary row and the tables."""
    pub, _ = loro_records(df, feats, target, "published")
    # The harness must reproduce the library path exactly, or the comparison
    # would not be of the published computation.
    pd.testing.assert_frame_equal(aggregate(pub), T.loro(df, feats, target), rtol=0, atol=0)
    trn, shifts = loro_records(df, feats, target, "train")
    m = pub.merge(trn, on=["holdout", "seed"], suffixes=("_pub", "_trn"))
    ap, at = aggregate(pub), aggregate(trn)
    am = ap.merge(at, on="holdout", suffixes=("_pub", "_trn"))
    row = dict(
        suite=label,
        features=feats_name(feats),
        target=target,
        folds=int(df.region.nunique()),
        per_seed_values=int(len(m)),
        max_abs_d_r2=float((m.r2_pub - m.r2_trn).abs().max()),
        max_abs_d_mae=float((m.mae_pub - m.mae_trn).abs().max()),
        max_abs_d_r2_mean=float((am.r2_mean_pub - am.r2_mean_trn).abs().max()),
        max_abs_d_mae_mean=float((am.mae_mean_pub - am.mae_mean_trn).abs().max()),
        n_r2_pos_pub=int((ap.r2_mean > 0).sum()),
        n_r2_pos_trn=int((at.r2_mean > 0).sum()),
        max_span_shift=float(
            shifts.drop(columns="holdout").to_numpy().max() if len(shifts) else 0.0
        ),
    )
    row["within_tol"] = bool(max(row["max_abs_d_r2"], row["max_abs_d_mae"]) <= TOL)
    return row, am, shifts.assign(suite=label, features=feats_name(feats), target=target)


def seed_control(df, feats, target):
    """Positive control: the comparison must see two seeds as different."""
    region = df.region.unique()[0]
    tr, te = df[df.region != region], df[df.region == region]
    a = T.rf_eval(tr, te, feats, target, SEEDS[0])
    b = T.rf_eval(tr, te, feats, target, SEEDS[1])
    return float(max(abs(a[0] - b[0]), abs(a[1] - b[1])))


def random_cv_compare(df, feats, target):
    """Supplementary, outside the gate: random 5-fold CV with the span from the training fold."""
    d = []
    for s in SEEDS:
        kf = KFold(n_splits=5, shuffle=True, random_state=s)
        for tr_i, te_i in kf.split(df):
            tr, te = df.iloc[tr_i], df.iloc[te_i]
            p = T.rf_eval(tr, te, feats, target, s)
            q = T.rf_eval(renorm(tr, tr), renorm(te, tr), feats, target, s)
            d.append(max(abs(p[0] - q[0]), abs(p[1] - q[1])))
    return float(max(d))


def write(out: Path, summary, tables):
    """Write one part's tables and summary, and print the summary."""
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def part_r1(out: Path):
    """Round one: reproduce the published run, then compare every fold."""
    t0 = time.time()
    repro = out / "repro"
    repro.mkdir(parents=True, exist_ok=True)
    r1.OUT = repro
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r1.main()
    stdout_sha = hashlib.sha256(buf.getvalue().encode()).hexdigest()
    cent_sha = hashlib.sha256((repro / "ml_retest_centroids_primary.csv").read_bytes()).hexdigest()
    (repro / "stdout.txt").write_text(buf.getvalue(), encoding="utf-8")

    suites = {"PRIMARY": r1.RUNS}
    for name, override in r1.SENSITIVITY_RUNS.items():
        suites[f"SENSITIVITY {name}"] = {**r1.RUNS, **override}

    rows, means, shifts, controls, cv = [], [], [], {}, {}
    for label, runs in suites.items():
        df = r1.terrain_features(pooled_norm(r1.build_centroids(runs)))
        for feats in (T.SET_A, T.SET_B, T.SET_C):
            for target in ("scalar", "offset"):
                row, am, sh = compare_loro(label, df, feats, target)
                rows.append(row)
                means.append(am.assign(suite=label, features=feats_name(feats), target=target))
                shifts.append(sh)
                print(row, flush=True)
        controls[label] = seed_control(df, T.SET_A, "scalar")
        if label == "PRIMARY":
            for target in ("scalar", "offset"):
                cv[target] = random_cv_compare(df, T.SET_A, target)

    table = pd.DataFrame(rows)
    gate_rows = table[table.features != "SET_B"]
    summary = dict(
        part="r1: scripts/analysis/ml_transfer_retest.py",
        environment=environment(),
        reproduction=dict(
            stdout_sha256=stdout_sha,
            pinned_stdout_sha256=PINNED_STDOUT.get(sklearn.__version__),
            stdout_matches_pin=stdout_sha == PINNED_STDOUT.get(sklearn.__version__),
            centroids_sha256=cent_sha,
            centroids_match_pin=cent_sha == PINNED_CENTROIDS,
        ),
        gate_tolerance=TOL,
        gate_pass=bool(table.within_tol.all()),
        gate_rows=int(len(table)),
        max_abs_d_any=float(table[["max_abs_d_r2", "max_abs_d_mae"]].to_numpy().max()),
        negative_control_set_b_exactly_equal=bool(
            (table[table.features == "SET_B"][["max_abs_d_r2", "max_abs_d_mae"]] == 0).all().all()
        ),
        positive_control_seed0_vs_seed1=controls,
        positive_control_fires=bool(min(controls.values()) > TOL),
        position_sets_span_shift_min=float(gate_rows.max_span_shift.min()),
        supplementary_random_cv_max_abs_d=cv,
        seconds=round(time.time() - t0, 1),
    )
    write(
        out,
        summary,
        {
            "comparison": table,
            "per_region_means": pd.concat(means),
            "span_shifts": pd.concat(shifts),
        },
    )


def part_r2(out: Path):
    """Round two: compare every fold of both pools, uncapped and capped at 3.0."""
    t0 = time.time()
    rows, means, shifts, controls, repro = [], [], [], {}, {}
    published = {
        None: ROOT / "output" / "ml_retest" / "expanded_loro_scalar.csv",
        3.0: ROOT / "output" / "ml_retest" / "expanded_loro_scalar_capped.csv",
    }
    for runs_name, runs in (("RUNS_5", r2.RUNS_5), ("RUNS_8", r2.RUNS_8)):
        base = r2.terrain_features(pooled_norm(r2.build_centroids(runs)))
        for cap in (None, 3.0):
            df = base.copy()
            if cap is not None:
                df["scalar"] = df["scalar"].clip(0.3, cap)
            label = f"{runs_name}{'' if cap is None else ' capped 3.0'}"
            row, am, sh = compare_loro(label, df, T.SET_A, "scalar")
            rows.append(row)
            means.append(am.assign(suite=label))
            shifts.append(sh)
            controls[label] = seed_control(df, T.SET_A, "scalar")
            print(row, flush=True)
            if runs_name != "RUNS_8":
                continue
            if not published[cap].exists():
                repro[label] = "no published file on this machine"
                continue
            got = am[["holdout", "r2_mean_pub", "r2_std_pub", "mae_mean_pub"]].rename(
                columns=lambda c: c.replace("_pub", "")
            )
            j = pd.read_csv(published[cap]).merge(got, on="holdout", suffixes=("_file", "_rerun"))
            repro[label] = float(
                max(
                    (j[f"{c}_file"] - j[f"{c}_rerun"]).abs().max()
                    for c in ("r2_mean", "r2_std", "mae_mean")
                )
            )

    table = pd.DataFrame(rows)
    summary = dict(
        part="r2: scripts/analysis/ml_transfer_expanded.py",
        environment=environment(),
        reproduction_vs_output_ml_retest_max_abs_d=repro,
        gate_tolerance=TOL,
        gate_pass=bool(table.within_tol.all()),
        gate_rows=int(len(table)),
        max_abs_d_any=float(table[["max_abs_d_r2", "max_abs_d_mae"]].to_numpy().max()),
        positive_control_seed0_vs_seed1=controls,
        positive_control_fires=bool(min(controls.values()) > TOL),
        span_shift_min=float(table.max_span_shift.min()),
        seconds=round(time.time() - t0, 1),
    )
    write(
        out,
        summary,
        {
            "comparison": table,
            "per_region_means": pd.concat(means),
            "span_shifts": pd.concat(shifts),
        },
    )


def part_pinn(out: Path):
    """The physics-informed RF baseline: compare every unit's predicted scalar and offset."""
    t0 = time.time()
    centroids = r1.terrain_features(pooled_norm(r1.build_centroids(r1.RUNS)))
    rows, per_unit, controls = [], [], {}
    for holdout in ["DK", "DE", "UK", "US", "BR"]:
        meta = pd.read_csv(PINN_CACHE / f"{holdout}_test" / "meta.csv", dtype={"ID": str})
        units = meta[["ID", "lon", "lat"]].copy()
        units["region"] = holdout
        units["abs_lat"] = units["lat"].abs()
        units = r1.terrain_features(units)
        train = centroids[centroids.region != holdout]
        # Published: train rows keep the pooled scaling; units use the pooled
        # centroid span (e1_loro.py:150-155, prep_rf_features.py:41-62).
        u_pub = renorm(units, centroids)
        t_trn, u_trn = renorm(train, train), renorm(units, train)
        preds = {}
        for mode, tr, u in (("pub", train, u_pub), ("trn", t_trn, u_trn)):
            for target in ("scalar", "offset"):
                preds[(mode, target)] = np.array(
                    [
                        RandomForestRegressor(random_state=s, **r1.RF_KW)
                        .fit(tr[r1.SET_A], tr[target])
                        .predict(u[r1.SET_A])
                        for s in PINN_SEEDS
                    ]
                )
        row = dict(holdout=holdout, units=int(len(units)))
        for target in ("scalar", "offset"):
            a, b = preds[("pub", target)], preds[("trn", target)]
            row[f"max_abs_d_{target}_per_seed"] = float(np.abs(a - b).max())
            row[f"max_abs_d_{target}_seed_mean"] = float(np.abs(a.mean(0) - b.mean(0)).max())
            row[f"n_exact_{target}_seed_mean"] = int((a.mean(0) == b.mean(0)).sum())
            per_unit.append(
                pd.DataFrame(
                    {
                        "holdout": holdout,
                        "ID": units["ID"].to_numpy(),
                        "target": target,
                        "published": a.mean(0),
                        "train_span": b.mean(0),
                    }
                )
            )
        row.update(span_shift(train, train))
        row["unit_lon_norm_shift_max"] = float((u_pub.lon_norm - u_trn.lon_norm).abs().max())
        row["unit_lat_norm_shift_max"] = float((u_pub.lat_norm - u_trn.lat_norm).abs().max())
        row["within_tol"] = bool(
            max(row["max_abs_d_scalar_per_seed"], row["max_abs_d_offset_per_seed"]) <= TOL
        )
        controls[holdout] = float(
            np.abs(preds[("pub", "scalar")][0] - preds[("pub", "scalar")][1]).max()
        )
        rows.append(row)
        print(row, flush=True)

    table = pd.DataFrame(rows)
    summary = dict(
        part="pinn: physics-informed RF baseline (rf_affine_predictions, e1_loro.py:160-176)",
        environment=environment(),
        test_fleets=str(PINN_CACHE.relative_to(ROOT)),
        published_predictions_on_disk=False,
        gate_tolerance=TOL,
        gate_pass=bool(table.within_tol.all()),
        max_abs_d_any=float(
            table[["max_abs_d_scalar_per_seed", "max_abs_d_offset_per_seed"]].to_numpy().max()
        ),
        positive_control_seed0_vs_seed1_scalar=controls,
        positive_control_fires=bool(min(controls.values()) > TOL),
        seconds=round(time.time() - t0, 1),
    )
    write(out, summary, {"comparison": table, "per_unit_seed_mean": pd.concat(per_unit)})


PARTS = {"r1": part_r1, "r2": part_r2, "pinn": part_pinn}


def main(part: str, out_dir: str) -> None:
    """Run one part, writing under ``out_dir/<part>/``."""
    PARTS[part](ROOT / out_dir / part)


def cli(argv: list[str] | None = None) -> None:
    """Parse the recorded command line, ``<part> <out_dir>``, and run :func:`main`."""
    parser = make_parser(__doc__)
    parser.add_argument("part", choices=sorted(PARTS), help="Which analysis to check")
    parser.add_argument("out_dir", help="Directory for the outputs, under output/")
    args = parser.parse_args(argv)
    main(args.part, args.out_dir)


if __name__ == "__main__":
    cli()
