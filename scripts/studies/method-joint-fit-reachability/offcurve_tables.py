"""Control, then the off-curve comparison, for the reachability follow-up.

Joins each row's ``reachability.json`` and ``reachability_offcurve.json``,
checks the follow-up's fits are the pass's, classifies every period against
the range with off-curve values dropped and counted as zero (thresholds as
registered), prints the tables and writes ``reachability_offcurve_periods.csv``.

Usage, from the repository root:

    PYTHONPATH=src python scripts/studies/method-joint-fit-reachability/offcurve_tables.py \
        <out_dir>
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vwf.cli.common import make_parser

UNREACHABLE, NEAR = 1e-6, 0.05


def classify(obs, lo, hi):
    if obs - hi > UNREACHABLE:
        return "unreachable above"
    if lo - obs > UNREACHABLE:
        return "unreachable below"
    if min(hi - obs, obs - lo) < NEAR:
        return "near-miss"
    return "reachable"


def main(out_dir: str) -> None:
    OUT = Path(out_dir)
    rows, control = [], {}
    for code in ["FR", "BE", "IE", "SE", "NO", "ES", "IT", "PT"]:
        a = json.loads((OUT / code / "train-reach" / "reachability.json").read_text())
        b = json.loads((OUT / code / "train-offcurve" / "reachability_offcurve.json").read_text())
        key = lambda r: (r["year"], r["time_slice"], r["n_clusters"])  # noqa: E731
        bb = {key(r): r for r in b}
        bad = 0
        for r in a:
            z = bb[key(r)]
            if not (
                z["success"] == r["success"]
                and z["fun"] == r["fun"]
                and z["offsets"] == r["offsets"]
            ):
                bad += 1
            outcome = (
                "refused abnormal"
                if not r["success"]
                else ("refused on a bound" if r["n_at_bound"] else "accepted")
            )
            lo_share = [c["lo_off_curve"] for c in z["per_cluster_zero"].values()]
            lo_at = [c["lo_offset"] for c in z["per_cluster_zero"].values()]
            rows.append(
                dict(
                    region=code,
                    n_clusters=r["n_clusters"],
                    year=r["year"],
                    time_slice=r["time_slice"],
                    outcome=outcome,
                    obs=r["obs"],
                    lo_drop=r["lo"],
                    hi_drop=r["hi"],
                    lo_zero=z["lo_zero"],
                    hi_zero=z["hi_zero"],
                    cls_drop=classify(r["obs"], r["lo"], r["hi"]),
                    cls_zero=classify(z["obs"], z["lo_zero"], z["hi_zero"]),
                    lo_off_curve_max=max(lo_share),
                    lo_offset_min=min(lo_at),
                )
            )
        control[code] = f"{len(a)} periods, {bad} fits differ from the pass"
    print("CONTROL")
    for k, v in control.items():
        print(f"  {k}: {v}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "reachability_offcurve_periods.csv", index=False)
    orders = ["reachable", "near-miss", "unreachable above", "unreachable below"]
    outs = ["accepted", "refused on a bound", "refused abnormal"]
    print("\nZERO-COUNTED RANGE: class x outcome, all rows")
    print(
        pd.crosstab(df["cls_zero"], df["outcome"])
        .reindex(index=orders, columns=outs, fill_value=0)
        .to_string()
    )
    u = df[(df["cls_drop"] == "unreachable below") & (df["outcome"] != "accepted")]
    print(f"\nTHE {len(u)} UNREACHABLE REFUSALS, classed with off-curve values as zero:")
    print(u["cls_zero"].value_counts().reindex(orders, fill_value=0).to_string())
    print(u.groupby("region")["cls_zero"].value_counts().unstack(fill_value=0).to_string())
    print("\nFLOORS (national lowest CF), dropped vs zero, per row")
    print(
        df.groupby("region")[["lo_drop", "lo_zero", "hi_drop", "hi_zero"]]
        .median()
        .round(4)
        .to_string()
    )
    print(
        "\nOff-curve share at each cluster's lowest-output offset (max over clusters), median by row"
    )
    print(df.groupby("region")["lo_off_curve_max"].median().round(3).to_string())
    print("\nclass changes across all periods:")
    print(pd.crosstab(df["cls_drop"], df["cls_zero"]).to_string())


def cli(argv: list[str] | None = None) -> None:
    """Parse the recorded command line, ``<out_dir>``, and run :func:`main`."""
    parser = make_parser(__doc__)
    parser.add_argument("out_dir", help="The pass's output directory")
    args = parser.parse_args(argv)
    main(args.out_dir)


if __name__ == "__main__":
    cli()
