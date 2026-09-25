"""Control, then classification, for the joint-fit reachability pass.

Reads each row's ``reachability.json`` (``reachability_pass.py``), checks every
period's fit against the licensed-curve record in
``output/country_curves_2026-09-25/diag/``, classifies each period with the
thresholds registered in ``docs/findings/method-joint-fit-reachability-prereg.md``
(unreachable beyond 1e-6 CF, near-miss within 0.05 CF), prints the tables and
writes ``reachability_periods.csv``.

Usage, from the repository root:

    PYTHONPATH=src python scripts/studies/method-joint-fit-reachability/reachability_tables.py \
        <out_dir>
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from pyvwf.cli.common import make_parser

UNREACHABLE = 1e-6
NEAR = 0.05
RECORD = Path("output/country_curves_2026-09-25/diag")


def key(r):
    return (r["year"], r["time_slice"], r["n_clusters"])


def main(out_dir: str) -> None:
    OUT = Path(out_dir)
    rows = []
    control = {}
    for code in ["FR", "BE", "IE", "SE", "NO", "ES", "IT", "PT"]:
        path = OUT / code / "train-reach" / "reachability.json"
        if not path.is_file():
            control[code] = "missing"
            continue
        got = json.loads(path.read_text())
        ref = json.loads((RECORD / code / "train-diag" / "c2_minimize_calls.json").read_text())
        ref_by = {key(r): r for r in ref}
        mismatch = []
        for r in got:
            e = ref_by.get(key(r))
            if e is None:
                mismatch.append((key(r), "not in record"))
                continue
            same = (
                r.get("success") == e["success"]
                and r.get("n_at_bound") == e["n_at_bound"]
                and r.get("offsets") == e["offsets"]
                and math.isclose(r.get("fun", float("nan")), e["fun"], rel_tol=0, abs_tol=0)
            )
            if not same:
                mismatch.append((key(r), "differs"))
        control[code] = f"{len(got)} periods, {len(ref)} recorded, {len(mismatch)} mismatched" + (
            f": {mismatch[:3]}" if mismatch else ""
        )
        for r in got:
            above = r["obs"] - r["hi"]
            below = r["lo"] - r["obs"]
            if above > UNREACHABLE:
                cls = "unreachable above"
            elif below > UNREACHABLE:
                cls = "unreachable below"
            elif min(r["hi"] - r["obs"], r["obs"] - r["lo"]) < NEAR:
                cls = "near-miss"
            else:
                cls = "reachable"
            if not r.get("success", False):
                outcome = "refused abnormal"
            elif r.get("n_at_bound", 0) > 0:
                outcome = "refused on a bound"
            else:
                outcome = "accepted"
            rows.append(
                dict(
                    region=code,
                    n_clusters=r["n_clusters"],
                    year=r["year"],
                    time_slice=r["time_slice"],
                    obs=r["obs"],
                    lo=r["lo"],
                    hi=r["hi"],
                    margin_to_top=r["hi"] - r["obs"],
                    margin_to_bottom=r["obs"] - r["lo"],
                    error=math.sqrt(r["fun"]) if "fun" in r else float("nan"),
                    cls=cls,
                    outcome=outcome,
                    refused=r["refused"],
                )
            )

    print("CONTROL")
    for code, msg in control.items():
        print(f"  {code}: {msg}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "reachability_periods.csv", index=False)
    order_c = ["reachable", "near-miss", "unreachable above", "unreachable below"]
    order_o = ["accepted", "refused on a bound", "refused abnormal"]
    tab = pd.crosstab(df["cls"], df["outcome"]).reindex(
        index=order_c, columns=order_o, fill_value=0
    )
    print("\nALL ROWS: class x outcome")
    print(tab.to_string())
    print("\nPER ROW")
    per = pd.crosstab([df["region"], df["cls"]], df["outcome"]).reindex(
        columns=order_o, fill_value=0
    )
    print(per.to_string())
    ref = df[df["outcome"] != "accepted"]
    n_unr = ref["cls"].str.startswith("unreachable").sum()
    print(
        f"\nrefused: {len(ref)}; unreachable among them: {n_unr} ({n_unr / max(len(ref), 1):.0%})"
    )
    acc = df[df["outcome"] == "accepted"]
    print(
        "accepted unreachable:",
        int(acc["cls"].str.startswith("unreachable").sum()),
        "; accepted near-miss:",
        int((acc["cls"] == "near-miss").sum()),
    )
    print("\nHIGH-ERROR ACCEPTED FITS (error > 1e-6)")
    print(
        acc[acc["error"] > 1e-6][
            [
                "region",
                "n_clusters",
                "year",
                "time_slice",
                "error",
                "cls",
                "margin_to_top",
                "margin_to_bottom",
            ]
        ].to_string(index=False)
    )


def cli(argv: list[str] | None = None) -> None:
    """Parse the recorded command line, ``<out_dir>``, and run :func:`main`."""
    parser = make_parser(__doc__)
    parser.add_argument("out_dir", help="The pass's output directory")
    args = parser.parse_args(argv)
    main(args.out_dir)


if __name__ == "__main__":
    cli()
