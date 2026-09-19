"""What the one-standard-error rule cost, per row, on the untouched test year.

`docs/findings/method-cluster-selection-prereg.md` requires every row to report
the gap between the selection and the minimising count on the test year, so the
rule's cost is visible per row rather than only where someone notices it. **A
gap above 0.002 is a finding about the rule, not about the row.**

Only Denmark offshore gets it free: its minimising count happened also to be
the chapter's, so the study's final run already scored it. Every other row's
final run scored the selection and the two baselines and nothing else, so the
missing counts are trained and scored here.

**This adds a reported number and touches no gate.** It re-selects nothing, and
the selections it reads are already recorded.

The counts come from the recorded fold scores, and are passed explicitly rather
than recomputed, so what was evaluated is visible in the invocation:

    PYVWF_INPUT=input/combined PYVWF_OFFSET_WORKERS=4 PYTHONPATH=src python \\
        scripts/studies/method-cluster-selection/cluster_selection_gaps.py <out_dir> "UK offshore" 50
"""

import importlib.util
import time
from pathlib import Path

import pandas as pd

from vwf.cli.common import make_parser

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "cluster_selection_study", Path(__file__).resolve().parent / "cluster_selection_study.py"
)
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)


def main(out_dir: str, label: str, *counts: str) -> None:
    out = Path(out_dir)
    wanted = tuple(sorted(int(c) for c in counts))
    match = [c for c in study.CONFIGURATIONS if c[0] == label]
    if not match:
        raise SystemExit(
            f"unknown row {label!r}; expected one of {[c[0] for c in study.CONFIGURATIONS]}"
        )
    _, stem, mode, _ = match[0]
    from vwf.harness import regions

    spec = regions.load_region(REPO / "configs" / "regions" / f"{stem}.toml")

    print(
        f"{label}: training {wanted} on {spec.train_years}, scoring {spec.test_years[0]}",
        flush=True,
    )
    started = time.monotonic()
    metrics = study.evaluate_at(
        spec, out, mode, wanted, spec.train_years, int(spec.test_years[0]), "gaps"
    )
    fitted = metrics[metrics["variant"] != "uncorrected"]
    stem_label = label.replace(" ", "_")
    fitted.to_csv(out / f"gaps_{stem_label}.csv", index=False)
    with pd.option_context("display.width", 200):
        print(fitted[["num_clu", "mae", "rmse"]].round(5).to_string(index=False))
    print(f"  {(time.monotonic() - started) / 60:.1f} minutes")
    print(f"written: {out / f'gaps_{stem_label}.csv'}")


def cli(argv: list[str] | None = None) -> None:
    """Parse the recorded command line, ``<out_dir> <label> <count> ...``, and run :func:`main`."""
    parser = make_parser(__doc__)
    parser.add_argument("out_dir", help="The study's output directory, under output/")
    parser.add_argument("label", help='A row of CONFIGURATIONS, e.g. "UK offshore"')
    parser.add_argument("counts", nargs="+", metavar="count", help="Cluster counts to fill in")
    args = parser.parse_args(argv)
    main(args.out_dir, args.label, *args.counts)


if __name__ == "__main__":
    cli()
