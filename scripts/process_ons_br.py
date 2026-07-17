#!/usr/bin/env python
"""Build the ONSBrazilSource inputs from the ONS + ANEEL open data.

Reads (all local, downloaded by the user; all open government data):
    one or more ONS FATOR_CAPACIDADE CSVs (semicolon-delimited, UTF-8) —
        the hourly capacity-factor series; carries coordinates and installed
        capacity itself, so it is the metadata source too
    (optional) ONS RESTRICAO_COFF_EOLICA CSVs — the constrained-off series,
        used to build the curtailment mask (2021+)
    (optional) the ANEEL SIGA CSV (semicolon, latin-1) — commissioning dates
        per plant, joined to ONS complexes through CEG

Writes (under <out>, default input/turbine_level_data/BR/):
    br_md.csv                  ONSBrazilSource metadata contract
    br_fc.csv                  hourly wind FC (id_ons, din_instante, CF)
    br_curtailment_mask.csv    months masked for curtailment (if COFF given)
    join_report.md             complex/coordinate/curtailment report

Finalisation (monthly means, coverage floor, curtailment mask) happens inside
ONSBrazilSource at load time, through the same audited code the tests pin.

    python scripts/process_ons_br.py \\
        --fc input/ons_raw/FATOR_CAPACIDADE-2_2021_*.csv \\
             input/ons_raw/FATOR_CAPACIDADE-2_2022_*.csv \\
             input/ons_raw/FATOR_CAPACIDADE-2_2023_*.csv \\
        --coff input/ons_raw/RESTRICAO_COFF_EOLICA_2021_*.csv \\
        --siga input/ons_raw/siga-empreendimentos-geracao.csv
"""
import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

from vwf.datasets.ons_br import (
    build_br_metadata,
    commissioning_from_siga,
    constrained_off_account,
    curtailment_mask_months,
    wind_complexes_from_fc,
    wind_rows,
)

FC_COLS = ["id_ons", "nom_tipousina", "din_instante", "val_fatorcapacidade"]


def _read_many(patterns, **kwargs) -> pd.DataFrame:
    paths = [p for pat in patterns for p in sorted(glob.glob(pat))]
    if not paths:
        raise FileNotFoundError(f"no files matched {patterns}")
    return pd.concat([pd.read_csv(p, **kwargs) for p in paths], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fc", nargs="+", required=True,
                    help="ONS FATOR_CAPACIDADE CSV glob(s)")
    ap.add_argument("--coff", nargs="+", default=None,
                    help="ONS RESTRICAO_COFF_EOLICA CSV glob(s) (curtailment mask)")
    ap.add_argument("--siga", default=None, help="ANEEL SIGA CSV (commissioning)")
    ap.add_argument("--curtailment-threshold", type=float, default=0.05,
                    help="Mask months whose curtailed fraction exceeds this")
    ap.add_argument("--out", default="input/turbine_level_data/BR")
    ap.add_argument("--height", type=float, default=100.0,
                    help="Uniform hub-height default, m (ONS has no hub height)")
    ap.add_argument("--model", default="Synthetic.Onshore2000",
                    help="Uniform power-curve key (must be a column of your "
                    "power_curves.csv; pick a real-library key for real runs)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- FC: metadata + the hourly series the adapter reads -----------------
    fc = _read_many(args.fc, sep=";")
    complexes = wind_complexes_from_fc(fc)

    commissioning = None
    if args.siga:
        siga = pd.read_csv(args.siga, sep=";", encoding="latin-1", dtype=str)
        commissioning = commissioning_from_siga(siga, fc)

    metadata = build_br_metadata(
        complexes, height=args.height, model=args.model, commissioning=commissioning
    )
    metadata.to_csv(out / "br_md.csv", index=False)

    wind_fc = wind_rows(fc)[FC_COLS]
    wind_fc.to_csv(out / "br_fc.csv", index=False)

    # --- constrained-off: curtailment mask ----------------------------------
    n_masked = 0
    if args.coff:
        coff = _read_many(args.coff, sep=";")
        account = constrained_off_account(coff)
        mask = curtailment_mask_months(account, threshold=args.curtailment_threshold)
        mask.to_csv(out / "br_curtailment_mask.csv", index=False)
        n_masked = len(mask)

    # --- report -------------------------------------------------------------
    with_comm = int(metadata["commissioning_date"].notna().sum()) if len(metadata) else 0
    lines = [
        "# Brazil (ONS) complex report",
        "",
        f"- ONS wind complexes with coordinates: {len(metadata)}",
        f"- total installed capacity: {metadata['capacity'].sum() / 1e6:.1f} GW",
        f"- complexes with a SIGA commissioning date (via CEG): {with_comm}",
        f"- curtailment-masked complex-months (fraction > "
        f"{args.curtailment_threshold}): {n_masked}",
        "",
        "Height and model are UNIFORM DEFAULTS (see *_source columns): "
        f"height={args.height} m, model={args.model}. ONS carries no hub height "
        "or turbine model (vintage-aware assignment is the named follow-up).",
        "Coordinates are the collector-substation location; capacity is the ONS "
        "installed capacity (max over the window).",
        "Pre-2021 CF (no constrained-off series) carries UNSCREENED curtailment.",
        "",
    ]
    (out / "join_report.md").write_text("\n".join(lines))
    print(f"metadata: {len(metadata)} complexes -> {out / 'br_md.csv'}")
    print(f"fc series: {len(wind_fc)} hourly rows -> {out / 'br_fc.csv'}")
    if args.coff:
        print(f"curtailment mask: {n_masked} complex-months -> "
              f"{out / 'br_curtailment_mask.csv'}")
    else:
        print("No --coff given: curtailment mask SKIPPED; observed CF carries "
              "curtailment.", file=sys.stderr)
    print(f"join report -> {out / 'join_report.md'}")


if __name__ == "__main__":
    main()
