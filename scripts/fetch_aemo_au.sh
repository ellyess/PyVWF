#!/usr/bin/env bash
# Fetch the AEMO MMSDM monthly archives for the AU-NEM window (2020-2023).
#
# USER-EXECUTED: downloads go through your connection, you run this yourself.
# 48 months x 2 files: DISPATCH_UNIT_SCADA (~15-22 MB/month, per-DUID 5-min
# SCADAVALUE) and DUDETAILSUMMARY (~0.2 MB/month, effective-dated registered
# capacity). Total ~0.9 GB zipped. Resumable: existing complete files are
# skipped (curl -C -).
#
# Also needed, fetched manually (URL changes quarterly): the AEMO
# "Generation Information" workbook for fuel-type=Wind DUID filtering:
#   https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/nem-forecasting-and-planning/forecasting-and-planning-data/generation-information
#
# Usage:
#   bash scripts/fetch_aemo_au.sh              # into $PYVWF_INPUT or ./input
#   AEMO_OUT=/data/aemo bash scripts/fetch_aemo_au.sh
set -euo pipefail

BASE="https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"
OUT="${AEMO_OUT:-${PYVWF_INPUT:-input}/aemo_raw}"
mkdir -p "$OUT/scada" "$OUT/dudetail"

for year in 2020 2021 2022 2023; do
  for month in 01 02 03 04 05 06 07 08 09 10 11 12; do
    dir="$BASE/$year/MMSDM_${year}_${month}/MMSDM_Historical_Data_SQLLoader/DATA"
    for pair in \
      "scada:PUBLIC_DVD_DISPATCH_UNIT_SCADA_${year}${month}010000.zip" \
      "dudetail:PUBLIC_DVD_DUDETAILSUMMARY_${year}${month}010000.zip"; do
      sub="${pair%%:*}"; file="${pair#*:}"
      target="$OUT/$sub/$file"
      if [ -s "$target" ]; then
        echo "skip   $file (present)"
      else
        echo "fetch  $file"
        curl -fsS -C - --retry 3 -o "$target" "$dir/$file" \
          || { echo "FAILED $file" >&2; exit 1; }
      fi
    done
  done
done

echo "Done: $(ls "$OUT/scada" | wc -l | tr -d ' ') SCADA + $(ls "$OUT/dudetail" | wc -l | tr -d ' ') DUDETAILSUMMARY archives in $OUT"
